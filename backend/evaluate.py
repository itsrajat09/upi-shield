"""Honest evaluation of every part of UPI Shield (synthetic data only).

Run:  python evaluate.py
Writes: evaluation_report.md  (copy the tables into your report)
        evaluation_report.json

What is measured
  A. Transaction model, standard test        (same kind of data it was trained on: optimistic)
  B. Transaction model, 5-fold cross-validation  (mean +- std, shows stability)
  C. Transaction model, STRESS test          (harder data: more stealth scams + more odd-but-legit payments)
  D. Threshold trade-off on the stress test  (precision vs recall vs false alarms)
  E. Message / link detector                 (125-message dev set + 50-message held-out set)
  F. Adaptivity to a brand-new scam          (before / novelty alert / after retraining)
"""
import csv
import json
import os

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import adapt
import model as M
from data_gen import FEATURES, FRAUD_TYPES, make_fraud, make_legit
from detector import analyze

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
SIGNATURE = ["is_collect", "qr_name_mismatch", "screen_share", "narration_refund", "narration_fee", "from_link"]


def new_rf():
    return RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced_subsample",
                                  random_state=42, n_jobs=-1)


def to_xy(rows):
    X = np.array([[float(r[k]) for k in FEATURES] for r in rows])
    return X, np.array([int(r["label"]) for r in rows]), np.array([r["fraud_type"] for r in rows])


def make_stress(n=3000, rate=0.12, seed=99):
    """Harder than the training data: 35% of scams hide their signature (look like normal payments),
    12% of legitimate payments look suspicious (new payee, young account, many senders), 5% are big."""
    rng = np.random.default_rng(seed)
    rows = []
    for kind in rng.choice(FRAUD_TYPES, size=int(n * rate)):
        r = make_fraud(rng, kind)
        if rng.random() < 0.35:
            for c in SIGNATURE:
                r[c] = 0
            r["payee_age_days"] = int(rng.integers(150, 2000))
            r["payee_senders_24h"] = int(rng.integers(0, 8))
        rows.append(r)
    for _ in range(n - int(n * rate)):
        r = make_legit(rng)
        if rng.random() < 0.12:
            r.update(is_new_payee=1, payee_txn_count=0, payee_age_days=int(rng.integers(1, 45)),
                     payee_senders_24h=int(rng.integers(4, 30)))
        if rng.random() < 0.05:
            r["amount"] = round(r["amount"] * 4, 2)
            r["amount_ratio"] = round(r["amount_ratio"] * 4, 2)
        rows.append(r)
    rng.shuffle(rows)
    return to_xy(rows)


def pct(v):
    return f"{v * 100:.1f}%"


def text_eval(path):
    tp = fp = tn = fn = 0
    missed, false_alarms = [], []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            y = int(row["label"])
            p = analyze(row["text"])["level"] in ("Medium", "High")
            if y and p: tp += 1
            elif y: fn += 1; missed.append(row["text"])
            elif p: fp += 1; false_alarms.append(row["text"])
            else: tn += 1
    m = M.binary_metrics(np.array([1] * (tp + fn) + [0] * (fp + tn)),
                         np.array([1] * tp + [0] * fn + [1] * fp + [0] * tn))
    return {**m, "total": tp + fp + tn + fn, "missed": missed, "false_alarms": false_alarms}


def main():
    out = {}
    X, y, t = M.load()

    # A. standard test (the split used by model.py)
    _, rep = M.train(save=False)
    out["A_standard"] = {k: rep[k] for k in ("test_size", "precision", "recall", "f1", "false_positive_rate", "tp", "fp", "tn", "fn")}
    out["A_standard"]["per_type_recall"] = rep["per_type_recall"]

    # B. 5-fold cross-validation
    folds = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, t):
        rf = new_rf().fit(X[tr], t[tr])
        folds.append(M.binary_metrics(y[te], (M.fraud_prob(rf, X[te]) >= M.THRESHOLD).astype(int)))
    out["B_cv"] = {k: [round(float(np.mean([f[k] for f in folds])), 3), round(float(np.std([f[k] for f in folds])), 3)]
                   for k in ("precision", "recall", "f1", "false_positive_rate")}

    # C + D. stress test, trained on the normal training file
    rf = new_rf().fit(X, t)
    Xs, ys, ts = make_stress()
    p = M.fraud_prob(rf, Xs)
    c = M.binary_metrics(ys, (p >= M.THRESHOLD).astype(int))
    c["roc_auc"] = round(float(roc_auc_score(ys, p)), 3)
    c["pr_auc"] = round(float(average_precision_score(ys, p)), 3)
    c["test_size"] = int(len(ys))
    c["fraud_rate"] = round(float(ys.mean()), 3)
    c["per_type_recall"] = {k: round(float(((p >= M.THRESHOLD)[ts == k]).mean()), 3) for k in FRAUD_TYPES}
    out["C_stress"] = c
    out["D_threshold"] = [{"threshold": th, **M.binary_metrics(ys, (p >= th).astype(int))} for th in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)]

    # E. message detector
    out["E_text_dev"] = text_eval(os.path.join(DATA, "demo_cases.csv"))
    hold = os.path.join(DATA, "holdout_cases.csv")
    if os.path.exists(hold):
        out["E_text_holdout"] = text_eval(hold)

    # F. adaptivity
    out["F_adapt"] = adapt.run_adaptation(k=30, apply=False)

    with open(os.path.join(HERE, "evaluation_report.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    md = markdown(out)
    with open(os.path.join(HERE, "evaluation_report.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print("\nSaved evaluation_report.md and evaluation_report.json in the backend folder.")


def row(name, m):
    return f"| {name} | {pct(m['precision'])} | {pct(m['recall'])} | {pct(m['false_positive_rate'])} | {m.get('tp', '')}/{m.get('fp', '')}/{m.get('tn', '')}/{m.get('fn', '')} |"


def markdown(o):
    L = []
    hdr = "| Test | Precision | Recall | False-positive rate | TP/FP/TN/FN |\n|---|---|---|---|---|"
    A, B, C = o["A_standard"], o["B_cv"], o["C_stress"]
    L += ["# UPI Shield evaluation (synthetic data)", "", "## 1. Transaction model", "", hdr,
          row(f"A. Standard test ({A['test_size']} txns)", A),
          f"| B. 5-fold CV (mean, std in brackets) | {pct(B['precision'][0])} ({pct(B['precision'][1])}) | {pct(B['recall'][0])} ({pct(B['recall'][1])}) | {pct(B['false_positive_rate'][0])} ({pct(B['false_positive_rate'][1])}) | |",
          row(f"C. Stress test ({C['test_size']} txns, {pct(C['fraud_rate'])} fraud)", C), "",
          f"Stress test ROC-AUC {C['roc_auc']}, PR-AUC {C['pr_auc']}.",
          "The stress test hides the signature of 35% of scams and makes 12% of legitimate payments look suspicious, so it is the number to quote as the honest one.", "",
          "### Recall by scam type (stress test)", "", "| Scam | Recall |", "|---|---|"]
    L += [f"| {k} | {pct(v)} |" for k, v in C["per_type_recall"].items()]
    L += ["", "### Threshold trade-off (stress test)", "", "| Threshold | Precision | Recall | False-positive rate |", "|---|---|---|---|"]
    L += [f"| {d['threshold']} | {pct(d['precision'])} | {pct(d['recall'])} | {pct(d['false_positive_rate'])} |" for d in o["D_threshold"]]
    L += ["", "## 2. Message / link detector", "", hdr, row(f"Dev set ({o['E_text_dev']['total']} messages, rules were tuned on this)", o["E_text_dev"])]
    if "E_text_holdout" in o:
        h = o["E_text_holdout"]
        L += [row(f"Held-out set ({h['total']} new messages, never used for tuning)", h), "", "Missed scams on the held-out set:"]
        L += [f"- {m}" for m in h["missed"]] or ["- none"]
        L += ["", "False alarms on the held-out set:"] + ([f"- {m}" for m in h["false_alarms"]] or ["- none"])
    F = o["F_adapt"]
    L += ["", "## 3. Adapting to a brand-new scam (fake investment)", "", hdr,
          row("Before (old model)", F["before"]), row("Novelty alert only", F["novelty_alert"]),
          row(f"After retraining on {F['labels_used']} analyst labels", F["after"]),
          row("Known 5 scams, before", F["known_scams_before"]), row("Known 5 scams, after", F["known_scams_after"]), ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()