"""Transaction fraud model.

Run:  python model.py        (trains, prints metrics, saves model.joblib + model_report.json)

Two models work together:
  * RandomForest (multi-class)  -> fraud probability + which known scam type it looks like
  * IsolationForest (legit only) -> flags transactions that look unlike normal behaviour,
                                    which is how brand-new scam patterns can still be caught
"""
import csv
import json
import os

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split

from data_gen import FEATURES

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data", "transactions.csv")
MODEL_PATH = os.path.join(HERE, "model.joblib")
REPORT_PATH = os.path.join(HERE, "model_report.json")
THRESHOLD = 0.5

TYPE_LABEL = {
    "legit": "Legitimate", "collect_refund": "Collect Request Trap", "qr_swap": "QR Swap",
    "remote_access": "Remote Access Scam", "mule_chain": "Mule Account Chain", "fake_kyc": "Fake KYC / Fee Link",
}

# Human-readable reason for each feature, shown when that feature pushed the score up.
REASONS = {
    "is_collect": lambda t: f"This is a collect request: you would PAY Rs {t['amount']:,.0f}, not receive it",
    "narration_refund": lambda t: "The note says refund/cashback, but money is leaving your account",
    "narration_fee": lambda t: "The note asks for a KYC, registration or verification fee",
    "qr_name_mismatch": lambda t: "The QR's receiver name does not match the shop you are paying",
    "screen_share": lambda t: "A screen-sharing or remote-access app is active during this payment",
    "is_new_payee": lambda t: "You have never paid this receiver before",
    "payee_txn_count": lambda t: "Very little payment history with this receiver",
    "payee_age_days": lambda t: f"The receiver's UPI account is only {t['payee_age_days']:.0f} days old",
    "payee_senders_24h": lambda t: f"{t['payee_senders_24h']:.0f} different people paid this receiver in the last 24 hours (mule-account pattern)",
    "velocity_1h": lambda t: f"{t['velocity_1h']:.0f} payments from you in the last hour (unusually fast)",
    "new_device": lambda t: "This payment is from a new or unrecognised device",
    "from_link": lambda t: "This payment was started from a link you received",
    "amount": lambda t: f"Rs {t['amount']:,.0f} is a large amount for this kind of payment",
    "amount_ratio": lambda t: f"The amount is {t['amount_ratio']:.1f}x your usual payment",
    "hour": lambda t: f"Unusual time of day ({int(t['hour']):02d}:00)",
}


def load(path=DATA):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    X = np.array([[float(r[k]) for k in FEATURES] for r in rows])
    y = np.array([int(r["label"]) for r in rows])
    types = np.array([r["fraud_type"] for r in rows])
    return X, y, types


def binary_metrics(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum()); fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum()); fn = int(((y_true == 1) & (y_pred == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": round(p, 3), "recall": round(r, 3),
            "f1": round(2 * p * r / (p + r), 3) if p + r else 0.0,
            "false_positive_rate": round(fp / (fp + tn), 3) if fp + tn else 0.0}


def fraud_prob(rf, X):
    legit_col = list(rf.classes_).index("legit")
    return 1 - rf.predict_proba(X)[:, legit_col]


def train(path=DATA, save=True):
    X, y, types = load(path)
    Xtr, Xte, ytr, yte, ttr, tte = train_test_split(X, y, types, test_size=0.25, stratify=types, random_state=42)

    rf = RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced_subsample",
                                random_state=42, n_jobs=-1).fit(Xtr, ttr)
    iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=42).fit(Xtr[ytr == 0])

    pred = (fraud_prob(rf, Xte) >= THRESHOLD).astype(int)
    report = {"train_size": int(len(ytr)), "test_size": int(len(yte)), "threshold": THRESHOLD,
              "test_fraud_rate": round(float(yte.mean()), 3), **binary_metrics(yte, pred)}
    report["per_type_recall"] = {str(t): round(float(pred[tte == t].mean()), 3) for t in sorted(set(tte)) if t != "legit"}
    imp = sorted(zip(FEATURES, rf.feature_importances_), key=lambda p: -p[1])
    report["feature_importance"] = [{"feature": f, "importance": round(float(v), 3)} for f, v in imp]

    bundle = {"rf": rf, "iso": iso, "features": FEATURES, "baseline": np.median(Xtr[ytr == 0], axis=0)}
    if save:
        joblib.dump(bundle, MODEL_PATH)
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
    return bundle, report


_cache = {}


def load_model():
    if "m" not in _cache:
        if not os.path.exists(MODEL_PATH):
            train()
        _cache["m"] = joblib.load(MODEL_PATH)
    return _cache["m"]


def score_transaction(txn, bundle=None):
    """txn: dict with the keys in FEATURES (amount_ratio optional). Returns score, type, reasons."""
    m = bundle or load_model()
    t = dict(txn)
    if "amount_ratio" not in t and t.get("user_avg_amount"):
        t["amount_ratio"] = t["amount"] / t["user_avg_amount"]
    t.setdefault("amount_ratio", 1.0)
    x = np.array([[float(t.get(k, 0)) for k in FEATURES]])
    rf, iso = m["rf"], m["iso"]

    prob = float(fraud_prob(rf, x)[0])
    probs = dict(zip(rf.classes_, rf.predict_proba(x)[0]))
    scam = max((c for c in probs if c != "legit"), key=probs.get)

    # explanation: reset one feature at a time to the normal value and see how much the score drops
    drops = []
    for i, name in enumerate(FEATURES):
        x2 = x.copy(); x2[0, i] = m["baseline"][i]
        d = prob - float(fraud_prob(rf, x2)[0])
        if d > 0.02:
            drops.append((d, name))
    drops.sort(reverse=True)
    reasons = [{"feature": n, "impact": round(d, 3), "text": REASONS[n](t)} for d, n in drops[:4]]

    anomaly = float(-iso.decision_function(x)[0])
    return {"fraud_probability": round(prob, 3), "flagged": prob >= THRESHOLD,
            "level": "Low" if prob < .35 else "Medium" if prob < .65 else "High",
            "scam_type": TYPE_LABEL[scam] if prob >= THRESHOLD else "Legitimate",
            "reasons": reasons, "anomaly_score": round(anomaly, 3), "looks_unusual": bool(anomaly > 0)}


if __name__ == "__main__":
    _, rep = train()
    print(f"Train {rep['train_size']} | Test {rep['test_size']} (fraud rate {rep['test_fraud_rate']})")
    print(f"Precision {rep['precision']} | Recall {rep['recall']} | F1 {rep['f1']} | False-positive rate {rep['false_positive_rate']}")
    print(f"TP {rep['tp']}  FP {rep['fp']}  TN {rep['tn']}  FN {rep['fn']}")
    print("Recall by scam type:", rep["per_type_recall"])
    print("Top features:", [(d["feature"], d["importance"]) for d in rep["feature_importance"][:5]])