"""Adaptivity: catch a brand-new scam type, then learn it from a few analyst labels.

Story in 4 steps (all on synthetic data):
  1. BEFORE : the trained model has never seen 'fake investment' scams, so it misses them.
  2. DETECT : a novelty detector (trained on normal behaviour only) says "this looks unlike normal".
  3. LABEL  : an analyst reviews the top alerts and labels them (we simulate the analyst).
  4. LEARN  : the model is retrained with those few labels, and now knows the new scam by name.

Run in terminal:  python adapt.py
Also exposes API routes (/adapt/run, /adapt/reset, /adapt/status) and `score_full()`.
"""
import os

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import model as M
from data_gen import FEATURES

HERE = os.path.dirname(__file__)
UNSEEN = os.path.join(HERE, "..", "data", "unseen_investment_scam.csv")
NEW_TYPE = "fake_investment"
M.TYPE_LABEL[NEW_TYPE] = "Fake Investment Scam"

# Novelty uses only behaviour of the payee/payer. The yes/no scam flags (is_collect, ...) are left out
# on purpose: a NEW scam has no known flag, and those columns just add noise to the "is this normal?" test.
BEHAV = ["payee_txn_count", "payee_age_days", "payee_senders_24h", "velocity_1h"]
IDX = [FEATURES.index(c) for c in BEHAV]


class Novelty:
    """Distance from 'normal' behaviour (Mahalanobis distance on log-scaled features)."""

    def __init__(self, q=99):
        self.q = q

    def _z(self, X):
        return np.log1p(X[:, IDX])

    def fit(self, X_legit):
        Z = self._z(X_legit)
        self.mu = Z.mean(0)
        self.P = np.linalg.pinv(np.cov(Z, rowvar=False))
        self.thr = float(np.percentile(self.score(X_legit), self.q))   # ~1% of normal traffic crosses this
        return self

    def score(self, X):
        D = self._z(X) - self.mu
        return np.sqrt(np.einsum("ij,jk,ik->i", D, self.P, D))


_state = {"nov": None, "orig": None, "adapted": False}


def _init():
    if _state["nov"] is None:
        X, y, _ = M.load()
        _state["nov"] = Novelty().fit(X[y == 0])
        _state["orig"] = M.load_model()


def _vec(txn):
    t = dict(txn)
    if "amount_ratio" not in t and t.get("user_avg_amount"):
        t["amount_ratio"] = t["amount"] / t["user_avg_amount"]
    t.setdefault("amount_ratio", 1.0)
    return np.array([[float(t.get(k, 0)) for k in FEATURES]])


def score_full(txn):
    """Known-scam model first; if it says 'legit' but behaviour is very unusual, raise a REVIEW alert."""
    _init()
    res = M.score_transaction(txn)
    nov = float(_state["nov"].score(_vec(txn))[0])
    res["novelty_score"] = round(nov, 2)
    res["novelty_threshold"] = round(_state["nov"].thr, 2)
    res["adapted"] = _state["adapted"]
    if not res["flagged"] and nov >= _state["nov"].thr:
        res.update(flagged=True, level="Medium", scam_type="Unusual pattern (possible new scam)",
                   reasons=[{"feature": "novelty", "impact": round(nov / _state["nov"].thr, 2),
                             "text": "This payment pattern is very unlike normal behaviour (fresh receiver, many senders, rapid payments)"}])
    return res


def _rates(y, flag):
    return M.binary_metrics(y, flag.astype(int))


def run_adaptation(k=30, apply=True, seed=7):
    """Do the 4-step story. k = how many alerts the analyst reviews and labels."""
    _init()
    nov, old = _state["nov"], _state["orig"]
    X, y, t = M.load()
    Xtr, Xte, ytr, yte, ttr, tte = train_test_split(X, y, t, test_size=0.25, stratify=t, random_state=42)

    Xu, yu, _ = M.load(UNSEEN)
    Xpool, Xtest, ypool, ytest = train_test_split(Xu, yu, test_size=0.5, stratify=yu, random_state=seed)

    # 1. BEFORE: old model on the unseen scam (held-out test half)
    before = _rates(ytest, M.fraud_prob(old["rf"], Xtest) >= M.THRESHOLD)
    # 2. DETECT: novelty alerts
    novel_alert = nov.score(Xtest) >= nov.thr
    detect = _rates(ytest, novel_alert)

    # 3. LABEL: analyst reviews the k most unusual transactions in the pool
    order = np.argsort(-nov.score(Xpool))[:k]
    Xl, yl = Xpool[order], ypool[order]
    labels = np.where(yl == 1, NEW_TYPE, "legit")

    # 4. LEARN: retrain with the known data + the k analyst labels
    rf = RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced_subsample",
                                random_state=42, n_jobs=-1).fit(np.vstack([Xtr, Xl]), np.concatenate([ttr, labels]))
    rf.n_jobs = 1   # single-row scoring is faster without threads
    after = _rates(ytest, M.fraud_prob(rf, Xtest) >= M.THRESHOLD)
    known_before = _rates(yte, M.fraud_prob(old["rf"], Xte) >= M.THRESHOLD)
    known_after = _rates(yte, M.fraud_prob(rf, Xte) >= M.THRESHOLD)

    if apply:
        _state["adapted"] = True
        M._cache["m"] = {**old, "rf": rf}
    return {"labels_used": int(k), "labelled_fraud": int((yl == 1).sum()), "labelled_legit": int((yl == 0).sum()),
            "test_size": int(len(ytest)), "test_fraud": int(ytest.sum()),
            "before": before, "novelty_alert": detect, "after": after,
            "known_scams_before": known_before, "known_scams_after": known_after}


def reset():
    _init()
    M._cache["m"] = _state["orig"]
    _state["adapted"] = False


router = APIRouter()


class RunIn(BaseModel):
    k: int = 30


@router.get("/adapt/status")
def status():
    _init()
    return {"adapted": _state["adapted"]}


@router.post("/adapt/run")
def api_run(body: RunIn):
    return run_adaptation(k=max(5, min(200, body.k)))


@router.post("/adapt/reset")
def api_reset():
    reset()
    return {"adapted": False}


if __name__ == "__main__":
    r = run_adaptation(apply=False)
    pr = lambda n, m: print(f"{n:<26} precision {m['precision']:.3f} | recall {m['recall']:.3f} | false-positive rate {m['false_positive_rate']:.3f}")
    print(f"Unseen scam test set: {r['test_size']} transactions, {r['test_fraud']} are fraud. Analyst labels used: {r['labels_used']}")
    pr("1. Before (old model)", r["before"])
    pr("2. Novelty alert only", r["novelty_alert"])
    pr("4. After retraining", r["after"])
    pr("Known scams, before", r["known_scams_before"])
    pr("Known scams, after", r["known_scams_after"])