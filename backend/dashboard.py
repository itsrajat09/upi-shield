"""Analyst dashboard backend.

* Every transaction scored through /score-txn is logged here (so live demos show up on the dashboard).
* /dash/seed fills the dashboard with a SYNTHETIC history (14 days) taken from the held-out test data,
  plus a burst of the new 'fake investment' scam in the last 4 days.
* /dash/rescore scores the same history again with the current (maybe adapted) model.
"""
import json
import os
import sqlite3
import time
from collections import Counter, defaultdict

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel
from sklearn.model_selection import train_test_split

import adapt
import model as M
from data_gen import FEATURES

DB = os.path.join(os.path.dirname(__file__), "upishield.db")
UNSEEN = adapt.UNSEEN
router = APIRouter()


def _con():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS txns(
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, amount REAL, prob REAL, flagged INTEGER,
        category TEXT, level TEXT, reasons TEXT, features TEXT, truth TEXT, status TEXT DEFAULT 'open')""")
    return con


def _row(txn, res, ts, truth):
    reasons = [r["text"] for r in res.get("reasons", [])]
    return (ts, float(txn.get("amount", 0)), float(res["fraud_probability"]), int(res["flagged"]),
            res["scam_type"], res["level"], json.dumps(reasons), json.dumps(txn), truth)


_INS = "INSERT INTO txns(ts,amount,prob,flagged,category,level,reasons,features,truth) VALUES(?,?,?,?,?,?,?,?,?)"


def scored_and_logged(txn):
    """Use this in /score-txn: score the payment, remember it for the dashboard."""
    res = adapt.score_full(txn)
    con = _con()
    con.execute(_INS, _row(txn, res, time.time(), "live"))
    con.commit()
    con.close()
    return res


def _score_many(X):
    """Fast batch scoring. The slow explanation step only runs for flagged rows."""
    adapt._init()
    m = M.load_model()
    prob = M.fraud_prob(m["rf"], X)
    novel = adapt._state["nov"].score(X) >= adapt._state["nov"].thr
    out = []
    for i, x in enumerate(X):
        txn = {k: float(v) for k, v in zip(FEATURES, x)}
        if prob[i] >= M.THRESHOLD or novel[i]:
            out.append((txn, adapt.score_full(txn)))
        else:
            out.append((txn, {"fraud_probability": round(float(prob[i]), 3), "flagged": False, "level": "Low",
                              "scam_type": "Legitimate", "reasons": []}))
    return out


def _history():
    """Held-out known-scam test rows + a burst of the new scam. Returns (X, truth types, age_limit_days)."""
    X, y, t = M.load()
    _, Xte, _, _, _, tte = train_test_split(X, y, t, test_size=0.25, stratify=t, random_state=42)
    rng = np.random.default_rng(3)
    pick = rng.choice(len(Xte), size=500, replace=False)
    Xu, yu, tu = M.load(UNSEEN)
    fr = np.where(yu == 1)[0]
    lg = np.where(yu == 0)[0]
    npick = np.concatenate([rng.choice(fr, 24, replace=False), rng.choice(lg, 24, replace=False)])
    Xall = np.vstack([Xte[pick], Xu[npick]])
    truth = list(tte[pick]) + list(tu[npick])
    limit = [14] * len(pick) + [4] * len(npick)
    return Xall, truth, limit, rng


def _stamp(rng, hour, limit_days):
    now = time.time()
    midnight = now - (now % 86400)            # UTC midnight is fine for a demo
    d = int(rng.integers(0, limit_days))
    return min(midnight - d * 86400 + int(hour) * 3600 + int(rng.integers(0, 3600)), now - 60)


@router.post("/dash/seed")
def seed():
    X, truth, limit, rng = _history()
    results = _score_many(X)
    rows = []
    for (txn, res), tr, lim in zip(results, truth, limit):
        rows.append(_row(txn, res, _stamp(rng, txn["hour"], lim), tr))
    con = _con()
    con.execute("DELETE FROM txns WHERE truth != 'live'")
    con.executemany(_INS, rows)
    con.commit()
    con.close()
    return {"seeded": len(rows)}


@router.post("/dash/rescore")
def rescore():
    """Score the stored history again with the current model (e.g. after adaptation). Keeps timestamps."""
    con = _con()
    old = con.execute("SELECT id,features FROM txns WHERE truth != 'live' ORDER BY id").fetchall()
    if not old:
        return {"rescored": 0}
    X = np.array([[float(json.loads(r["features"]).get(k, 0)) for k in FEATURES] for r in old])
    results = _score_many(X)
    upd = []
    for r, (txn, res) in zip(old, results):
        upd.append((res["fraud_probability"], int(res["flagged"]), res["scam_type"], res["level"],
                    json.dumps([x["text"] for x in res.get("reasons", [])]), r["id"]))
    con.executemany("UPDATE txns SET prob=?, flagged=?, category=?, level=?, reasons=?, status='open' WHERE id=?", upd)
    con.commit()
    con.close()
    return {"rescored": len(upd)}


@router.get("/dash/summary")
def summary(days: int = 14):
    con = _con()
    rows = con.execute("SELECT ts,amount,flagged,category,truth FROM txns").fetchall()
    con.close()
    total = len(rows)
    flagged = [r for r in rows if r["flagged"]]
    by_cat = Counter(r["category"] for r in flagged)

    now = time.time()
    midnight = now - (now % 86400)
    keys = [time.strftime("%Y-%m-%d", time.gmtime(midnight - d * 86400)) for d in range(days - 1, -1, -1)]
    per_day = {k: Counter() for k in keys}
    for r in flagged:
        k = time.strftime("%Y-%m-%d", time.gmtime(r["ts"]))
        if k in per_day:
            per_day[k][r["category"]] += 1

    lab = [r for r in rows if r["truth"] != "live"]
    tp = sum(1 for r in lab if r["flagged"] and r["truth"] != "legit")
    fp = sum(1 for r in lab if r["flagged"] and r["truth"] == "legit")
    fn = sum(1 for r in lab if not r["flagged"] and r["truth"] != "legit")
    tn = sum(1 for r in lab if not r["flagged"] and r["truth"] == "legit")
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    fpr = fp / (fp + tn) if fp + tn else 0
    per_type = defaultdict(lambda: [0, 0, 0])   # caught, total, caught WITH a scam name (not just "unusual")
    for r in lab:
        if r["truth"] != "legit":
            per_type[r["truth"]][1] += 1
            per_type[r["truth"]][0] += int(r["flagged"])
            per_type[r["truth"]][2] += int(r["flagged"] and not r["category"].startswith("Unusual"))
    return {"total": total, "flagged": len(flagged), "amount_at_risk": round(sum(r["amount"] for r in flagged)),
            "by_category": dict(by_cat),
            "by_day": [{"day": k, "counts": dict(per_day[k])} for k in keys],
            "labelled": len(lab), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(prec, 3), "recall": round(rec, 3), "false_positive_rate": round(fpr, 3),
            "recall_by_type": {t: {"caught": c, "total": n, "named": nm, "recall": round(c / n, 3)} for t, (c, n, nm) in per_type.items()}}


@router.get("/dash/alerts")
def alerts(limit: int = 60, category: str = "", status: str = ""):
    q, args = "SELECT id,ts,amount,prob,category,level,reasons,status,truth FROM txns WHERE flagged=1", []
    if category:
        q += " AND category=?"; args.append(category)
    if status:
        q += " AND status=?"; args.append(status)
    q += " ORDER BY ts DESC LIMIT ?"; args.append(max(1, min(limit, 300)))
    con = _con()
    rows = con.execute(q, args).fetchall()
    con.close()
    return [{**dict(r), "reasons": json.loads(r["reasons"])} for r in rows]


class StatusIn(BaseModel):
    status: str


@router.post("/dash/alerts/{aid}")
def set_status(aid: int, body: StatusIn):
    if body.status not in ("open", "confirmed", "false_alarm"):
        return {"error": "bad status"}
    con = _con()
    con.execute("UPDATE txns SET status=? WHERE id=?", (body.status, aid))
    con.commit()
    con.close()
    return {"id": aid, "status": body.status}