import csv
import json
import os
import sqlite3
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from detector import analyze, extract_upi_ids, extract_urls, host_of, parse_upi_qr
from tinyfish_intel import enabled as tinyfish_enabled, enrich
from adapt import router as adapt_router
from dashboard import router as dash_router, scored_and_logged

DB = os.path.join(os.path.dirname(__file__), "upishield.db")
CASES_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "demo_cases.csv")

app = FastAPI(title="UPI Shield")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def db():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS reports(
        id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, category TEXT, score INTEGER,
        indicators TEXT, created REAL)""")
    return con


class TextIn(BaseModel):
    text: str
    deep: bool = False


class QrIn(BaseModel):
    payload: str
    deep: bool = False


DEMOS = [
    {"name": "Fake cashback", "text": "Congratulations! Aapne Rs 5000 cashback jeeta hai. Abhi claim karo: http://paytm-reward-claim.xyz/win aur apna UPI PIN enter karo."},
    {"name": "Bank impersonation", "text": "Dear customer, aapka SBI KYC expire ho gaya hai. Account band hone se bachane ke liye turant update karo: bit.ly/sbi-kyc-now"},
    {"name": "Collect request trap", "text": "Aapka Rs 2000 refund aaya hai. Collect request approve karo aur UPI PIN dalo, paise receive ho jayenge. refund.support@ybl"},
    {"name": "Legit message", "text": "Rs 450.00 debited from A/c XX1234 on 12-10-26. Ref No 9876543. Do not share OTP or PIN with anyone. -HDFC Bank"},
]


@app.get("/intel/status")
def intel_status():
    return {"tinyfish": tinyfish_enabled()}


@app.get("/demo")
def demo():
    return DEMOS


@app.post("/analyze")
def analyze_text(body: TextIn):
    res = analyze(body.text)
    return enrich(res) if body.deep else res


@app.post("/analyze-qr")
def analyze_qr(body: QrIn):
    qr = parse_upi_qr(body.payload)
    if not qr["scheme_ok"]:
        res = analyze(body.payload)
        res["qr"] = qr
        res["qr_note"] = "Ye UPI QR nahi hai - sirf link/text ki tarah check kiya."
        return enrich(res) if body.deep else res
    res = analyze(f"{qr['pn']} {qr['tn']}", extra_upi_ids=[qr["pa"]] if qr["pa"] else [])
    res["qr"] = qr
    notes = ["QR scan karke sirf paisa BHEJA jaata hai, receive nahi hota. Koi 'paisa lene ke liye scan karo' bole to scam hai."]
    if qr["am"]:
        notes.append(f"QR mein amount pehle se fixed hai (Rs {qr['am']}) - dhyan se check karo.")
        res["score"] = min(100, res["score"] + 5)
    res["qr_note"] = " ".join(notes)
    return enrich(res) if body.deep else res


@app.post("/report")
def report(body: TextIn):
    res = analyze(body.text)
    ind = sorted(set([host_of(u) for u in res["urls"]] + res["upi_ids"]))
    con = db()
    cur = con.execute("INSERT INTO reports(text,category,score,indicators,created) VALUES(?,?,?,?,?)",
                      (body.text, res["category"], res["score"], json.dumps(ind), time.time()))
    con.commit()
    return {"id": cur.lastrowid, "indicators": ind, "category": res["category"]}


@app.get("/cases")
def cases():
    rows = db().execute("SELECT id,text,category,score,indicators FROM reports ORDER BY id DESC").fetchall()
    return [{**dict(r), "indicators": json.loads(r["indicators"])} for r in rows]


@app.get("/campaigns")
def campaigns():
    """Cytoscape elements: reports joined through shared URLs / UPI IDs (shared by >= 2 reports)."""
    rows = db().execute("SELECT id,category,indicators FROM reports").fetchall()
    by_ind = {}
    for r in rows:
        for i in json.loads(r["indicators"]):
            by_ind.setdefault(i, []).append(r)
    nodes, edges, seen = [], [], set()
    for ind, rs in by_ind.items():
        if len(rs) < 2:
            continue
        nodes.append({"data": {"id": f"i:{ind}", "label": ind, "kind": "indicator"}})
        for r in rs:
            if r["id"] not in seen:
                seen.add(r["id"])
                nodes.append({"data": {"id": f"r{r['id']}", "label": f"Report #{r['id']}", "kind": "report",
                                       "category": r["category"]}})
            edges.append({"data": {"source": f"r{r['id']}", "target": f"i:{ind}"}})
    return {"elements": nodes + edges, "campaigns": sum(1 for rs in by_ind.values() if len(rs) >= 2)}


@app.get("/metrics")
def metrics():
    """Evaluate the detector on data/demo_cases.csv (columns: text,label  where label 1=scam, 0=legit)."""
    if not os.path.exists(CASES_CSV):
        return {"error": "data/demo_cases.csv nahi mila"}
    tp = fp = tn = fn = 0
    errors = []
    with open(CASES_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            truth = int(row["label"])
            pred = 1 if analyze(row["text"])["level"] in ("Medium", "High") else 0
            if truth and pred: tp += 1
            elif truth and not pred: fn += 1; errors.append({"type": "missed scam", "text": row["text"]})
            elif not truth and pred: fp += 1; errors.append({"type": "false alarm", "text": row["text"]})
            else: tn += 1
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    fpr = fp / (fp + tn) if fp + tn else 0
    return {"total": tp + fp + tn + fn, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3),
            "false_positive_rate": round(fpr, 3), "errors": errors}

@app.post("/score-txn")
def score_txn(txn: dict):
    """Score one transaction. Body = dict of features."""
    return scored_and_logged(txn)


app.include_router(adapt_router)
app.include_router(dash_router)

# Serve the web app from the same server (keep this at the very end so API routes win)
FRONTEND = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(FRONTEND):
    app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")