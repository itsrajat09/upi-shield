"""TinyFish evidence layer.

Fetch API  -> opens a suspicious link in TinyFish's remote browser and reads the page
Search API -> looks for existing scam reports about a domain or UPI ID
Uses only the standard library. Key is read from TINYFISH_API_KEY (env var or backend/.env).
"""
import json
import os
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from detector import BRANDS, OFFICIAL, SHORTENERS, host_of

FETCH_URL = "https://api.fetch.tinyfish.ai"
SEARCH_URL = "https://api.search.tinyfish.ai"


def _load_env():
    path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env()


def enabled():
    return bool(os.getenv("TINYFISH_API_KEY"))


def _call(url, body=None, timeout=20):
    headers = {"X-API-Key": os.environ["TINYFISH_API_KEY"], "Content-Type": "application/json"}
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


PAGE_RULES = [
    (30, "asks for your UPI PIN", r"upi\s*pin|enter.{0,15}\bpin\b"),
    (25, "asks for OTP, CVV or card number", r"\botp\b|\bcvv\b|card\s*number|debit\s*card"),
    (15, "asks for Aadhaar, PAN or a password", r"aadhaar|\bpan\b|password"),
    (15, "pushes KYC, cashback or reward claims", r"\bkyc\b|cashback|reward|claim"),
]
BAD_WORDS = re.compile(r"scam|fraud|fake|phishing|cheat|complaint|beware|warning", re.I)


def page_check(url):
    """Open the URL via TinyFish Fetch and look for scam signals in the page text."""
    full = url if "://" in url else "http://" + url
    try:
        data = _call(FETCH_URL, {"urls": [full], "format": "markdown", "ttl": 0})
    except Exception as e:
        return {"url": url, "error": f"TinyFish request failed ({str(e)[:80]})", "evidence": []}
    if not data.get("results"):
        return {"url": url, "error": "this link does not open. Scam links are often taken down quickly, so be careful", "evidence": []}
    page = data["results"][0]
    text = (str(page.get("title", "")) + " " + str(page.get("text", ""))).lower()[:20000]
    ev = [(w, f"TinyFish page check: the page {reason}") for w, reason, rx in PAGE_RULES if re.search(rx, text)]
    host = host_of(url)
    brand = next((b for b in BRANDS if re.search(rf"\b{b}\b", text)), None)
    if brand and not host.endswith(OFFICIAL):
        ev.append((25, f"TinyFish page check: the page uses the '{brand}' brand but '{host}' is not an official domain"))
    return {"url": url, "title": page.get("title", ""), "evidence": ev}


def web_reports(indicator):
    """Search the web for scam reports that mention this domain or UPI ID."""
    q = f'"{indicator}" scam OR fraud OR complaint'
    try:
        data = _call(SEARCH_URL + "?" + urllib.parse.urlencode({"query": q}))
    except Exception as e:
        return {"indicator": indicator, "error": str(e)[:80], "count": 0, "sources": []}
    hits = []
    for r in data.get("results", []):
        blob = " ".join(str(r.get(k) or "") for k in ("title", "snippet", "description", "url")).lower()
        if indicator.lower() in blob and BAD_WORDS.search(blob):
            hits.append({"title": r.get("title", ""), "url": r.get("url", "")})
    return {"indicator": indicator, "count": len(hits), "sources": hits[:3]}


def enrich(res):
    """Add TinyFish evidence to an /analyze result and re-score it."""
    intel = {"enabled": enabled(), "pages": [], "reports": []}
    res["intel"] = intel
    if not intel["enabled"]:
        return res

    urls = res["urls"][:2]
    hosts = [h for h in dict.fromkeys(host_of(u) for u in urls) if h and h not in SHORTENERS and not h.endswith(OFFICIAL)]
    indicators = hosts[:2] + res["upi_ids"][:2]
    with ThreadPoolExecutor(4) as ex:
        pf = [ex.submit(page_check, u) for u in urls]
        sf = [ex.submit(web_reports, i) for i in indicators]
        pages = [f.result() for f in pf]
        reports = [f.result() for f in sf]

    added, kind = [], None
    for p in pages:
        for w, reason in p.pop("evidence"):
            added.append((w, reason))
            kind = "Phishing Page"
    page_pts = min(50, sum(w for w, _ in added))
    for r in reports:
        if r["count"] >= 2:
            added.append((25, f"TinyFish web search: '{r['indicator']}' appears in {r['count']} scam-related web results"))
            kind = kind or "Reported Scam"
        elif r["count"] == 1:
            added.append((10, f"TinyFish web search: '{r['indicator']}' appears in 1 scam-related web result"))
    intel["pages"], intel["reports"] = pages, reports
    if not added:
        return res

    # page signals are capped so one noisy page cannot dominate the score
    report_pts = sum(w for w, reason in added if reason.startswith("TinyFish web search"))
    res["evidence"] += [{"type": "tinyfish", "weight": w, "reason": reason} for w, reason in added]
    res["evidence"].sort(key=lambda e: -abs(e["weight"]))
    res["score"] = max(0, min(100, res["score"] + page_pts + report_pts))
    res["level"] = "Low" if res["score"] < 30 else "Medium" if res["score"] < 60 else "High"
    if res["category"] in ("Safe", "Suspicious") and res["score"] >= 30 and kind:
        res["category"] = kind
        res["next_step"] = ("Do not enter any details on this page. Report it and delete the message."
                            if kind == "Phishing Page" else
                            "Other people have reported this. Do not pay or click until you verify the sender.")
    return res