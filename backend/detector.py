"""UPI Shield detection engine: rules (English + Hinglish) + link/UPI ID checks."""
import re
from urllib.parse import urlparse

SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "cutt.ly", "rb.gy", "is.gd", "shorturl.at", "tiny.cc"}
BAD_TLDS = {"xyz", "top", "click", "icu", "site", "online", "buzz", "live", "shop", "vip", "link", "work", "cfd"}
BRANDS = ["sbi", "hdfc", "icici", "axis", "paytm", "phonepe", "gpay", "googlepay", "npci", "kotak", "pnb", "bhim"]
OFFICIAL = ("sbi.co.in", "onlinesbi.sbi", "sbi.bank.in", "hdfcbank.com", "icicibank.com", "axisbank.com",
            "paytm.com", "phonepe.com", "npci.org.in", "pay.google.com", "kotak.com", "pnbindia.in", "bhimupi.org.in")

# (id, weight, category, reason, regex)
RULES = [
    ("credential", 40, "Credential Theft",
     "UPI PIN / OTP / CVV / password maanga ja raha hai (asli bank kabhi nahi maangta)",
     r"((share|send|bhej\w*|bata\w*|batao|enter|dal\w*|daal\w*|confirm|verify)\W+(?:\w+\W+){0,3}(upi\s*pin|pin|otp|cvv|password)|(upi\s*pin|pin|otp|cvv|password)\W+(?:\w+\W+){0,3}(share|send|bhej\w*|bata\w*|batao|enter|dal\w*|daal\w*|confirm))"),
    ("collect", 40, "Collect Request Trap",
     "Collect request / approve karne ko bola ja raha hai - isse paisa aata nahi, katata hai",
     r"(collect\s*request|request\s*(approve|accept)|approve\s*kar|accept\s*kar|refund.{0,40}(pin|approve|accept)|paise\s*(lene|receive).{0,30}pin)"),
    ("cashback", 25, "Fake Cashback / Reward",
     "Cashback / reward / lottery ka lalach diya ja raha hai",
     r"(cashback|reward\s*points?|lucky\s*draw|lottery|jeet\w*|you\s*have\s*won|prize|inaam|scratch\s*card|free\s*gift)"),
    ("kyc", 28, "Bank Impersonation",
     "KYC / account band hone ki dhamki - bank impersonation ka common pattern",
     r"(kyc|account.{0,20}(block|band|suspend|freez)|(card|account|pan).{0,15}(expire|expired)|update\s*(now|kar))"),
    ("urgency", 15, "Pressure Tactic",
     "Urgency / jaldi karne ka pressure",
     r"(urgent|immediately|turant|jaldi|abhi\s+\w+|within\s*\d+\s*(min|hour|hr)|last\s*chance|24\s*(ghante|hours))"),
    ("remote", 35, "Remote Access Scam",
     "Screen-share / remote access app install karne ko bola ja raha hai",
     r"(anydesk|teamviewer|quicksupport|rustdesk|screen\s*shar\w*)"),
    ("apk", 30, "Malware Link", "APK file install karne ko bola ja raha hai", r"\.apk\b"),
    ("safe_ctx", -25, "Safe",
     "'Kisi ke saath share na karein' jaisi asli bank warning mili",
     r"(do\s*not\s*share|never\s*share|don'?t\s*share|share\s*na\s*kar\w*|share\s*mat)"),
    ("txn_alert", -10, "Safe",
     "Normal transaction alert jaisa format", r"(debited|credited|a/c\s*(no|x+)|ref\s*no|txn\s*id)"),
]

URL_RE = re.compile(
    r"(?:https?://|www\.)[^\s<>\"']+|(?<![@\w.-])(?:[a-z0-9-]+\.)+(?:com|in|net|org|xyz|top|click|icu|site|online|buzz|live|shop|vip|link|work|cfd|co\.in|ly|me|co)\b(?:/[^\s<>\"']*)?",
    re.I)
UPI_RE = re.compile(r"\b[a-z0-9._-]{2,}@[a-z]{2,}\b", re.I)

NEXT_STEPS = {
    "Credential Theft": "UPI PIN / OTP kisi ko mat batao. Bank ke official number par call karke confirm karo.",
    "Collect Request Trap": "Collect request approve mat karo. Paisa receive karne ke liye PIN ki zaroorat nahi hoti.",
    "Fake Cashback / Reward": "Link par click mat karo. Offer sirf official app ke andar verify karo.",
    "Bank Impersonation": "Link ignore karo. Bank app ya branch se KYC status check karo.",
    "Remote Access Scam": "App install mat karo. Agar kiya hai to turant uninstall karke bank ko call karo.",
    "Malware Link": "APK install mat karo. Sirf Play Store se app lo.",
    "Pressure Tactic": "Ruko aur soch lo. Scammer jaldi ka pressure banate hain.",
    "Suspicious Link": "Link mat kholo. Official website khud type karke jao.",
    "Suspicious UPI ID": "Is UPI ID par payment mat bhejo jab tak receiver verify na ho.",
}


def extract_urls(text):
    urls = []
    for m in URL_RE.finditer(text):
        u = m.group(0).rstrip(".,);")
        if u not in urls:
            urls.append(u)
    return urls


def extract_upi_ids(text):
    return list(dict.fromkeys(m.group(0).lower() for m in UPI_RE.finditer(text)))


def host_of(url):
    p = urlparse(url if "://" in url else "http://" + url)
    return (p.hostname or "").lower()


def check_url(url):
    """Return list of (weight, reason) for one URL."""
    out, host = [], host_of(url)
    if not host:
        return out
    official = host.endswith(OFFICIAL)
    if host in SHORTENERS:
        out.append((20, f"Shortened link ({host}) - asli destination chhupi hai"))
    if re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host):
        out.append((30, "Link domain ki jagah IP address use kar raha hai"))
    tld = host.rsplit(".", 1)[-1]
    if tld in BAD_TLDS and not official:
        out.append((20, f"Suspicious domain ending (.{tld})"))
    brand = next((b for b in BRANDS if b in host), None)
    if brand and not official:
        out.append((35, f"Lookalike domain: '{host}' bank/UPI brand ({brand}) ki nakal lag raha hai"))
    if host.count("-") >= 2:
        out.append((10, "Domain mein bahut saare hyphens (fake domains mein common)"))
    if url.lower().startswith("http://"):
        out.append((5, "Link secure (HTTPS) nahi hai"))
    return out


def check_upi_id(upi):
    out = []
    handle = upi.split("@")[0]
    if any(b in handle for b in BRANDS):
        out.append((25, f"UPI ID '{upi}' mein bank/brand ka naam hai - asli bank aise ID se paisa nahi maangta"))
    if re.search(r"(refund|support|care|help|kyc|reward|cashback|lottery|official)", handle):
        out.append((15, f"UPI ID '{upi}' mein 'support/refund' jaisa bharosa dilane wala shabd hai"))
    return out


def analyze(text, extra_upi_ids=None):
    t = (text or "").lower()
    evidence, best = [], {}  # best: category -> weight
    score = 0

    for rid, w, cat, reason, rx in RULES:
        if re.search(rx, t, re.I):
            evidence.append({"type": rid, "weight": w, "reason": reason})
            score += w
            if w > 0:
                best[cat] = max(best.get(cat, 0), w)

    urls = extract_urls(text or "")
    for u in urls:
        for w, reason in check_url(u):
            evidence.append({"type": "link", "weight": w, "reason": reason})
            score += w
            best["Suspicious Link"] = max(best.get("Suspicious Link", 0), w - 20)

    upis = list(dict.fromkeys(extract_upi_ids(text or "") + [u.lower() for u in (extra_upi_ids or [])]))
    for up in upis:
        for w, reason in check_upi_id(up):
            evidence.append({"type": "upi_id", "weight": w, "reason": reason})
            score += w
            best["Suspicious UPI ID"] = max(best.get("Suspicious UPI ID", 0), w)

    # Combo bonus: link + credential/reward together is the classic scam shape
    if urls and any(e["type"] in ("credential", "cashback", "kyc") for e in evidence):
        evidence.append({"type": "combo", "weight": 10, "reason": "Link ke saath PIN/reward/KYC ki baat - classic scam combination"})
        score += 10

    score = max(0, min(100, score))
    level = "Low" if score < 30 else "Medium" if score < 60 else "High"
    category = max(best, key=best.get) if best and score >= 30 else ("Safe" if score < 30 else "Suspicious")
    next_step = NEXT_STEPS.get(category, "Koi bada red flag nahi mila, phir bhi PIN/OTP kisi ko mat batao." if level == "Low"
                               else "Savdhan raho, bank ke official channel se verify karo.")
    return {"score": score, "level": level, "category": category,
            "evidence": sorted(evidence, key=lambda e: -abs(e["weight"])),
            "urls": urls, "upi_ids": upis, "next_step": next_step}


def parse_upi_qr(payload):
    """Parse upi://pay?pa=x@y&pn=Name&am=100 into a dict."""
    from urllib.parse import parse_qs
    p = urlparse(payload)
    q = {k: v[0] for k, v in parse_qs(p.query).items()}
    return {"scheme_ok": p.scheme == "upi", "action": p.netloc or p.path.strip("/"),
            "pa": q.get("pa", ""), "pn": q.get("pn", ""), "am": q.get("am", ""), "tn": q.get("tn", "")}