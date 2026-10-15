"""Synthetic UPI transaction generator. No real customer data is used anywhere.

Run:  python data_gen.py
Makes: ../data/transactions.csv            (training + test data, 5 known fraud types)
       ../data/unseen_investment_scam.csv  (a NEW scam type, used later for the adaptivity demo)
"""
import csv
import os

import numpy as np

FEATURES = [
    "amount", "amount_ratio", "hour", "is_new_payee", "payee_txn_count", "payee_age_days",
    "payee_senders_24h", "velocity_1h", "new_device", "screen_share", "from_link",
    "is_collect", "qr_name_mismatch", "narration_refund", "narration_fee",
]
COLUMNS = FEATURES + ["fraud_type", "label"]
FRAUD_TYPES = ["collect_refund", "qr_swap", "remote_access", "mule_chain", "fake_kyc"]

_w = np.array([1, 1, 1, 1, 1, 2, 3, 5, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 8, 5, 3], dtype=float)
HOUR_P = _w / _w.sum()
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _base(rng):
    return dict(
        hour=int(rng.choice(24, p=HOUR_P)), new_device=int(rng.random() < .05),
        screen_share=int(rng.random() < .01), from_link=int(rng.random() < .08),
        velocity_1h=int(rng.choice([0, 0, 1, 1, 2, 3])), is_collect=0, qr_name_mismatch=0,
        narration_refund=0, narration_fee=0, is_new_payee=0, payee_txn_count=0,
        payee_age_days=0, payee_senders_24h=0, amount=0.0)


def _finish(r, avg, kind, label):
    r["amount"] = round(float(r["amount"]), 2)
    r["amount_ratio"] = round(r["amount"] / avg, 2)
    r["fraud_type"], r["label"] = kind, label
    return r


def _ri(rng, lo, hi):
    return int(rng.integers(lo, hi + 1))


def make_legit(rng):
    kind = rng.choice(["known", "merchant", "person", "bill", "night", "collect_new"],
                      p=[.40, .25, .13, .08, .06, .08])
    avg = float(rng.lognormal(6.4, .6))  # this user's typical payment size
    r = _base(rng)
    if kind == "known":
        r.update(payee_txn_count=_ri(rng, 3, 60), payee_age_days=_ri(rng, 200, 3000),
                 payee_senders_24h=_ri(rng, 0, 5), is_collect=int(rng.random() < .10),
                 amount=avg * rng.lognormal(0, .5))
    elif kind == "merchant":
        new = rng.random() < .4
        r.update(is_new_payee=int(new), payee_txn_count=0 if new else _ri(rng, 1, 10),
                 payee_age_days=_ri(rng, 200, 3000), payee_senders_24h=_ri(rng, 3, 80),
                 qr_name_mismatch=int(rng.random() < .03), amount=avg * rng.lognormal(0, .6))
    elif kind == "person":
        new = rng.random() < .7
        r.update(is_new_payee=int(new), payee_txn_count=0 if new else _ri(rng, 1, 8),
                 payee_age_days=_ri(rng, 100, 3000), payee_senders_24h=_ri(rng, 0, 4),
                 narration_refund=int(rng.random() < .08), amount=rng.uniform(200, 8000))
    elif kind == "bill":
        new = rng.random() < .2
        r.update(hour=_ri(rng, 9, 20), velocity_1h=_ri(rng, 0, 1), is_new_payee=int(new),
                 payee_txn_count=0 if new else _ri(rng, 1, 12), payee_age_days=_ri(rng, 300, 3000),
                 payee_senders_24h=_ri(rng, 5, 200), amount=rng.uniform(5000, 50000))
    elif kind == "night":
        r.update(hour=int(rng.choice([22, 23, 0, 1, 2, 3])), payee_txn_count=_ri(rng, 3, 30),
                 payee_age_days=_ri(rng, 200, 3000), payee_senders_24h=_ri(rng, 0, 8),
                 amount=avg * .5 * rng.lognormal(0, .4))
    else:  # collect_new: genuine collect request from a new payee, e.g. a seller or a biller
        r.update(is_collect=1, is_new_payee=1, payee_age_days=_ri(rng, 200, 3000),
                 payee_senders_24h=_ri(rng, 2, 40), narration_refund=int(rng.random() < .03),
                 amount=rng.uniform(300, 6000))
    if rng.random() < .08:                      # genuinely new accounts exist too (new shops, new friends)
        r["payee_age_days"] = _ri(rng, 1, 120)
    if rng.random() < .02:                      # genuine registration / admission fees exist
        r["narration_fee"] = 1
    return _finish(r, avg, "legit", 0)


def make_fraud(rng, kind):
    avg = float(rng.lognormal(6.4, .6))
    r = _base(rng)
    hit = lambda p: int(rng.random() < p)  # a scam's signature is not always visible

    if kind == "collect_refund":      # "refund" that is really a collect request
        r.update(is_collect=hit(.9), is_new_payee=hit(.9), narration_refund=hit(.85),
                 payee_age_days=_ri(rng, 1, 120), payee_senders_24h=_ri(rng, 0, 10),
                 from_link=hit(.2), amount=rng.uniform(2000, 20000))
    elif kind == "qr_swap":           # scammer's QR pasted over the shop's QR
        r.update(qr_name_mismatch=hit(.85), is_new_payee=hit(.9), payee_age_days=_ri(rng, 1, 60),
                 payee_senders_24h=_ri(rng, 1, 15), amount=rng.uniform(100, 5000))
    elif kind == "remote_access":     # victim guided on a screen-share app
        r.update(screen_share=hit(.85), new_device=hit(.5), is_new_payee=hit(.8),
                 hour=_ri(rng, 9, 23), velocity_1h=_ri(rng, 2, 6), payee_age_days=_ri(rng, 1, 90),
                 payee_senders_24h=_ri(rng, 0, 12), amount=rng.uniform(10000, 100000))
    elif kind == "mule_chain":        # fresh account collecting from many victims
        r.update(payee_senders_24h=_ri(rng, 8, 60), payee_age_days=_ri(rng, 1, 14),
                 is_new_payee=hit(.9), velocity_1h=_ri(rng, 1, 4),
                 hour=int(rng.choice([0, 1, 2, 3, 4, 5])) if rng.random() < .4 else r["hour"],
                 amount=rng.uniform(4000, 49000))
    elif kind == "fake_kyc":          # fee demanded via a link: "pay Rs 10 to verify KYC"
        r.update(from_link=hit(.85), narration_fee=hit(.8), is_new_payee=hit(.9),
                 payee_age_days=_ri(rng, 1, 60), payee_senders_24h=_ri(rng, 0, 20),
                 amount=rng.uniform(5, 500))
    elif kind == "fake_investment":   # NEW pattern: repeated "deposits" to a fake platform
        r.update(payee_txn_count=_ri(rng, 1, 6), velocity_1h=_ri(rng, 3, 8),
                 payee_age_days=_ri(rng, 5, 40), payee_senders_24h=_ri(rng, 20, 120),
                 amount=avg * rng.uniform(2, 6))
    if kind in FRAUD_TYPES:
        if kind != "mule_chain" and rng.random() < .20:   # hijacked old account, no "new account" clue
            r["payee_age_days"] = _ri(rng, 150, 2000)
        if rng.random() < .10:                            # stealth scam: looks like a normal payment
            r.update(is_collect=0, qr_name_mismatch=0, screen_share=0, narration_refund=0, narration_fee=0,
                     from_link=0, payee_age_days=_ri(rng, 200, 2000), payee_senders_24h=_ri(rng, 0, 8),
                     is_new_payee=hit(.5), velocity_1h=_ri(rng, 0, 2))
    return _finish(r, avg, kind, 1)


def write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows({k: r[k] for k in COLUMNS} for r in rows)
    print(f"wrote {len(rows)} rows -> {os.path.normpath(path)}")


def main(n=3000, fraud_rate=0.12, seed=42):
    rng = np.random.default_rng(seed)
    n_fraud = int(n * fraud_rate)
    kinds = rng.choice(FRAUD_TYPES, size=n_fraud, p=[.25, .20, .20, .20, .15])
    rows = [make_fraud(rng, k) for k in kinds] + [make_legit(rng) for _ in range(n - n_fraud)]
    rng.shuffle(rows)
    write(os.path.join(DATA_DIR, "transactions.csv"), rows)

    rng2 = np.random.default_rng(seed + 1)
    unseen = [make_fraud(rng2, "fake_investment") for _ in range(200)] + [make_legit(rng2) for _ in range(400)]
    rng2.shuffle(unseen)
    write(os.path.join(DATA_DIR, "unseen_investment_scam.csv"), unseen)


if __name__ == "__main__":
    main()