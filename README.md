# UPI Shield: AI detection of creative payment frauds (CC6-AI02)

UPI Shield scores a UPI payment **before the money leaves**, reads the message, link or QR behind it, warns the
user in plain language ("You are about to PAY, not receive, ₹5,000") and gives analysts a dashboard.
All data is **synthetic**. No real customer data is used and nothing touches a live payment system.

## What it does

| Requirement | Built | Where to see it |
|---|---|---|
| 1. Transaction risk scoring | RandomForest on 15 features, fraud probability + scam name | `pay.html` |
| 2. Payment-context analysis | Rules for messages, links, UPI IDs, QR strings (English + Hinglish) | `index.html` |
| 3. Creative-fraud patterns | 5 India-specific scams: collect-request refund, QR swap, screen-share, mule account, fake KYC link | `pay.html` scenarios |
| 4. Real-time intervention | Warning sheet before payment, Cancel as main button | `pay.html` |
| 5. Explainability | Up to 5 reasons with impact bars; scam signature always shown | warning sheet, dashboard alerts |
| 6. Adaptivity | Novelty detector flags an unseen scam; analyst labels retrain the model | `adapt.html` |
| 7. Analyst dashboard | KPIs, daily trend, categories, recall by scam, Confirm / False alarm | `dashboard.html` |

## Run it (Windows PowerShell)

```powershell
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python data_gen.py        # makes data/transactions.csv and the unseen-scam file
python model.py           # trains and saves model.joblib
python evaluate.py        # full evaluation, writes evaluation_report.md/.json
uvicorn main:app --reload
```

Open http://127.0.0.1:8000. Pages: `index.html` (check a message or QR), `pay.html`, `adapt.html`, `dashboard.html`.
Swagger docs: http://127.0.0.1:8000/docs

## Folder structure

```
backend/
  main.py        FastAPI app, serves the frontend (static mount must stay last)
  data_gen.py    synthetic transactions: 5 known scams + 1 unseen (fake investment)
  model.py       RandomForest, scoring, one-feature-at-a-time explanations
  adapt.py       novelty detector (Mahalanobis) + retraining from analyst labels
  dashboard.py   sqlite log, /dash/* endpoints
  detector.py    rule engine for messages, links, UPI IDs, QR
  evaluate.py    honest evaluation (standard, 5-fold CV, stress test, thresholds, held-out text)
  tinyfish_intel.py  optional link enrichment (off unless configured)
data/            transactions.csv, unseen_investment_scam.csv, demo_cases.csv (dev), holdout_cases.csv (held-out)
frontend/        index.html, pay.html, adapt.html, dashboard.html
```

## Results (synthetic data)

| Test | Precision | Recall | False-positive rate |
|---|---|---|---|
| Transaction model, standard test (750) | 98.8% | 93.3% | 0.2% |
| Transaction model, 5-fold CV | 95.8% | 92.5% | 0.6% |
| **Transaction model, stress test (3,000)** | **73.5%** | **80.8%** | **4.0%** |
| Message detector, dev set (125, rules tuned on it) | 100% | 100% | 0% |
| **Message detector, held-out set (50)** | **100%** | **80%** | **0%** |
| New scam, before / novelty alert / after 30 labels (recall) | n/a | 0% / 89% / 77% | 1.5% / 3.0% / 1.5% |

The stress test hides the signature of 35% of scams and makes 12% of legitimate payments look suspicious.
Quote it, not the standard test: the data is synthetic, so the standard numbers are optimistic.

## Demo (4 minutes)

1. `index.html`: click "Collect request trap" example, show High risk and reasons.
2. `pay.html`: "Collect-request refund" > Pay > read the warning > Cancel. Repeat with QR swap and Screen-share.
   "Normal payment" and "Genuine collect request" pass with no warning.
3. `pay.html`: "NEW scam: fake investment" > Pay: only a yellow "Unusual pattern" alert.
   `adapt.html` > Run adaptation. Back on `pay.html`: now "Fake Investment Scam".
4. `dashboard.html`: Load demo data, then Re-score, open an alert, Confirm.

Start from a freshly restarted server so the model is not already adapted.

## Limitations

- Synthetic data: scam patterns are written by us, so real fraud will be messier.
- Message rules are keyword based and miss paraphrases (5 of 25 scams missed on the held-out set).
- Small test sets (50 messages, 3,000 stress transactions): rough estimates only.
- The analyst in the adaptation demo is simulated from dataset labels.
- A real app must supply payee age/history, senders in 24 h and a screen-share signal.
- The novelty alert flags about 3% of legitimate payments: use it as a review prompt, never a block.
- Not integrated with live UPI rails (out of scope).

## Next steps

Add a public fraud dataset beside the synthetic one, replace message rules with a trained text classifier,
add graph analysis for mule chains, and feed dashboard verdicts straight into retraining.
