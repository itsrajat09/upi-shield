# UPI Shield evaluation (synthetic data)

## 1. Transaction model

| Test | Precision | Recall | False-positive rate | TP/FP/TN/FN |
|---|---|---|---|---|
| A. Standard test (750 txns) | 98.8% | 93.3% | 0.2% | 84/1/659/6 |
| B. 5-fold CV (mean, std in brackets) | 95.8% (1.9%) | 92.5% (4.2%) | 0.6% (0.3%) | |
| C. Stress test (3000 txns, 12.0% fraud) | 73.5% | 80.8% | 4.0% | 291/105/2535/69 |

Stress test ROC-AUC 0.972, PR-AUC 0.877.
The stress test hides the signature of 35% of scams and makes 12% of legitimate payments look suspicious, so it is the number to quote as the honest one.

### Recall by scam type (stress test)

| Scam | Recall |
|---|---|
| collect_refund | 77.4% |
| qr_swap | 64.0% |
| remote_access | 98.8% |
| mule_chain | 92.6% |
| fake_kyc | 72.1% |

### Threshold trade-off (stress test)

| Threshold | Precision | Recall | False-positive rate |
|---|---|---|---|
| 0.2 | 45.1% | 94.7% | 15.7% |
| 0.3 | 53.4% | 90.6% | 10.8% |
| 0.4 | 61.7% | 85.6% | 7.2% |
| 0.5 | 73.5% | 80.8% | 4.0% |
| 0.6 | 83.7% | 75.8% | 2.0% |
| 0.7 | 90.3% | 69.7% | 1.0% |
| 0.8 | 94.1% | 61.7% | 0.5% |

## 2. Message / link detector

| Test | Precision | Recall | False-positive rate | TP/FP/TN/FN |
|---|---|---|---|---|
| Dev set (125 messages, rules were tuned on this) | 100.0% | 100.0% | 0.0% | 63/0/62/0 |
| Held-out set (50 new messages, never used for tuning) | 100.0% | 80.0% | 0.0% | 20/0/25/5 |

Missed scams on the held-out set:
- Your electricity will be disconnected tonight 9:30 PM as previous month bill was not updated. Call 9650123478 immediately
- Your parcel is held at Delhi airport, customs duty Rs 1,250 pending. Pay at bit.ly/parcel-duty-pay today
- Earn money by liking videos. Daily income Rs 2,000 guaranteed. Message on Telegram @quickincome
- Dear user, Rs 25,000 loan approved instantly without documents. Pay processing fee Rs 999 to loan.fast@okaxis
- Congrats, you won Rs 50 lakh KBC lottery! To claim, send Rs 5,000 processing fee and your UPI PIN for verification

False alarms on the held-out set:
- none

## 3. Adapting to a brand-new scam (fake investment)

| Test | Precision | Recall | False-positive rate | TP/FP/TN/FN |
|---|---|---|---|---|
| Before (old model) | 0.0% | 0.0% | 1.5% | 0/3/197/100 |
| Novelty alert only | 93.7% | 89.0% | 3.0% | 89/6/194/11 |
| After retraining on 30 analyst labels | 96.3% | 77.0% | 1.5% | 77/3/197/23 |
| Known 5 scams, before | 98.8% | 93.3% | 0.2% | 84/1/659/6 |
| Known 5 scams, after | 98.8% | 94.4% | 0.2% | 85/1/659/5 |
