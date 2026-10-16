# Model card: telco churn scorer

Production model version 1, MLflow run `e6cfd563cfaf4940920c02d4f240a9ee`, trained 2026-10-14. The scorer is an XGBoost classifier inside a scikit-learn pipeline (median-free numeric imputer, one-hot categories). The decision threshold is 0.5 and was not tuned. Serving reads `models/versions/1/pipeline.joblib`.

## Data

Public IBM telco customer churn extract, 7043 customers and 21 columns. SHA-256 `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`. Source and the blank-`TotalCharges` note are in `data/SOURCE.md`.

The target is `Churn` (`Yes` / `No`). The positive rate on the full table is 0.2654. `TotalCharges` is blank for 11 customers with tenure 0; the loader turns those cells into null and the pipeline imputes them as 0.

Inputs are `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, and `TotalCharges`. `customerID` is not a feature.

The training split is a stratified 80/20 holdout, `random_state=42`: 5634 train rows and 1409 test rows. Optuna TPE (25 trials, seed 42) tunes on an inner 20% of the training split. The test set is scored once. The table has no subscription date, so the split is random within the extract, not a forecast over time.

Churn counts on the full 7043-row table:

| Group | Customers | Churners | Churn rate |
| --- | ---: | ---: | ---: |
| Contract = Month-to-month | 3875 | 1655 | 0.4271 |
| Contract = One year | 1473 | 166 | 0.1127 |
| Contract = Two year | 1695 | 48 | 0.0283 |
| SeniorCitizen = 0 | 5901 | 1393 | 0.2361 |
| SeniorCitizen = 1 | 1142 | 476 | 0.4168 |

## Metrics

Held-out test set, threshold 0.5. These match `reports/training_metrics.json` and `models/versions/1/metrics.json`.

| Metric | Value |
| --- | ---: |
| ROC AUC | 0.8459 |
| Average precision | 0.6587 |
| Precision at 0.5 | 0.5122 |
| Recall at 0.5 | 0.7888 |
| Log loss | 0.4844 |
| Brier score | 0.1631 |

The selected booster has `max_depth` 3 and 45 trees. The largest gain is `cat__Contract_Month-to-month` at 0.2934.

### Holdout slices

Same 1409-row test set, scored with the production pipeline. Selection rate is the share of rows with predicted probability at least 0.5. FPR is false positives divided by actual non-churners. ROC AUC on a slice is ranking quality inside that slice.

| Slice | Rows | Churners | Churn rate | Selection rate | Precision | Recall | FPR | ROC AUC | Average precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Contract = Month-to-month | 773 | 329 | 0.4256 | 0.7180 | 0.5243 | 0.8845 | 0.5946 | 0.7534 | 0.6918 |
| Contract = One year | 300 | 36 | 0.1200 | 0.0700 | 0.1905 | 0.1111 | 0.0644 | 0.7495 | 0.2785 |
| Contract = Two year | 336 | 9 | 0.0268 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7360 | 0.0841 |
| SeniorCitizen = 0 | 1187 | 276 | 0.2325 | 0.3614 | 0.4825 | 0.7500 | 0.2437 | 0.8475 | 0.6366 |
| SeniorCitizen = 1 | 222 | 98 | 0.4414 | 0.6622 | 0.5986 | 0.8980 | 0.4758 | 0.7859 | 0.7184 |

Month-to-month customers are most of the test churn (329 of 374). Inside that slice, ROC AUC is 0.7534, below the overall 0.8459, and the 0.5 threshold flags 0.7180 of them (recall 0.8845, FPR 0.5946). One-year customers are flagged rarely (selection rate 0.0700, recall 0.1111). Two-year customers are not flagged at all on this holdout: 9 churners, selection rate 0.0000, recall 0.0000, FPR 0.0000. Their slice ROC AUC is 0.7360, estimated from those 9 events.

Senior citizens are 222 of 1409 test rows, with holdout churn rate 0.4414 against 0.2325 for everyone else. The model flags them more often (selection rate 0.6622 vs 0.3614) and recovers more of their churn (recall 0.8980 vs 0.7500). The false-positive rate is 0.4758 for seniors and 0.2437 for non-seniors. Slice ROC AUC is 0.7859 for seniors and 0.8475 for non-seniors.

`Contract` and `SeniorCitizen` are inputs. These gaps are how the trained model treats those fields on this holdout, not a separate fairness constraint.

## Limitations

- One stratified random split, seed 42. There is no event time in the extract, so the metrics are not a forward-looking test.
- The threshold is 0.5 on every row. It was not chosen for precision, recall, or any slice.
- Two-year churn on the holdout is 9 rows. The 0.0000 recall and the 0.7360 AUC for that slice are small-sample figures.
- The senior slice is 222 rows (98 churners). The FPR gap (0.4758 vs 0.2437) is large enough to matter for a review queue, and it is still one split.
- The largest gain feature is month-to-month contract. Overall AUC is helped by separating low-churn fixed-term customers from month-to-month customers; within month-to-month the AUC is 0.7534.
- The label is recorded churn from the IBM demo extract. It folds in whatever retention treatment those customers already received.
- `TotalCharges` is imputed as 0 for the 11 tenure-0 rows.
- The training population is this 7043-row public sample. A live book of business can differ in base rate and in which plans exist.
- Scores are not calibrated per slice. The overall Brier score is 0.1631.
- `gender` is an input. This card does not report a gender slice.
- The training objective is ROC AUC. Nothing in the fit penalizes the selection-rate or FPR gap between seniors and non-seniors, or between contract types.

## Use

The score is a relative churn ranking for a retention review of customers who look like this sample. It is not a credit, eligibility, or denial decision.
