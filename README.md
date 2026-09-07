# VectorFlow — Early Network Attack Forecaster
**SIH 2026 | Problem Statement 26153**

VectorFlow anticipates network intrusions **120 seconds in advance** before malicious payloads land by modeling statistical flow shifts across 10-second temporal windows with historical lags and velocity deltas.

---

## Project Structure

```text
26153/
├── app.py                      # Redesigned Streamlit interactive dashboard
├── inference.py                # Ensemble prediction engine with threshold calibration
├── run.py                      # Retraining pipeline with validation calibration
├── requirements.txt            # Python dependencies (polars, scikit-learn, xgboost, streamlit, pytest, ruff)
├── src/
│   ├── polars_aggregator.py    # High-speed flow aggregation using Polars (10s windows + lags + deltas)
│   ├── data.py                 # Chronological splitting & session auditing
│   ├── models.py               # Model builders (LogReg, Random Forest, XGBoost)
│   ├── evaluate.py             # Evaluation metrics, PR-AUC, & F1 threshold optimizer
│   ├── persist.py              # Serialization of models, scaler, & thresholds
│   └── plots.py                # Feature importance and PR curves
├── tests/
│   └── test_vectorflow.py      # Pytest automated test suite (7/7 tests passing)
├── cic-ids2018-processed/      # Raw day-by-day flow CSVs (10 days)
├── data/                       # Preprocessed & core training dataset
├── models/                     # Saved models (scaler.joblib, logreg, randomforest, xgboost.ubj, thresholds.json)
└── outputs/                    # Output plots (feature importance, PR curves)
```

---

## Quickstart

### 1. Activate Environment
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Aggregator (Polars)
Aggregate any raw flow files from `cic-ids2018-processed/` into 10-second window feature sets (15 base + 45 lags + 15 deltas):
```bash
python src/polars_aggregator.py "cic-ids2018-processed/*.csv" "data/cic_ids2018_polars_aggregated.csv"
```

### 3. Retrain and Calibrate Models
Trains Logistic Regression, Random Forest, and XGBoost; optimizes operational alerting thresholds on Validation data (Recall $\ge$ 50%):
```bash
python run.py
```

### 4. Run Automated Test Suite
```bash
pytest tests/ -v
```

### 5. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```

---

## Model Calibration & Thresholds

Under temporal distribution shift (where train days have 11–21% attack rate while validation/test days have ~1%), default 0.5 decision thresholds yield zero true positives. 

VectorFlow optimizes decision thresholds on the validation set to maintain practical detection sensitivity:

| Model | Val PR-AUC | Optimal Threshold | Val Recall | Val F1 | Test Recall | Test F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | **0.0266** | `0.0019` | **62.5%** | 0.0500 | **51.2%** | 0.0265 |
| **Random Forest** | **0.0524** | `0.0879` | **45.8%** | 0.0665 | **41.5%** | 0.0293 |
| **LogReg** | **0.0103** | `0.0168` | **95.8%** | 0.0194 | **100.0%** | 0.0236 |

Calibrated thresholds enable early attack alerts without silent detector failure.
