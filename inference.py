"""
inference.py — Load saved models, scaler, and calibrated thresholds to run forecasts.
Accepts raw feature dictionaries or pandas DataFrames and returns probabilities + alerts.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

FEATURE_COLS: list[str] = [
    "Tot Fwd Pkts_sum",
    "Tot Bwd Pkts_sum",
    "TotLen Fwd Pkts_sum",
    "TotLen Bwd Pkts_sum",
    "Flow Duration_mean",
    "Flow Duration_std",
    "Flow IAT Mean_mean",
    "Dst Port_nunique",
    "SYN Flag Cnt_sum",
    "ACK Flag Cnt_sum",
    "RST Flag Cnt_sum",
    "FIN Flag Cnt_sum",
    "PSH Flag Cnt_sum",
    "flow_count",
    "bwd_fwd_pkt_ratio",
    "Tot Fwd Pkts_sum_lag1",
    "Tot Fwd Pkts_sum_lag2",
    "Tot Fwd Pkts_sum_lag3",
    "Tot Bwd Pkts_sum_lag1",
    "Tot Bwd Pkts_sum_lag2",
    "Tot Bwd Pkts_sum_lag3",
    "TotLen Fwd Pkts_sum_lag1",
    "TotLen Fwd Pkts_sum_lag2",
    "TotLen Fwd Pkts_sum_lag3",
    "TotLen Bwd Pkts_sum_lag1",
    "TotLen Bwd Pkts_sum_lag2",
    "TotLen Bwd Pkts_sum_lag3",
    "Flow Duration_mean_lag1",
    "Flow Duration_mean_lag2",
    "Flow Duration_mean_lag3",
    "Flow Duration_std_lag1",
    "Flow Duration_std_lag2",
    "Flow Duration_std_lag3",
    "Flow IAT Mean_mean_lag1",
    "Flow IAT Mean_mean_lag2",
    "Flow IAT Mean_mean_lag3",
    "Dst Port_nunique_lag1",
    "Dst Port_nunique_lag2",
    "Dst Port_nunique_lag3",
    "SYN Flag Cnt_sum_lag1",
    "SYN Flag Cnt_sum_lag2",
    "SYN Flag Cnt_sum_lag3",
    "ACK Flag Cnt_sum_lag1",
    "ACK Flag Cnt_sum_lag2",
    "ACK Flag Cnt_sum_lag3",
    "RST Flag Cnt_sum_lag1",
    "RST Flag Cnt_sum_lag2",
    "RST Flag Cnt_sum_lag3",
    "FIN Flag Cnt_sum_lag1",
    "FIN Flag Cnt_sum_lag2",
    "FIN Flag Cnt_sum_lag3",
    "PSH Flag Cnt_sum_lag1",
    "PSH Flag Cnt_sum_lag2",
    "PSH Flag Cnt_sum_lag3",
    "flow_count_lag1",
    "flow_count_lag2",
    "flow_count_lag3",
    "bwd_fwd_pkt_ratio_lag1",
    "bwd_fwd_pkt_ratio_lag2",
    "bwd_fwd_pkt_ratio_lag3",
    "Tot Fwd Pkts_sum_delta1",
    "Tot Bwd Pkts_sum_delta1",
    "TotLen Fwd Pkts_sum_delta1",
    "TotLen Bwd Pkts_sum_delta1",
    "Flow Duration_mean_delta1",
    "Flow Duration_std_delta1",
    "Flow IAT Mean_mean_delta1",
    "Dst Port_nunique_delta1",
    "SYN Flag Cnt_sum_delta1",
    "ACK Flag Cnt_sum_delta1",
    "RST Flag Cnt_sum_delta1",
    "FIN Flag Cnt_sum_delta1",
    "PSH Flag Cnt_sum_delta1",
    "flow_count_delta1",
    "bwd_fwd_pkt_ratio_delta1",
]

MODEL_DIR = os.getenv("MODEL_DIR", "models")


def _load_artifacts():
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Artifacts not found in '{MODEL_DIR}'. Run `python run.py` first to train and save models."
        )

    scaler = joblib.load(scaler_path)
    logreg = joblib.load(os.path.join(MODEL_DIR, "logreg.joblib"))
    rf = joblib.load(os.path.join(MODEL_DIR, "randomforest.joblib"))

    xgb = XGBClassifier()
    xgb.load_model(os.path.join(MODEL_DIR, "xgboost.ubj"))

    thresh_path = os.path.join(MODEL_DIR, "thresholds.json")
    if os.path.exists(thresh_path):
        with open(thresh_path, "r") as f:
            thresholds = json.load(f)
    else:
        thresholds = {"LogReg": 0.5, "RandomForest": 0.5, "XGBoost": 0.5}

    return scaler, logreg, rf, xgb, thresholds


# Lazy-load artifacts so importing inference is fast
_artifacts = None


def get_artifacts():
    global _artifacts
    if _artifacts is None:
        _artifacts = _load_artifacts()
    return _artifacts


def predict(
    features: dict[str, float] | pd.DataFrame,
    use_calibrated_threshold: bool = True,
    custom_threshold: float | None = None,
) -> pd.DataFrame:
    """
    Execute attack prediction using all 3 trained models.

    Parameters:
    - features: Dict with all 75 features or pandas DataFrame with the 75 columns.
    - use_calibrated_threshold: If True, uses optimal thresholds from validation set.
    - custom_threshold: If supplied, overrides thresholds for all models.

    Returns:
    DataFrame with columns: ['row', 'model', 'probability', 'threshold', 'attack_predicted']
    """
    scaler, logreg, rf, xgb, thresholds = get_artifacts()

    if isinstance(features, dict):
        df = pd.DataFrame([features])
    elif isinstance(features, pd.DataFrame):
        df = features.copy()
    else:
        raise TypeError("Input 'features' must be a dict or a pandas DataFrame")

    missing = [col for col in FEATURE_COLS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Input is missing {len(missing)} required feature(s):\n  "
            + "\n  ".join(missing[:10])
            + ("..." if len(missing) > 10 else "")
        )

    # Clean numeric array and apply scaler
    X = df[FEATURE_COLS].values.astype(float)
    X = np.nan_to_num(X, nan=0.0)
    X_scaled = scaler.transform(X)

    model_tuples = [("LogReg", logreg), ("RandomForest", rf), ("XGBoost", xgb)]
    results = []

    for name, model in model_tuples:
        proba = model.predict_proba(X_scaled)[:, 1]

        if custom_threshold is not None:
            thresh = custom_threshold
        elif use_calibrated_threshold:
            thresh = thresholds.get(name, 0.5)
        else:
            thresh = 0.5

        pred = (proba >= thresh).astype(int)

        for i, (p, a) in enumerate(zip(proba, pred)):
            results.append(
                {
                    "row": i,
                    "model": name,
                    "probability": round(float(p), 4),
                    "threshold": round(float(thresh), 4),
                    "attack_predicted": bool(a == 1),
                }
            )

    return pd.DataFrame(results)


if __name__ == "__main__":
    DATA_PATH = os.getenv("DATA_PATH", "data/cic_ids2018_complete_dataset.csv")
    if os.path.exists(DATA_PATH):
        raw = pd.read_csv(DATA_PATH)
        sample = raw[raw["window_start"] >= "2018-03-01"].head(5).reset_index(drop=True)
        print(f"Running inference sanity check on {len(sample)} test windows:")
        preds = predict(sample, use_calibrated_threshold=True)
        for i in range(len(sample)):
            ts = sample.loc[i, "window_start"]
            target = int(sample.loc[i, "Future_Attack_Target"])
            print(f"\n[Window {i}] Time: {ts} | Actual Target: {target}")
            sub = preds[preds["row"] == i][
                ["model", "probability", "threshold", "attack_predicted"]
            ]
            print(sub.to_string(index=False))
