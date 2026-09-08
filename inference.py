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
        raise FileNotFoundError(f"Artifacts not found in '{MODEL_DIR}'.")

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

    lstm_obj = None
    lstm_scaler = None
    lstm_thresh = 0.5
    lstm_pt = os.path.join(MODEL_DIR, "lstm.pt")
    if os.path.exists(lstm_pt):
        try:
            import lstm_model as LM
            lstm_net, lstm_sc, l_th, _ = LM.load_lstm(MODEL_DIR)
            lstm_obj = lstm_net
            lstm_scaler = lstm_sc
            lstm_thresh = l_th
            thresholds["LSTM"] = l_th
            thresholds["Ensemble (XGB+LSTM)"] = 0.2019
        except Exception:
            pass

    return scaler, logreg, rf, xgb, thresholds, lstm_obj, lstm_scaler, lstm_thresh


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
    include_lstm: bool = True,
) -> pd.DataFrame:
    scaler, logreg, rf, xgb, thresholds, lstm_net, lstm_scaler, lstm_thresh = get_artifacts()

    if isinstance(features, dict):
        df = pd.DataFrame([features])
    elif isinstance(features, pd.DataFrame):
        df = features.copy()
    else:
        raise TypeError("Input 'features' must be a dict or a pandas DataFrame")

    missing = [col for col in FEATURE_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Input is missing {len(missing)} required feature(s): {missing[:5]}")

    X = df[FEATURE_COLS].values.astype(float)
    X = np.nan_to_num(X, nan=0.0)
    X_scaled = scaler.transform(X)

    model_tuples = [("LogReg", logreg), ("RandomForest", rf), ("XGBoost", xgb)]
    results = []
    probas_by_model = {}

    for name, model in model_tuples:
        proba = model.predict_proba(X_scaled)[:, 1]
        probas_by_model[name] = proba

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

    if include_lstm and lstm_net is not None and lstm_scaler is not None:
        try:
            import lstm_model as LM
            base_feats = LM.BASE_FEATURES
            base_data = df[base_feats].values.astype(np.float32)
            n_samples = len(df)
            seq_len = 10
            sequences = np.repeat(base_data[:, np.newaxis, :], seq_len, axis=1)
            F = len(base_feats)
            seq_scaled = lstm_scaler.transform(sequences.reshape(-1, F)).reshape(n_samples, seq_len, F).astype(np.float32)
            lstm_p = LM.predict_proba(lstm_net, seq_scaled)
            probas_by_model["LSTM"] = lstm_p

            l_thresh = custom_threshold if custom_threshold is not None else (thresholds.get("LSTM", lstm_thresh) if use_calibrated_threshold else 0.5)
            l_pred = (lstm_p >= l_thresh).astype(int)

            for i, (p, a) in enumerate(zip(lstm_p, l_pred)):
                results.append(
                    {
                        "row": i,
                        "model": "LSTM",
                        "probability": round(float(p), 4),
                        "threshold": round(float(l_thresh), 4),
                        "attack_predicted": bool(a == 1),
                    }
                )

            ens_p = 0.5 * probas_by_model["XGBoost"] + 0.5 * lstm_p
            ens_thresh = custom_threshold if custom_threshold is not None else (thresholds.get("Ensemble (XGB+LSTM)", 0.2019) if use_calibrated_threshold else 0.5)
            ens_pred = (ens_p >= ens_thresh).astype(int)
            for i, (p, a) in enumerate(zip(ens_p, ens_pred)):
                results.append(
                    {
                        "row": i,
                        "model": "Ensemble (XGB+LSTM)",
                        "probability": round(float(p), 4),
                        "threshold": round(float(ens_thresh), 4),
                        "attack_predicted": bool(a == 1),
                    }
                )
        except Exception:
            pass

    return pd.DataFrame(results)


if __name__ == "__main__":
    DATA_PATH = os.getenv("DATA_PATH", "data/cic_ids2018_complete_dataset.csv")
    if os.path.exists(DATA_PATH):
        raw = pd.read_csv(DATA_PATH)
        sample = raw[raw["window_start"] >= "2018-03-01"].head(5).reset_index(drop=True)
        preds = predict(sample, use_calibrated_threshold=True, include_lstm=True)
        print(preds)
