"""
persist.py — Serialization and deserialization of models, scalers, and optimal thresholds.
"""

import json
import os

import joblib
from xgboost import XGBClassifier


def save(
    models: dict,
    scaler,
    model_dir: str = "models",
    thresholds: dict[str, float] | None = None,
):
    """
    Persist all models, scaler, and calibrated decision thresholds to model_dir.
    - LogisticRegression, RandomForest -> joblib
    - XGBoost -> native binary .ubj format
    - StandardScaler -> joblib
    - Thresholds -> thresholds.json
    """
    os.makedirs(model_dir, exist_ok=True)

    # Save fitted scaler
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler       -> {scaler_path}")

    # Save models
    for name, model in models.items():
        if name == "XGBoost":
            path = os.path.join(model_dir, "xgboost.ubj")
            model.save_model(path)
        else:
            filename = name.lower().replace(" ", "_") + ".joblib"
            path = os.path.join(model_dir, filename)
            joblib.dump(model, path)
        print(f"Saved {name:<12} -> {path}")

    # Save optimal decision thresholds
    thresh_path = os.path.join(model_dir, "thresholds.json")
    saved_thresh = (
        thresholds
        if thresholds
        else {"LogReg": 0.5, "RandomForest": 0.5, "XGBoost": 0.5}
    )
    with open(thresh_path, "w") as f:
        json.dump(saved_thresh, f, indent=2)
    print(f"Saved thresholds   -> {thresh_path}")


def load(model_dir: str = "models") -> tuple[object, dict, dict]:
    """
    Load scaler, all models, and decision thresholds from model_dir.
    Returns: (scaler, models_dict, thresholds_dict)
    """
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))

    xgb = XGBClassifier()
    xgb.load_model(os.path.join(model_dir, "xgboost.ubj"))

    models = {
        "LogReg": joblib.load(os.path.join(model_dir, "logreg.joblib")),
        "RandomForest": joblib.load(os.path.join(model_dir, "randomforest.joblib")),
        "XGBoost": xgb,
    }

    thresh_path = os.path.join(model_dir, "thresholds.json")
    if os.path.exists(thresh_path):
        with open(thresh_path, "r") as f:
            thresholds = json.load(f)
    else:
        thresholds = {"LogReg": 0.5, "RandomForest": 0.5, "XGBoost": 0.5}

    return scaler, models, thresholds
