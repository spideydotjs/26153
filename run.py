"""
run.py — Main VectorFlow Model Training and Calibration Pipeline.
Trains LogReg, Random Forest, and XGBoost; optimizes alerting thresholds on Validation;
and evaluates test performance under distribution shift.
"""

import os
import sys
import warnings

from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler

# Ensure local packages are on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import data as D
import evaluate as E
import models as M
import persist as Persist
import plots as P

warnings.filterwarnings("ignore")
load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "data/cic_ids2018_complete_dataset.csv")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
MODEL_DIR = os.getenv("MODEL_DIR", "models")

OUT_FI = os.path.join(OUTPUT_DIR, "feature_importance.png")
OUT_PR = os.path.join(OUTPUT_DIR, "pr_curves.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    print("=" * 65)
    print(" VectorFlow — Network Attack Forecasting Training Pipeline")
    print("=" * 65)

    # 1. Load dataset
    df = D.load(DATA_PATH)
    print(f"Loaded {len(df):,} windows, {df.shape[1]} columns")
    print(f"Time range: {df['window_start'].min()} -> {df['window_start'].max()}")

    # 2. Session Audit
    D.audit(df)

    # 3. Chronological Train/Val/Test Split
    train_df, val_df, test_df = D.split(df)
    feat_cols = D.feature_cols(df)

    X_train, y_train = D.xy(train_df, feat_cols)
    X_val, y_val = D.xy(val_df, feat_cols)
    X_test, y_test = D.xy(test_df, feat_cols)

    print(
        f"\nFeature matrix shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}"
    )

    # 4. Standard Scaling (Fit on Train only)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # 5. Train all 3 models
    trained_models = M.train_all(X_train_s, y_train)

    # 6. Optimize decision thresholds on Validation set (constraint: Recall >= 0.50)
    print("\n" + "=" * 65)
    print(" Threshold Calibration on Validation Set (target: Recall >= 50%)")
    print("=" * 65)
    optimal_thresholds = {}
    for name, model in trained_models.items():
        val_proba = model.predict_proba(X_val_s)[:, 1]
        t, f1, p, r = E.find_best_threshold(y_val, val_proba, min_recall=0.50)
        optimal_thresholds[name] = t
        print(
            f"  [{name:<12}] Optimal threshold: {t:.4f} -> Val F1: {f1:.4f} | Prec: {p:.4f} | Recall: {r:.4f}"
        )

    # 7. Persist models, scaler, and optimal thresholds
    print("\n" + "=" * 65)
    print(" Persisting Model Artifacts")
    print("=" * 65)
    Persist.save(trained_models, scaler, MODEL_DIR, optimal_thresholds)

    # 8. Multi-split Evaluation (with default 0.50 vs calibrated threshold)
    splits = {
        "train": (X_train_s, y_train),
        "val": (X_val_s, y_val),
        "test": (X_test_s, y_test),
    }

    print("\n--- Performance with Default 0.50 Threshold ---")
    res_default = E.evaluate_all(trained_models, splits, thresholds=None)
    E.print_comparison(res_default)

    print("\n--- Performance with Calibrated Decision Thresholds ---")
    res_calibrated = E.evaluate_all(
        trained_models, splits, thresholds=optimal_thresholds
    )
    E.print_comparison(res_calibrated)

    # 9. Top-15 Feature Importance & Plots
    P.print_feature_importance(trained_models, feat_cols)
    P.plot_feature_importance(trained_models, feat_cols, OUT_FI)
    P.plot_pr_curves(trained_models, splits, OUT_PR)

    print("\nTraining and evaluation pipeline completed successfully!")


if __name__ == "__main__":
    main()
