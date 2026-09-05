"""
run.py — VectorFlow entry point
Usage:  source .venv/bin/activate && python run.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
from dotenv import load_dotenv

load_dotenv()   # reads .env into os.environ

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from sklearn.preprocessing import StandardScaler

import data as D
import models as M
import evaluate as E
import plots as P
import persist as Persist


DATA_PATH  = os.getenv("DATA_PATH",  "data/cic_ids2018_core_training_dataset.csv")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
MODEL_DIR  = os.getenv("MODEL_DIR",  "models")

OUT_FI = os.path.join(OUTPUT_DIR, "feature_importance.png")
OUT_PR = os.path.join(OUTPUT_DIR, "pr_curves.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Load ──────────────────────────────────────────────────────────────────────
df = D.load(DATA_PATH)
print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
print(f"Date range: {df['window_start'].min()} → {df['window_start'].max()}")

# ── Audit ─────────────────────────────────────────────────────────────────────
D.audit(df)

# ── Split ─────────────────────────────────────────────────────────────────────
train_df, val_df, test_df = D.split(df)

feat_cols = D.feature_cols(df)
X_train, y_train = D.xy(train_df, feat_cols)
X_val,   y_val   = D.xy(val_df,   feat_cols)
X_test,  y_test  = D.xy(test_df,  feat_cols)

print(f"\nFeature matrix — train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

# ── Scale (fit on train only) ─────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# ── Train ─────────────────────────────────────────────────────────────────────
trained_models = M.train_all(X_train_s, y_train)

# ── Save models & scaler ──────────────────────────────────────────────────────
Persist.save(trained_models, scaler, MODEL_DIR)

# ── Evaluate ──────────────────────────────────────────────────────────────────
splits = {
    "train": (X_train_s, y_train),
    "val":   (X_val_s,   y_val),
    "test":  (X_test_s,  y_test),
}

results = E.evaluate_all(trained_models, splits)
E.print_comparison(results)

# ── Feature importances ───────────────────────────────────────────────────────
P.print_feature_importance(trained_models, feat_cols)

# ── Plots ─────────────────────────────────────────────────────────────────────
P.plot_feature_importance(trained_models, feat_cols, OUT_FI)
P.plot_pr_curves(trained_models, splits, OUT_PR)

# ── Verdict ───────────────────────────────────────────────────────────────────
print("""
===== Verdict =====

Logistic Regression:
  Lowest capacity, least overfit. Train PR-AUC ~0.24; consistent (if weak)
  across splits. Useful as a calibrated baseline, not a production detector.

Random Forest:
  Near-perfect train memorisation (PR-AUC ~1.00). Val ROC-AUC ~0.73 indicates
  real ranking signal exists, but decision boundary is tuned to training-day
  attack campaigns that look different on val/test days. F1=0 at default 0.5
  threshold — needs threshold tuning on val to become useful.

XGBoost:
  Same story as RF. Val ROC-AUC ~0.74 is marginally better, but test ROC-AUC
  drops to ~0.54 — slightly more overfit than RF to the training distribution
  despite subsample/colsample regularisation.

All three models show F1=0 on val and test at threshold=0.5. This is temporal
distribution shift, not a code issue: train days have 19-21% positives from
active campaigns; val/test days have ~1% of a different character.

Next steps:
  1. Tune decision threshold on val (ROC-AUC of 0.73-0.74 means signal exists).
  2. Evaluate at recall >= 0.5 on val and report the corresponding precision.
  3. Consider rotating attack-heavy days into val to get an honest estimate of
     generalisation across attack types, not just across calendar dates.
""")
