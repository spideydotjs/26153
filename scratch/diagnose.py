"""
diagnose.py — Read-only investigation of two issues:
  1. Attack-family coverage per split in the NEW split config.
  2. XGBoost recall collapse: imbalance ratio, scale_pos_weight, and
     predicted probability distribution on the Test set.
"""

import numpy as np
import pandas as pd

from src.data import TRAIN_DATES, VAL_DATES, TEST_DATES, load, split, feature_cols, xy
from src.persist import load as load_artifacts

# ---------------------------------------------------------------------------
# 1. Attack-family coverage
# ---------------------------------------------------------------------------

# Established day-to-attack-type mapping from raw file audit
DAY_ATTACK_MAP = {
    "2018-02-14": ["SSH-BruteForce", "FTP-BruteForce"],
    "2018-02-15": ["DoS Slowloris", "DoS GoldenEye"],
    "2018-02-16": ["DoS SlowHTTPTest", "DoS Hulk"],
    "2018-02-20": ["DDoS LOIC-HTTP"],
    "2018-02-21": ["DDoS HOIC", "DDoS LOIC-UDP"],
    "2018-02-22": ["Web Attacks (SQLi, XSS, BruteForce-Web)"],
    "2018-02-23": ["Web Attacks (SQLi, XSS, BruteForce-Web)"],
    "2018-02-28": ["Infiltration"],
    "2018-03-01": ["Infiltration"],
    "2018-03-02": ["Botnet"],
}

print("=" * 60)
print(" 1. ATTACK-FAMILY COVERAGE — NEW SPLIT")
print("=" * 60)

splits_meta = {
    "Train": sorted(TRAIN_DATES),
    "Val":   sorted(VAL_DATES),
    "Test":  sorted(TEST_DATES),
}

for split_name, dates in splits_meta.items():
    print(f"\n{split_name} days: {', '.join(dates)}")
    families = []
    for d in dates:
        families.extend(DAY_ATTACK_MAP.get(d, ["UNKNOWN"]))
    for f in families:
        print(f"    {f}")

# Explicit Botnet question
train_dates_sorted = sorted(TRAIN_DATES)
botnet_in_train = "2018-03-02" in TRAIN_DATES
print(f"\n>>> Botnet (Mar 02) in Train? {'YES' if botnet_in_train else 'NO — Test-only, zero Train representation'}")
print(f">>> Infiltration in Train? {'YES (Feb 28)' if '2018-02-28' in TRAIN_DATES else 'NO'}")
print(f">>> Infiltration in Test?  {'YES (Mar 01)' if '2018-03-01' in TEST_DATES else 'NO'}")

# ---------------------------------------------------------------------------
# 2. XGBoost recall collapse diagnosis
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print(" 2. XGBOOST RECALL COLLAPSE DIAGNOSIS")
print("=" * 60)

# 2a. Class imbalance in new vs old split
DATA_PATH = "data/cic_ids2018_complete_dataset.csv"
df = load(DATA_PATH)
train_df, val_df, test_df = split(df)
cols = feature_cols(df)

X_train, y_train = xy(train_df, cols)
X_test,  y_test  = xy(test_df,  cols)

neg_new = int((y_train == 0).sum())
pos_new = int((y_train == 1).sum())
spw_new = neg_new / max(pos_new, 1)
pct_new = pos_new / len(y_train) * 100

# Old split values from session history
neg_old, pos_old = 14_371, 1_477   # Train Feb14-23 from old run
spw_old = 9.60
pct_old = pos_old / (neg_old + pos_old) * 100

print(f"\nClass imbalance — NEW Train: {neg_new:,} neg / {pos_new:,} pos  ({pct_new:.2f}% positive)")
print(f"Class imbalance — OLD Train: {neg_old:,} neg / {pos_old:,} pos  ({pct_old:.2f}% positive)")
print(f"scale_pos_weight — OLD: {spw_old:.2f}  |  NEW (recalculated): {spw_new:.2f}")
print(f"\nNEW split is MORE imbalanced ({pct_new:.2f}% vs {pct_old:.2f}%).")
print(f"scale_pos_weight was CORRECTLY recalculated to {spw_new:.2f} (was {spw_old:.2f}).")

# 2b. Load saved XGBoost and run predict_proba on test set
scaler, models, thresholds = load_artifacts("models")
xgb = models["XGBoost"]
xgb_thresh = thresholds.get("XGBoost", 0.5)

X_test_scaled = scaler.transform(X_test)
proba = xgb.predict_proba(X_test_scaled)[:, 1]

pcts = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
print(f"\nXGBoost Test-set predicted probability distribution (n={len(proba):,}):")
print(f"  min    : {proba.min():.6f}")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"  p{p:02d}    : {np.percentile(proba, p):.6f}")
print(f"  max    : {proba.max():.6f}")
print(f"  mean   : {proba.mean():.6f}")
print(f"  std    : {proba.std():.6f}")
print(f"\nCalibrated threshold applied: {xgb_thresh:.4f}")
print(f"Predictions >= threshold    : {(proba >= xgb_thresh).sum():,} out of {len(proba):,}")

# 2c. Breakdown on true attack windows vs benign
proba_attack = proba[y_test == 1]
proba_benign = proba[y_test == 0]

print(f"\nProbability on TRUE ATTACK windows (n={len(proba_attack)}):")
print(f"  min={proba_attack.min():.4f}  mean={proba_attack.mean():.4f}  "
      f"p50={np.median(proba_attack):.4f}  max={proba_attack.max():.4f}")
print(f"  Fraction above threshold {xgb_thresh:.4f}: "
      f"{(proba_attack >= xgb_thresh).mean()*100:.1f}%  (= recall)")

print(f"\nProbability on BENIGN windows (n={len(proba_benign):,}):")
print(f"  min={proba_benign.min():.4f}  mean={proba_benign.mean():.4f}  "
      f"p50={np.median(proba_benign):.4f}  max={proba_benign.max():.4f}")
print(f"  Fraction above threshold {xgb_thresh:.4f}: "
      f"{(proba_benign >= xgb_thresh).mean()*100:.1f}%  (= FP rate)")

# 2d. Threshold sweep on test to show where recall actually lives
print(f"\nXGBoost threshold sweep on Test set ({len(y_test):,} rows, {y_test.sum()} attacks):")
print(f"  {'Thresh':>8}  {'TP':>5}  {'FP':>6}  {'FN':>5}  {'Recall':>8}  {'Prec':>8}  {'F1':>6}")
for t in [0.01, 0.02, 0.05, 0.10, 0.1407, 0.20, 0.30, 0.40, 0.50]:
    preds = (proba >= t).astype(int)
    tp = int(((preds == 1) & (y_test == 1)).sum())
    fp = int(((preds == 1) & (y_test == 0)).sum())
    fn = int(((preds == 0) & (y_test == 1)).sum())
    rec  = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    print(f"  {t:>8.4f}  {tp:>5}  {fp:>6}  {fn:>5}  {rec*100:>7.1f}%  {prec*100:>7.1f}%  {f1:.4f}")
