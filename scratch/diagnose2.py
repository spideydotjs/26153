"""
diagnose2.py — Methodology investigation (read-only).

1. Val-based calibrated threshold vs Test-swept thresholds, side by side.
2. Botnet cold-start: check all days for Botnet-like traffic, then
   report metrics with Mar02 excluded vs included.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data import TRAIN_DATES, VAL_DATES, TEST_DATES, load, split, feature_cols, xy
from src.persist import load as load_artifacts

# ── helpers ────────────────────────────────────────────────────────────────

def threshold_row(name, split_label, t, y_true, proba):
    preds = (proba >= t).astype(int)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    rec  = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    return dict(model=name, subset=split_label, threshold=t,
                TP=tp, FP=fp, FN=fn, TN=tn,
                Recall=round(rec*100, 1), Precision=round(prec*100, 1),
                F1=round(f1, 4))

def auc_scores(y_true, proba):
    pr  = average_precision_score(y_true, proba) if y_true.sum() > 0 else float("nan")
    roc = roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else float("nan")
    return pr, roc

# ── load data & models ─────────────────────────────────────────────────────

DATA_PATH = "data/cic_ids2018_complete_dataset.csv"
df = load(DATA_PATH)
train_df, val_df, test_df = split(df)
cols = feature_cols(df)

_, y_val  = xy(val_df,  cols)
X_test, y_test = xy(test_df, cols)

scaler, models, thresholds = load_artifacts("models")

X_val_s  = scaler.transform(val_df[cols].values.astype(float))
X_test_s = scaler.transform(X_test)

# Separate Test into Botnet (Mar02) vs non-Botnet (Feb21 + Mar01)
test_df_nb = test_df[test_df["date"] != "2018-03-02"]
test_df_bt = test_df[test_df["date"] == "2018-03-02"]

X_test_nb_s = scaler.transform(test_df_nb[cols].values.astype(float))
X_test_bt_s = scaler.transform(test_df_bt[cols].values.astype(float))
y_test_nb   = test_df_nb["Future_Attack_Target"].values.astype(int)
y_test_bt   = test_df_bt["Future_Attack_Target"].values.astype(int)

SWEEP = [0.01, 0.05, 0.10, None]  # None = calibrated

# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — Threshold strategy: Val-calibrated vs Test-swept, side by side
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print(" PART 1 — VAL-CALIBRATED vs TEST-SWEPT THRESHOLDS (side by side)")
print("=" * 70)

for model_name, model in models.items():
    cal_thresh = thresholds[model_name]
    proba_val  = model.predict_proba(X_val_s)[:, 1]
    proba_test = model.predict_proba(X_test_s)[:, 1]

    pr_auc, roc_auc = auc_scores(y_test, proba_test)

    print(f"\n{'─'*70}")
    print(f"  {model_name}  |  Val-calibrated threshold: {cal_thresh:.4f}  "
          f"|  Test PR-AUC: {pr_auc:.4f}  ROC-AUC: {roc_auc:.4f}")
    print(f"{'─'*70}")
    print(f"  {'Source':12s}  {'Threshold':>10}  {'TP':>4}  {'FP':>5}  "
          f"{'FN':>4}  {'Recall':>8}  {'Precision':>10}  {'F1':>7}")
    print(f"  {'─'*12}  {'─'*10}  {'─'*4}  {'─'*5}  {'─'*4}  {'─'*8}  {'─'*10}  {'─'*7}")

    # Val-calibrated row on Val set (what calibration optimised for)
    r = threshold_row(model_name, "Val@cal", cal_thresh, y_val, proba_val)
    print(f"  {'Val@calibrated':12s}  {cal_thresh:>10.4f}  {r['TP']:>4}  {r['FP']:>5}  "
          f"{r['FN']:>4}  {r['Recall']:>7.1f}%  {r['Precision']:>9.1f}%  {r['F1']:>7.4f}")

    # Val-calibrated applied to Test
    r = threshold_row(model_name, "Test@cal", cal_thresh, y_test, proba_test)
    print(f"  {'Test@cal':12s}  {cal_thresh:>10.4f}  {r['TP']:>4}  {r['FP']:>5}  "
          f"{r['FN']:>4}  {r['Recall']:>7.1f}%  {r['Precision']:>9.1f}%  {r['F1']:>7.4f}  ← frozen")

    # Test-swept thresholds
    for t in [0.01, 0.05, 0.10, 0.20, 0.30]:
        r = threshold_row(model_name, f"Test@{t:.2f}", t, y_test, proba_test)
        print(f"  {'Test@'+str(t):12s}  {t:>10.4f}  {r['TP']:>4}  {r['FP']:>5}  "
              f"{r['FN']:>4}  {r['Recall']:>7.1f}%  {r['Precision']:>9.1f}%  {r['F1']:>7.4f}")

# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — Botnet cold-start investigation
# ═══════════════════════════════════════════════════════════════════════════

print(f"\n\n{'='*70}")
print(" PART 2 — BOTNET COLD-START")
print(f"{'='*70}")

# 2a. All days and attack types — confirm no other Botnet day
print("""
Day-to-attack-type mapping (all 10 capture days):
  2018-02-14  SSH-BruteForce, FTP-BruteForce
  2018-02-15  DoS Slowloris, DoS GoldenEye
  2018-02-16  DoS SlowHTTPTest, DoS Hulk
  2018-02-20  DDoS LOIC-HTTP
  2018-02-21  DDoS HOIC, DDoS LOIC-UDP
  2018-02-22  Web Attacks (SQLi, XSS, BruteForce-Web)
  2018-02-23  Web Attacks (SQLi, XSS, BruteForce-Web)
  2018-02-28  Infiltration
  2018-03-01  Infiltration
  2018-03-02  Botnet   ← ONLY occurrence in the entire dataset

Conclusion: No other capture day in CIC-IDS2018 contains Botnet traffic.
Botnet MUST remain Test-only. Zero Train/Val representation is unavoidable
with this dataset. We cannot move it into Train without invalidating the Test set.
""")

# 2b. Window counts
print(f"Mar02 (Botnet) windows: {len(test_df_bt):,} total | "
      f"{int(y_test_bt.sum())} attack precursors | "
      f"{int((y_test_bt==0).sum())} benign")
print(f"Non-Botnet Test (Feb21+Mar01): {len(test_df_nb):,} total | "
      f"{int(y_test_nb.sum())} attack precursors | "
      f"{int((y_test_nb==0).sum())} benign\n")

# 2c. Per-model metrics: full Test vs non-Botnet Test vs Botnet-only
print(f"{'─'*70}")
print(f"  {'Model':12s}  {'Subset':20s}  {'Thresh':>8}  {'TP':>4}  {'FP':>5}  "
      f"{'FN':>4}  {'Recall':>8}  {'Prec':>8}  {'PR-AUC':>8}  {'ROC-AUC':>9}")
print(f"  {'─'*12}  {'─'*20}  {'─'*8}  {'─'*4}  {'─'*5}  {'─'*4}  "
      f"{'─'*8}  {'─'*8}  {'─'*8}  {'─'*9}")

for model_name, model in models.items():
    cal_thresh = thresholds[model_name]

    proba_test    = model.predict_proba(X_test_s)[:, 1]
    proba_test_nb = model.predict_proba(X_test_nb_s)[:, 1]
    proba_test_bt = model.predict_proba(X_test_bt_s)[:, 1]

    for label, y_t, prob_t in [
        ("Full Test",          y_test,    proba_test),
        ("Test excl. Botnet",  y_test_nb, proba_test_nb),
        ("Botnet only",        y_test_bt, proba_test_bt),
    ]:
        r = threshold_row(model_name, label, cal_thresh, y_t, prob_t)
        pr_auc, roc_auc = auc_scores(y_t, prob_t)
        print(f"  {model_name:12s}  {label:20s}  {cal_thresh:>8.4f}  "
              f"{r['TP']:>4}  {r['FP']:>5}  {r['FN']:>4}  "
              f"{r['Recall']:>7.1f}%  {r['Precision']:>7.1f}%  "
              f"{pr_auc:>8.4f}  {roc_auc:>9.4f}")
    print()

# 2d. Botnet-specific recall at various thresholds (just XGBoost and RF)
print(f"\nBotnet-only threshold sweep (Mar02, {len(y_test_bt)} windows, {y_test_bt.sum()} attacks):")
print(f"  {'Model':12s}  {'Thresh':>8}  {'TP':>4}  {'FP':>5}  {'FN':>4}  {'Recall':>8}  {'Prec':>8}")
for model_name in ["RandomForest", "XGBoost"]:
    model = models[model_name]
    proba_bt = model.predict_proba(X_test_bt_s)[:, 1]
    for t in [0.01, 0.05, 0.10, 0.20, thresholds[model_name]]:
        r = threshold_row(model_name, "Botnet", t, y_test_bt, proba_bt)
        print(f"  {model_name:12s}  {t:>8.4f}  {r['TP']:>4}  {r['FP']:>5}  "
              f"{r['FN']:>4}  {r['Recall']:>7.1f}%  {r['Precision']:>7.1f}%")
    print()
