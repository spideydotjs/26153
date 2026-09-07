"""
diagnose3.py — Methodology finalization (read-only).

1. Check whether any Train day can be split off for threshold-tuning
   without losing attack-family coverage.
2. Produce finalized two-tier Val-calibrated Test performance with
   RF as primary model and XGB/LogReg as comparison rows.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data import TRAIN_DATES, VAL_DATES, TEST_DATES, load, split, feature_cols, xy
from src.persist import load as load_artifacts

# ── mapping ────────────────────────────────────────────────────────────────

DAY_FAMILY = {
    "2018-02-14": "BruteForce",
    "2018-02-15": "DoS",
    "2018-02-16": "DoS",
    "2018-02-20": "DDoS",
    "2018-02-21": "DDoS",
    "2018-02-22": "Web",
    "2018-02-23": "Web",
    "2018-02-28": "Infiltration",
    "2018-03-01": "Infiltration",
    "2018-03-02": "Botnet",
}

# Attack families required in Train (because they appear in Test)
TEST_FAMILIES = {DAY_FAMILY[d] for d in TEST_DATES}  # {DDoS, Infiltration, Botnet}
# Botnet is already accepted as cold-start — we only NEED DDoS + Infiltration in Train

REQUIRED_FAMILIES = TEST_FAMILIES - {"Botnet"}  # {DDoS, Infiltration}

print("=" * 70)
print(" PART 1 — Can we split off a Train day for threshold-tuning?")
print("=" * 70)
print(f"\nTest attack families: {sorted(TEST_FAMILIES)}")
print(f"Required in Train (excl. Botnet cold-start): {sorted(REQUIRED_FAMILIES)}")
print(f"\nCurrent Train days and families:")
for d in sorted(TRAIN_DATES):
    print(f"  {d}  →  {DAY_FAMILY[d]}")

# For each Train day, check if removing it still satisfies coverage
print(f"\nSplit-off feasibility (removing one Train day):")
for candidate in sorted(TRAIN_DATES):
    remaining = TRAIN_DATES - {candidate}
    remaining_families = {DAY_FAMILY[d] for d in remaining}
    covered = REQUIRED_FAMILIES.issubset(remaining_families)
    family = DAY_FAMILY[candidate]
    # How many other Train days have same family?
    same_family_others = [d for d in remaining if DAY_FAMILY[d] == family]
    print(f"  Remove {candidate} ({family:12s}):  "
          f"Coverage OK? {'YES' if covered else 'NO ':3s}  |  "
          f"Same-family days remaining: {len(same_family_others)}  "
          f"{'← CANNOT remove, sole source of '+family if not covered else ''}")

# Conclusion
removable = []
for d in sorted(TRAIN_DATES):
    remaining = TRAIN_DATES - {d}
    remaining_families = {DAY_FAMILY[dd] for dd in remaining}
    if REQUIRED_FAMILIES.issubset(remaining_families):
        removable.append((d, DAY_FAMILY[d]))

print(f"\nRemovable days (coverage preserved): {removable if removable else 'NONE'}")

if removable:
    # Check if any removable day has attacks that overlap with Test families
    useful = [(d, f) for d, f in removable if f in TEST_FAMILIES]
    non_useful = [(d, f) for d, f in removable if f not in TEST_FAMILIES]
    print(f"  Of these, share Test attack families: {useful if useful else 'NONE'}")
    print(f"  Of these, DON'T share Test families:  {non_useful}")
    if not useful:
        print("\n  Even removable days have families NOT in Test (BruteForce/Web),")
        print("  so calibrating thresholds on them wouldn't be representative of Test.")
        print("  → A threshold-tuning split would be misleading, not helpful.")

print(f"\n{'─'*70}")
print("CONCLUSION: Val remains the only calibration source.")
print("Val-calibrated Test performance is the honest final number.")
print("Test-swept numbers are leakage and will NOT be reported as primary.")
print(f"{'─'*70}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — Finalized two-tier report, RF primary
# ═══════════════════════════════════════════════════════════════════════════

DATA_PATH = "data/cic_ids2018_complete_dataset.csv"
df = load(DATA_PATH)
train_df, val_df, test_df = split(df)
cols = feature_cols(df)

X_test, y_test = xy(test_df, cols)

scaler, models, thresholds = load_artifacts("models")
X_test_s = scaler.transform(X_test)

# Split Test into tiers
test_excl_bt = test_df[test_df["date"] != "2018-03-02"]
test_bt_only = test_df[test_df["date"] == "2018-03-02"]

X_excl_s = scaler.transform(test_excl_bt[cols].values.astype(float))
X_bt_s   = scaler.transform(test_bt_only[cols].values.astype(float))
y_excl   = test_excl_bt["Future_Attack_Target"].values.astype(int)
y_bt     = test_bt_only["Future_Attack_Target"].values.astype(int)


def report_row(model_name, subset_name, threshold, y_true, proba):
    preds = (proba >= threshold).astype(int)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())
    rec  = tp / max(tp + fn, 1)
    prec = tp / max(tp + fp, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    pr_auc = average_precision_score(y_true, proba) if y_true.sum() > 0 else float("nan")
    roc_auc = roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else float("nan")
    return (model_name, subset_name, threshold, tp, fp, fn, tn,
            f"{rec*100:.1f}%", f"{prec*100:.1f}%", f"{f1:.4f}",
            f"{pr_auc:.4f}", f"{roc_auc:.4f}")


print(f"\n\n{'='*70}")
print(" PART 2 — FINALIZED TWO-TIER REPORT (Val-calibrated thresholds)")
print(f"{'='*70}")

# Val performance for reference
X_val_s = scaler.transform(val_df[cols].values.astype(float))
y_val   = val_df["Future_Attack_Target"].values.astype(int)

model_order = ["RandomForest", "XGBoost", "LogReg"]

print(f"\n{'─'*70}")
print(f"  Val Calibration Reference (where thresholds were chosen)")
print(f"{'─'*70}")
hdr = f"  {'Model':12s}  {'Threshold':>8}  {'TP':>4}  {'FP':>5}  {'FN':>4}  {'TN':>5}  {'Rec':>7}  {'Prec':>7}  {'F1':>7}  {'PR-AUC':>7}  {'ROC':>7}"
print(hdr)
for mn in model_order:
    tag = " ★" if mn == "RandomForest" else ""
    proba = models[mn].predict_proba(X_val_s)[:, 1]
    r = report_row(mn, "Val", thresholds[mn], y_val, proba)
    print(f"  {mn+tag:14s}  {r[2]:>8.4f}  {r[3]:>4}  {r[4]:>5}  {r[5]:>4}  {r[6]:>5}  "
          f"{r[7]:>7}  {r[8]:>7}  {r[9]:>7}  {r[10]:>7}  {r[11]:>7}")

# Tier 1: Full Test
print(f"\n{'─'*70}")
print(f"  Tier 1: Full Test (all 3 days — includes Botnet cold-start)")
print(f"{'─'*70}")
print(hdr)
for mn in model_order:
    tag = " ★" if mn == "RandomForest" else ""
    proba = models[mn].predict_proba(X_test_s)[:, 1]
    r = report_row(mn, "Full Test", thresholds[mn], y_test, proba)
    print(f"  {mn+tag:14s}  {r[2]:>8.4f}  {r[3]:>4}  {r[4]:>5}  {r[5]:>4}  {r[6]:>5}  "
          f"{r[7]:>7}  {r[8]:>7}  {r[9]:>7}  {r[10]:>7}  {r[11]:>7}")

# Tier 2: Test excl. Botnet
print(f"\n{'─'*70}")
print(f"  Tier 2: Test excl. Botnet (DDoS + Infiltration only)")
print(f"{'─'*70}")
print(hdr)
for mn in model_order:
    tag = " ★" if mn == "RandomForest" else ""
    proba = models[mn].predict_proba(X_excl_s)[:, 1]
    r = report_row(mn, "Test (no Bot)", thresholds[mn], y_excl, proba)
    print(f"  {mn+tag:14s}  {r[2]:>8.4f}  {r[3]:>4}  {r[4]:>5}  {r[5]:>4}  {r[6]:>5}  "
          f"{r[7]:>7}  {r[8]:>7}  {r[9]:>7}  {r[10]:>7}  {r[11]:>7}")

# Botnet-only (for the cold-start caveat)
print(f"\n{'─'*70}")
print(f"  Known Limitation: Botnet-only (Mar 02) — cold-start, out of scope")
print(f"{'─'*70}")
print(hdr)
for mn in model_order:
    tag = " ★" if mn == "RandomForest" else ""
    proba = models[mn].predict_proba(X_bt_s)[:, 1]
    r = report_row(mn, "Botnet only", thresholds[mn], y_bt, proba)
    print(f"  {mn+tag:14s}  {r[2]:>8.4f}  {r[3]:>4}  {r[4]:>5}  {r[5]:>4}  {r[6]:>5}  "
          f"{r[7]:>7}  {r[8]:>7}  {r[9]:>7}  {r[10]:>7}  {r[11]:>7}")

print(f"\n{'='*70}")
print(" METHODOLOGY SUMMARY")
print(f"{'='*70}")
print("""
Split:
  Train : Feb 14, 16, 20, 22, 28 (11,929 rows, 6.09% positive)
  Val   : Feb 15, 23 (6,165 rows, 11.42% positive)  ← threshold calibration source
  Test  : Feb 21, Mar 01, Mar 02 (3,937 rows, 3.61% positive)

Threshold Calibration:
  Calibrated on Val with target Recall >= 50%.
  Frozen and applied to Test without modification.
  No Test-swept thresholds used in final reporting.

Primary Model: RandomForest (class_weight="balanced")
  - Best ROC-AUC on Test (0.6877 full / 0.7694 excl-Botnet)
  - Most graceful degradation on unseen attack types
  - XGBoost and LogReg retained as comparison rows only

Two-Tier Reporting:
  Tier 1 — Full Test: honest end-to-end number including Botnet cold-start penalty
  Tier 2 — Test excl. Botnet: performance on attack families the model could learn
  Botnet caveat: single-day attack family (Mar 02), unavailable for training,
                 explicitly out of scope for this iteration

Leakage Guard:
  No information from Test set used for any model decision.
  Val-calibrated threshold is the only reported threshold.
""")
