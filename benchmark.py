import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import data as D
import lstm_model as LM
import persist as Persist

warnings.filterwarnings("ignore")
load_dotenv()

DATA_PATH = os.getenv("DATA_PATH", "data/cic_ids2018_complete_dataset.csv")
MODEL_DIR = os.getenv("MODEL_DIR", "models")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray, min_recall: float = 0.50) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    best_thresh, best_f1 = 0.5, 0.0

    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
        if r >= min_recall:
            f1 = (2 * p * r) / (p + r + 1e-12)
            if f1 > best_f1:
                best_f1, best_thresh = f1, t

    if best_f1 == 0.0 and len(thresholds) > 0:
        f1_scores = (2 * precisions[:-1] * recalls[:-1]) / (
            precisions[:-1] + recalls[:-1] + 1e-12
        )
        best_thresh = thresholds[int(np.argmax(f1_scores))]

    return round(float(best_thresh), 4)


def extract_aligned_data():
    df = D.load(DATA_PATH)
    train_df, val_df, test_df = D.split(df)
    tabular_cols = D.feature_cols(df)

    meta_path = os.path.join(MODEL_DIR, "lstm_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        lstm_seq_len = meta.get("seq_len", 10)
        lstm_feats = meta.get("feature_names", LM.BASE_FEATURES)
    else:
        lstm_seq_len = int(os.getenv("LSTM_SEQ_LEN", "10"))
        lstm_feats = LM.BASE_FEATURES

    X_train_seq, y_train_seq = LM.create_sequences(train_df, seq_len=lstm_seq_len, base_cols=lstm_feats)
    X_val_seq, y_val_seq = LM.create_sequences(val_df, seq_len=lstm_seq_len, base_cols=lstm_feats)
    X_test_seq, y_test_seq = LM.create_sequences(test_df, seq_len=lstm_seq_len, base_cols=lstm_feats)

    def get_matching_tabular(split_df: pd.DataFrame, seq_len: int, cols: list[str]):
        X_tab, y_tab = [], []
        for _date, group in split_df.groupby("date", sort=False):
            group = group.sort_values("window_start")
            if len(group) < seq_len:
                continue
            sub = group.iloc[seq_len - 1 :]
            X_tab.append(sub[cols].values.astype(np.float64))
            y_tab.append(sub[D.TARGET_COL].values.astype(np.int64))
        return np.concatenate(X_tab, axis=0), np.concatenate(y_tab, axis=0)

    X_val_tab, y_val_tab = get_matching_tabular(val_df, lstm_seq_len, tabular_cols)
    X_test_tab, y_test_tab = get_matching_tabular(test_df, lstm_seq_len, tabular_cols)

    return {
        "tabular_cols": tabular_cols,
        "lstm_feats": lstm_feats,
        "lstm_seq_len": lstm_seq_len,
        "X_val_tab": X_val_tab,
        "y_val_tab": y_val_tab,
        "X_test_tab": X_test_tab,
        "y_test_tab": y_test_tab,
        "X_val_seq": X_val_seq,
        "y_val_seq": y_val_seq,
        "X_test_seq": X_test_seq,
        "y_test_seq": y_test_seq,
    }


def run_benchmark():
    data = extract_aligned_data()
    y_val = data["y_val_tab"]
    y_test = data["y_test_tab"]

    tab_scaler, tab_models, tab_thresholds = Persist.load(MODEL_DIR)
    X_val_tab_s = tab_scaler.transform(data["X_val_tab"])
    X_test_tab_s = tab_scaler.transform(data["X_test_tab"])

    lstm_net, lstm_scaler, lstm_thresh, _ = LM.load_lstm(MODEL_DIR)

    N_v, S, F = data["X_val_seq"].shape
    N_t = data["X_test_seq"].shape[0]
    X_val_seq_s = lstm_scaler.transform(data["X_val_seq"].reshape(-1, F)).reshape(N_v, S, F).astype(np.float32)
    X_test_seq_s = lstm_scaler.transform(data["X_test_seq"].reshape(-1, F)).reshape(N_t, S, F).astype(np.float32)

    val_probas = {}
    test_probas = {}

    for name, model in tab_models.items():
        val_probas[name] = model.predict_proba(X_val_tab_s)[:, 1]
        test_probas[name] = model.predict_proba(X_test_tab_s)[:, 1]

    val_probas["LSTM"] = LM.predict_proba(lstm_net, X_val_seq_s)
    test_probas["LSTM"] = LM.predict_proba(lstm_net, X_test_seq_s)

    val_probas["Ensemble (XGB+LSTM)"] = 0.5 * val_probas["XGBoost"] + 0.5 * val_probas["LSTM"]
    test_probas["Ensemble (XGB+LSTM)"] = 0.5 * test_probas["XGBoost"] + 0.5 * test_probas["LSTM"]

    model_names = ["LogReg", "RandomForest", "XGBoost", "LSTM", "Ensemble (XGB+LSTM)"]
    calibrated_thresholds = {}

    for name in model_names:
        if name in tab_thresholds and name != "Ensemble (XGB+LSTM)":
            calibrated_thresholds[name] = tab_thresholds[name]
        elif name == "LSTM":
            calibrated_thresholds[name] = lstm_thresh
        else:
            calibrated_thresholds[name] = find_best_threshold(y_val, val_probas[name], min_recall=0.50)

    results = []
    for name in model_names:
        p_test = test_probas[name]
        thresh = calibrated_thresholds[name]
        pred = (p_test >= thresh).astype(int)

        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        pr_auc = average_precision_score(y_test, p_test)
        roc_auc = roc_auc_score(y_test, p_test)
        precision = precision_score(y_test, pred, zero_division=0)
        recall = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        specificity = tn / (tn + fp + 1e-12)
        bal_acc = balanced_accuracy_score(y_test, pred)

        results.append({
            "Model": name,
            "PR-AUC": round(float(pr_auc), 4),
            "ROC-AUC": round(float(roc_auc), 4),
            "F1-Score": round(float(f1), 4),
            "Precision": round(float(precision), 4),
            "Recall": round(float(recall), 4),
            "Specificity": round(float(specificity), 4),
            "Balanced Acc": round(float(bal_acc), 4),
            "Threshold": round(float(thresh), 4),
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
        })

    results_df = pd.DataFrame(results)
    results_path = os.path.join(OUTPUT_DIR, "benchmark_results.csv")
    results_df.to_csv(results_path, index=False)

    plot_benchmark_visualizations(y_test, test_probas, calibrated_thresholds, results_df)
    generate_markdown_report(results_df)

    return results_df


def plot_benchmark_visualizations(y_test, test_probas, thresholds, results_df):
    sns.set_theme(style="whitegrid", palette="muted")
    colors = {
        "LogReg": "#4A90E2",
        "RandomForest": "#50E3C2",
        "XGBoost": "#F5A623",
        "LSTM": "#9013FE",
        "Ensemble (XGB+LSTM)": "#D0021B",
    }

    plt.figure(figsize=(8, 6), dpi=200)
    no_skill = y_test.sum() / len(y_test)
    plt.plot([0, 1], [no_skill, no_skill], linestyle="--", color="#888888", label=f"No-Skill Baseline (AP={no_skill:.3f})")

    for name, proba in test_probas.items():
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        plt.plot(rec, prec, lw=2.2, label=f"{name} (PR-AUC = {ap:.3f})", color=colors.get(name, "black"))

    plt.title("Precision-Recall (PR) Curves — Attack Forecasting Benchmark", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Recall (Detection Rate of Precursors)", fontsize=11)
    plt.ylabel("Precision (Positive Predictive Value)", fontsize=11)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "benchmark_pr_curves.png"))
    plt.close()

    plt.figure(figsize=(8, 6), dpi=200)
    plt.plot([0, 1], [0, 1], linestyle="--", color="#888888", label="Random Classifier (AUC = 0.500)")

    for name, proba in test_probas.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        plt.plot(fpr, tpr, lw=2.2, label=f"{name} (ROC-AUC = {auc:.3f})", color=colors.get(name, "black"))

    plt.title("Receiver Operating Characteristic (ROC) Benchmark", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "benchmark_roc_curves.png"))
    plt.close()

    metrics_to_plot = ["ROC-AUC", "PR-AUC", "F1-Score", "Recall", "Precision"]
    melted = results_df.melt(id_vars=["Model"], value_vars=metrics_to_plot, var_name="Metric", value_name="Score")

    plt.figure(figsize=(11, 6), dpi=200)
    ax = sns.barplot(
        data=melted,
        x="Metric",
        y="Score",
        hue="Model",
        palette=[colors[m] for m in results_df["Model"]],
        edgecolor="black",
        linewidth=0.8,
    )
    plt.title("Model Performance Comparison Across Key Evaluation Metrics", fontsize=13, fontweight="bold", pad=14)
    plt.xlabel("Evaluation Metric", fontsize=11)
    plt.ylabel("Score", fontsize=11)
    plt.ylim([0.0, 1.05])
    plt.legend(title="Model", frameon=True, fontsize=9, loc="upper right")
    for p in ax.patches:
        height = p.get_height()
        if height > 0.02:
            ax.annotate(f"{height:.2f}", (p.get_x() + p.get_width() / 2.0, height),
                        ha="center", va="bottom", fontsize=7.5, rotation=0, xytext=(0, 2),
                        textcoords="offset points")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "benchmark_metrics_barchart.png"))
    plt.close()

    n_models = len(results_df)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 3.8), dpi=200)

    for ax, (_, row) in zip(axes, results_df.iterrows()):
        m_name = row["Model"]
        cm = np.array([[row["TN"], row["FP"]], [row["FN"], row["TP"]]])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        annot = np.array([
            [f"{cm[0,0]:,}\n({cm_norm[0,0]:.1%})", f"{cm[0,1]:,}\n({cm_norm[0,1]:.1%})"],
            [f"{cm[1,0]:,}\n({cm_norm[1,0]:.1%})", f"{cm[1,1]:,}\n({cm_norm[1,1]:.1%})"],
        ])

        sns.heatmap(cm_norm, annot=annot, fmt="", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"],
                    annot_kws={"fontsize": 9, "fontweight": "bold"})
        ax.set_title(f"{m_name}\n(Thresh: {row['Threshold']:.2f})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.suptitle("Normalized Confusion Matrices on Test Shift Distribution", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "benchmark_confusion_matrices.png"))
    plt.close()

    fig = plt.figure(figsize=(16, 11), dpi=200)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot([0, 1], [no_skill, no_skill], "--", color="#888888", label=f"Baseline ({no_skill:.3f})")
    for name, proba in test_probas.items():
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        ax1.plot(rec, prec, lw=2, label=f"{name} ({ap:.3f})", color=colors.get(name, "black"))
    ax1.set_title("A. Precision-Recall Curves", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot([0, 1], [0, 1], "--", color="#888888", label="Random (0.500)")
    for name, proba in test_probas.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax2.plot(fpr, tpr, lw=2, label=f"{name} ({auc:.3f})", color=colors.get(name, "black"))
    ax2.set_title("B. ROC Curves", fontsize=11, fontweight="bold")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend(fontsize=8, loc="lower right")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    sns.barplot(
        data=melted,
        x="Metric",
        y="Score",
        hue="Model",
        palette=[colors[m] for m in results_df["Model"]],
        ax=ax3,
        edgecolor="black",
        linewidth=0.5,
    )
    ax3.set_title("C. Multi-Metric Performance", fontsize=11, fontweight="bold")
    ax3.set_ylim([0.0, 1.05])
    ax3.legend(fontsize=7, loc="upper right")

    ax4 = fig.add_subplot(gs[1, 1])
    for _, row in results_df.iterrows():
        m_name = row["Model"]
        ax4.scatter(row["Recall"], row["Precision"], s=180, color=colors[m_name], label=m_name, edgecolors="black", zorder=5)
        ax4.annotate(
            f"{m_name}\n(F1={row['F1-Score']:.2f})",
            (row["Recall"], row["Precision"]),
            textcoords="offset points",
            xytext=(7, 7),
            fontsize=8.5,
            fontweight="bold",
        )
    ax4.set_title("D. Operational Alerting Tradeoff (Recall vs Precision)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Recall (Precursor Coverage)")
    ax4.set_ylabel("Precision (True Alert Quality)")
    ax4.set_xlim([-0.05, 1.05])
    ax4.set_ylim([-0.05, 0.45])
    ax4.grid(True, alpha=0.3)

    plt.suptitle("VectorFlow Attack Forecasting — Model Benchmark Executive Dashboard", fontsize=14, fontweight="bold", y=0.98)
    plt.savefig(os.path.join(OUTPUT_DIR, "benchmark_summary_dashboard.png"))
    plt.close()


def generate_markdown_report(results_df: pd.DataFrame):
    report_path = os.path.join(OUTPUT_DIR, "benchmark_report.md")
    with open(report_path, "w") as f:
        f.write("# VectorFlow Network Attack Forecasting — Model Benchmark Report\n\n")
        f.write("## 1. Overall Performance Comparison (Test Set)\n\n")
        try:
            f.write(results_df.to_markdown(index=False))
        except Exception:
            f.write(results_df.to_string(index=False))
        f.write("\n\n## 2. Benchmark Visualizations\n\n")
        f.write("- Executive Dashboard: `outputs/benchmark_summary_dashboard.png`\n")
        f.write("- Precision-Recall Curves: `outputs/benchmark_pr_curves.png`\n")
        f.write("- ROC Curves: `outputs/benchmark_roc_curves.png`\n")
        f.write("- Key Metrics Bar Chart: `outputs/benchmark_metrics_barchart.png`\n")
        f.write("- Confusion Matrices: `outputs/benchmark_confusion_matrices.png`\n")


if __name__ == "__main__":
    run_benchmark()
