import os
import subprocess
import sys
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from inference import FEATURE_COLS, get_artifacts, predict

st.set_page_config(
    page_title="VectorFlow | Early Attack Forecaster",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 24px 30px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid #334155;
    }
    .main-title {
        font-size: 26px;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .main-subtitle {
        color: #94a3b8;
        font-size: 14px;
        margin-top: 6px;
        margin-bottom: 0;
    }
    .badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 9999px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-cyan { background-color: rgba(6, 182, 212, 0.15); color: #22d3ee; border: 1px solid rgba(6, 182, 212, 0.3); }
    .badge-purple { background-color: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.3); }
    .metric-container {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        text-align: center;
    }
    .threat-card-danger {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 1.5px solid #ef4444;
        border-radius: 10px;
        padding: 20px;
        color: #991b1b;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.1);
    }
    .threat-card-safe {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1.5px solid #22c55e;
        border-radius: 10px;
        padding: 20px;
        color: #166534;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 8px 18px;
        font-weight: 600;
    }
</style>
""",
    unsafe_allow_html=True,
)

FEATURE_DEFAULTS = {
    "Tot Fwd Pkts_sum": 1459.41,
    "Tot Bwd Pkts_sum": 1885.97,
    "TotLen Fwd Pkts_sum": 103488.75,
    "TotLen Bwd Pkts_sum": 1635409.47,
    "Flow Duration_mean": 16144159.71,
    "Flow Duration_std": 32744385.79,
    "Flow IAT Mean_mean": 6671282.01,
    "Dst Port_nunique": 44.61,
    "SYN Flag Cnt_sum": 10.63,
    "ACK Flag Cnt_sum": 68.23,
    "RST Flag Cnt_sum": 29.29,
    "FIN Flag Cnt_sum": 1.29,
    "PSH Flag Cnt_sum": 88.09,
    "flow_count": 233.96,
    "bwd_fwd_pkt_ratio": 0.96,
    "Tot Fwd Pkts_sum_lag1": 2396.5,
    "Tot Fwd Pkts_sum_lag2": 4034.46,
    "Tot Fwd Pkts_sum_lag3": 4245.13,
    "Tot Bwd Pkts_sum_lag1": 1893.84,
    "Tot Bwd Pkts_sum_lag2": 1905.25,
    "Tot Bwd Pkts_sum_lag3": 1906.78,
    "TotLen Fwd Pkts_sum_lag1": 134184.03,
    "TotLen Fwd Pkts_sum_lag2": 186621.82,
    "TotLen Fwd Pkts_sum_lag3": 193475.94,
    "TotLen Bwd Pkts_sum_lag1": 1644350.3,
    "TotLen Bwd Pkts_sum_lag2": 1659035.37,
    "TotLen Bwd Pkts_sum_lag3": 1659864.54,
    "Flow Duration_mean_lag1": 16163851.77,
    "Flow Duration_mean_lag2": 16168840.88,
    "Flow Duration_mean_lag3": 16178781.08,
    "Flow Duration_std_lag1": 32754356.22,
    "Flow Duration_std_lag2": 32737307.99,
    "Flow Duration_std_lag3": 32743504.85,
    "Flow IAT Mean_mean_lag1": 6663442.99,
    "Flow IAT Mean_mean_lag2": 6653314.74,
    "Flow IAT Mean_mean_lag3": 6653988.79,
    "Dst Port_nunique_lag1": 44.65,
    "Dst Port_nunique_lag2": 44.7,
    "Dst Port_nunique_lag3": 44.68,
    "SYN Flag Cnt_sum_lag1": 10.67,
    "SYN Flag Cnt_sum_lag2": 10.68,
    "SYN Flag Cnt_sum_lag3": 10.68,
    "ACK Flag Cnt_sum_lag1": 68.32,
    "ACK Flag Cnt_sum_lag2": 68.41,
    "ACK Flag Cnt_sum_lag3": 68.41,
    "RST Flag Cnt_sum_lag1": 29.37,
    "RST Flag Cnt_sum_lag2": 29.39,
    "RST Flag Cnt_sum_lag3": 29.38,
    "FIN Flag Cnt_sum_lag1": 1.29,
    "FIN Flag Cnt_sum_lag2": 1.29,
    "FIN Flag Cnt_sum_lag3": 1.29,
    "PSH Flag Cnt_sum_lag1": 88.28,
    "PSH Flag Cnt_sum_lag2": 88.38,
    "PSH Flag Cnt_sum_lag3": 88.46,
    "flow_count_lag1": 234.4,
    "flow_count_lag2": 234.78,
    "flow_count_lag3": 234.98,
    "bwd_fwd_pkt_ratio_lag1": 0.96,
    "bwd_fwd_pkt_ratio_lag2": 0.97,
    "bwd_fwd_pkt_ratio_lag3": 0.97,
    "Tot Fwd Pkts_sum_delta1": -937.09,
    "Tot Bwd Pkts_sum_delta1": -7.87,
    "TotLen Fwd Pkts_sum_delta1": -30695.28,
    "TotLen Bwd Pkts_sum_delta1": -8940.83,
    "Flow Duration_mean_delta1": -19692.06,
    "Flow Duration_std_delta1": -9970.43,
    "Flow IAT Mean_mean_delta1": 7839.03,
    "Dst Port_nunique_delta1": -0.04,
    "SYN Flag Cnt_sum_delta1": -0.03,
    "ACK Flag Cnt_sum_delta1": -0.08,
    "RST Flag Cnt_sum_delta1": -0.08,
    "FIN Flag Cnt_sum_delta1": 0.0,
    "PSH Flag Cnt_sum_delta1": -0.19,
    "flow_count_delta1": -0.44,
    "bwd_fwd_pkt_ratio_delta1": 0.0,
}


def render_header():
    st.markdown(
        """
    <div class="main-header">
        <div class="main-title">
            <span>🛡️ VECTORFLOW</span>
            <span class="badge badge-cyan">AI Pre-Attack Forecaster</span>
            <span class="badge badge-purple">PyTorch LSTM & XGBoost</span>
        </div>
        <p class="main-subtitle">
            Anticipating cyber attack payloads <b>120 seconds in advance</b> via 10s network traffic telemetry and temporal sequence learning.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_single_prediction(preds: pd.DataFrame, primary_model: str):
    primary = preds[preds["model"] == primary_model].iloc[0]
    prob = primary["probability"]
    thresh = primary["threshold"]
    is_attack = primary["attack_predicted"]

    if is_attack:
        st.markdown(
            f"""
        <div class="threat-card-danger">
            <h3 style="margin:0; font-size:20px; font-weight:700;">🚨 CRITICAL: IMMINENT ATTACK FORECASTED (Next 120s)</h3>
            <p style="margin:8px 0 0 0; font-size:14px;">
                <b>{primary_model}</b> triggered an operational alarm with probability <b>{prob:.2%}</b> 
                (threshold: {thresh:.4f}). Immediate traffic shaping or active firewall ACL mitigation recommended.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
        <div class="threat-card-safe">
            <h3 style="margin:0; font-size:20px; font-weight:700;">✅ NORMAL: NETWORK IN HEALTHY STATE</h3>
            <p style="margin:8px 0 0 0; font-size:14px;">
                <b>{primary_model}</b> estimated precursor probability at <b>{prob:.2%}</b> 
                (below decision threshold of {thresh:.4f}).
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Multi-Model Ensemble Intelligence")
    cols = st.columns(len(preds))
    for col, (_, row) in zip(cols, preds.iterrows()):
        status = "ALERT" if row["attack_predicted"] else "SAFE"
        col.metric(
            label=f"{row['model']}",
            value=f"{row['probability']:.1%}",
            delta=f"{status} (θ={row['threshold']:.2f})",
            delta_color="inverse" if row["attack_predicted"] else "normal",
        )

    st.markdown("#### Model Attack Probability Comparison")
    chart_df = preds[["model", "probability"]].set_index("model")
    st.bar_chart(chart_df, color="#3b82f6")


def render_batch_predictions(
    preds: pd.DataFrame, input_df: pd.DataFrame, primary_model: str
):
    primary_preds = preds[preds["model"] == primary_model].copy()
    total_windows = len(primary_preds)
    alert_count = int(primary_preds["attack_predicted"].sum())
    alert_rate = (alert_count / max(total_windows, 1)) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Windows (10s)", f"{total_windows:,}")
    col2.metric(
        f"Active Alerts ({primary_model})",
        f"{alert_count:,}",
        delta="Threats Detected" if alert_count > 0 else "All Normal",
        delta_color="inverse" if alert_count > 0 else "normal",
    )
    col3.metric("Alert Incidence Rate", f"{alert_rate:.1f}%")
    col4.metric("Forecast Horizon", "120 Seconds")

    st.markdown("---")

    pivot = preds.pivot(index="row", columns="model", values="probability")
    pivot.columns = [f"{c}_prob" for c in pivot.columns]
    pivot["forecast_alert"] = primary_preds["attack_predicted"].values

    if "window_start" in input_df.columns:
        pivot.insert(0, "window_start", input_df["window_start"].values)
    if "Future_Attack_Target" in input_df.columns:
        pivot["actual_target"] = input_df["Future_Attack_Target"].values

    st.markdown("#### Attack Probability Timeline Across Windows")
    prob_cols = [c for c in pivot.columns if "_prob" in c]
    st.line_chart(pivot[prob_cols])

    st.markdown("#### Detailed Window Log")
    st.dataframe(
        pivot.style.format({col: "{:.4f}" for col in prob_cols}),
        use_container_width=True,
    )


def render_benchmarks():
    st.markdown("### 📊 Comprehensive Model Benchmarks & Comparison")
    st.write(
        "Evaluated on identical chronological test sequences under distribution shift (Feb 21 DDoS HOIC, Mar 01 Infiltration, Mar 02 Botnet)."
    )

    csv_path = "outputs/benchmark_results.csv"
    if os.path.exists(csv_path):
        df_bench = pd.read_csv(csv_path)

        st.markdown("#### Benchmark Metrics Table (Test Set)")
        styled_df = df_bench.style.highlight_max(
            subset=["ROC-AUC", "PR-AUC", "F1-Score", "Recall", "Specificity", "Balanced Acc"],
            color="#dcfce7",
        )
        st.dataframe(styled_df, use_container_width=True)

        k1, k2, k3, k4 = st.columns(4)
        best_roc = df_bench.loc[df_bench["ROC-AUC"].idxmax()]
        best_pr = df_bench.loc[df_bench["PR-AUC"].idxmax()]
        best_f1 = df_bench.loc[df_bench["F1-Score"].idxmax()]
        best_rec = df_bench.loc[df_bench["Recall"].idxmax()]

        k1.metric("Top ROC-AUC", f"{best_roc['ROC-AUC']:.4f}", best_roc["Model"])
        k2.metric("Top PR-AUC", f"{best_pr['PR-AUC']:.4f}", best_pr["Model"])
        k3.metric("Top F1-Score", f"{best_f1['F1-Score']:.4f}", best_f1["Model"])
        k4.metric("Top Recall", f"{best_rec['Recall']:.1%}", best_rec["Model"])

        st.markdown("---")
        st.markdown("#### Benchmark Visualizations")

        dash_path = "outputs/benchmark_summary_dashboard.png"
        pr_path = "outputs/benchmark_pr_curves.png"
        roc_path = "outputs/benchmark_roc_curves.png"
        bar_path = "outputs/benchmark_metrics_barchart.png"
        cm_path = "outputs/benchmark_confusion_matrices.png"

        view_opt = st.selectbox(
            "Select Benchmark View",
            [
                "Executive Summary Dashboard (Unified 4-Panel)",
                "Precision-Recall (PR) Curves",
                "Receiver Operating Characteristic (ROC) Curves",
                "Key Metrics Comparison Bar Chart",
                "Normalized Confusion Matrices",
            ],
        )

        if view_opt.startswith("Executive") and os.path.exists(dash_path):
            st.image(dash_path, use_container_width=True)
        elif view_opt.startswith("Precision") and os.path.exists(pr_path):
            st.image(pr_path, use_container_width=True)
        elif view_opt.startswith("Receiver") and os.path.exists(roc_path):
            st.image(roc_path, use_container_width=True)
        elif view_opt.startswith("Key") and os.path.exists(bar_path):
            st.image(bar_path, use_container_width=True)
        elif view_opt.startswith("Normalized") and os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True)

    else:
        st.info("Benchmark artifacts not found yet. Click below to execute the benchmark suite.")

    if st.button("🔄 Re-Run Full Benchmark Suite", type="secondary"):
        with st.spinner("Executing benchmark across all models..."):
            res = subprocess.run([sys.executable, "benchmark.py"], capture_output=True, text=True)
            if res.returncode == 0:
                st.success("Benchmark completed successfully! Refreshing dashboard...")
                st.rerun()
            else:
                st.error(f"Benchmark run failed: {res.stderr}")


def render_architecture():
    st.markdown("### 🔬 System Architecture & Research Design")
    st.markdown(
        """
    **VectorFlow** provides proactive, zero-trust network threat forecasting by treating network traffic as a continuous time series.

    #### Key Architectural Components:
    1. **Temporal Sequence BiLSTM with Attention (`models/lstm.pt`)**:
       - 2-layer Bidirectional LSTM backbone with LayerNorm stabilization.
       - Temporal attention pooling mechanism that weights critical precursor spikes across time steps.
       - Binary Focal Loss that prioritizes hard precursors over abundant background benign flows.
    2. **XGBoost & Random Forest Ensembles (`models/xgboost.ubj`, `models/randomforest.joblib`)**:
       - 400 gradient boosted / bagged trees trained with feature subsampling and minimum leaf constraints.
       - Class weighting to counteract the 6% positive imbalance.
    3. **Threshold Calibration Strategy**:
       - Standard 0.50 threshold frequently collapses under temporal distribution shift.
       - Precision-Recall curves on Validation calibrate decision thresholds to guarantee minimum recall constraints (≥50%).
    4. **Forecasting Horizon**:
       - Input: 10-second statistical flow windows.
       - Output: Imminent attack presence in the subsequent 120 seconds.
    """
    )


def main():
    scaler, logreg, rf, xgb, thresholds, lstm_net, lstm_scaler, lstm_thresh = get_artifacts()

    render_header()

    available_models = ["XGBoost", "RandomForest", "LogReg"]
    if lstm_net is not None:
        available_models.extend(["LSTM", "Ensemble (XGB+LSTM)"])

    st.sidebar.title("⚙️ VectorFlow Controls")
    st.sidebar.caption("Defense Configuration Panel")

    primary_model = st.sidebar.selectbox(
        "Primary Alert Model", available_models, index=0
    )

    use_calibration = st.sidebar.checkbox(
        "Use Calibrated Alert Thresholds",
        value=True,
        help="Applies optimal F1-tuned decision thresholds from the validation set.",
    )

    custom_thresh = None
    if not use_calibration:
        custom_thresh = st.sidebar.slider(
            "Manual Decision Threshold", 0.001, 0.999, 0.500, step=0.005
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Current Thresholds:**")
    for m in available_models:
        th = thresholds.get(m, 0.5)
        st.sidebar.markdown(f"- **{m}:** `{th:.4f}`")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "⚙️ Live Threat Simulator",
            "📂 Batch File Forecasting",
            "📊 Model Benchmarks",
            "🔬 Architecture & Research",
        ]
    )

    with tab1:
        st.markdown("#### Live Network Telemetry Simulator")
        st.write("Inject simulated 10-second traffic flow parameters to test real-time early warning capabilities.")

        sim_type = st.selectbox(
            "Pre-fill Threat Scenario",
            [
                "Baseline Normal Traffic",
                "Port Scan / Recon Phase",
                "RST Flood Precursor",
                "DDoS Spike Precursor",
            ],
        )

        inputs = FEATURE_DEFAULTS.copy()
        if sim_type == "Port Scan / Recon Phase":
            inputs["Dst Port_nunique"] = 450.0
            inputs["Dst Port_nunique_lag1"] = 320.0
            inputs["Dst Port_nunique_delta1"] = 130.0
            inputs["flow_count"] = 1200.0
        elif sim_type == "RST Flood Precursor":
            inputs["RST Flag Cnt_sum"] = 850.0
            inputs["RST Flag Cnt_sum_lag1"] = 720.0
            inputs["RST Flag Cnt_sum_delta1"] = 130.0
            inputs["Tot Fwd Pkts_sum"] = 8500.0
        elif sim_type == "DDoS Spike Precursor":
            inputs["Tot Fwd Pkts_sum"] = 15000.0
            inputs["TotLen Fwd Pkts_sum"] = 850000.0
            inputs["flow_count"] = 3500.0
            inputs["SYN Flag Cnt_sum"] = 1200.0

        with st.expander("Base Flow Features (Current 10-Second Window)", expanded=True):
            cols = st.columns(3)
            for i, f in enumerate(FEATURE_COLS[:15]):
                inputs[f] = cols[i % 3].number_input(
                    f, value=float(inputs[f]), format="%.2f"
                )

        with st.expander("Temporal Lag Features (Lag 1 to 3)", expanded=False):
            cols = st.columns(3)
            for i, f in enumerate(FEATURE_COLS[15:60]):
                inputs[f] = cols[i % 3].number_input(
                    f, value=float(inputs[f]), format="%.2f"
                )

        with st.expander("Velocity Delta Features (Current - Lag 1)", expanded=False):
            cols = st.columns(3)
            for i, f in enumerate(FEATURE_COLS[60:]):
                inputs[f] = cols[i % 3].number_input(
                    f, value=float(inputs[f]), format="%.2f"
                )

        if st.button("🚀 Forecast Pre-Attack Risk", type="primary", use_container_width=True):
            with st.spinner("Evaluating models..."):
                preds = predict(
                    inputs,
                    use_calibrated_threshold=use_calibration,
                    custom_threshold=custom_thresh,
                    include_lstm=True,
                )
            render_single_prediction(preds, primary_model)

    with tab2:
        st.markdown("#### Batch CSV Telemetry Ingestion")
        st.write("Upload aggregated flow CSV files to run automated time-series attack precursor forecasting.")

        file = st.file_uploader("Upload CSV Window File", type=["csv"])

        sample_path = "data/sample_for_upload.csv"
        if os.path.exists(sample_path) and st.button("Load Built-In Test Sample (50 Windows)"):
            sample_df = pd.read_csv(sample_path)
            with st.spinner("Executing batch inference..."):
                preds = predict(
                    sample_df,
                    use_calibrated_threshold=use_calibration,
                    custom_threshold=custom_thresh,
                    include_lstm=True,
                )
            render_batch_predictions(preds, sample_df, primary_model)

        elif file is not None:
            try:
                df = pd.read_csv(file)
                st.info(f"Loaded {len(df):,} windows from `{file.name}`")
                with st.spinner("Computing forecast probabilities..."):
                    preds = predict(
                        df,
                        use_calibrated_threshold=use_calibration,
                        custom_threshold=custom_thresh,
                        include_lstm=True,
                    )
                render_batch_predictions(preds, df, primary_model)
            except Exception as e:
                st.error(f"Inference error: {e}")

    with tab3:
        render_benchmarks()

    with tab4:
        render_architecture()


if __name__ == "__main__":
    main()
