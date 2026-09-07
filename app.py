"""
app.py — VectorFlow Network Attack Forecaster
Interactive Streamlit Dashboard for Hackathon Demonstration (SIH PS 26153).
"""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from inference import FEATURE_COLS, get_artifacts, predict

st.set_page_config(
    page_title="VectorFlow — Early Attack Forecaster",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Clean CSS Styling
st.markdown(
    """
<style>
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .alert-banner-danger {
        background-color: #ffeef0;
        border-left: 6px solid #dc3545;
        padding: 16px;
        border-radius: 4px;
        margin-bottom: 20px;
    }
    .alert-banner-safe {
        background-color: #e6f4ea;
        border-left: 6px solid #28a745;
        padding: 16px;
        border-radius: 4px;
        margin-bottom: 20px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Default baseline values for manual simulation
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


def render_single_prediction(preds: pd.DataFrame, primary_model: str):
    primary = preds[preds["model"] == primary_model].iloc[0]
    prob = primary["probability"]
    thresh = primary["threshold"]
    is_attack = primary["attack_predicted"]

    if is_attack:
        st.markdown(
            f"""
        <div class="alert-banner-danger">
            <h3 style="color:#721c24; margin:0;">🚨 WARNING: IMMINENT ATTACK FORECASTED (Next 120 Seconds)</h3>
            <p style="margin:6px 0 0 0; color:#721c24;">
                <b>{primary_model}</b> triggered alert with probability <b>{prob:.2%}</b> 
                (threshold: {thresh:.4f}). Immediate traffic shaping or firewall rule engagement recommended.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
        <div class="alert-banner-safe">
            <h3 style="color:#155724; margin:0;">✅ NORMAL NETWORK TRAFFIC (No Attack Imminent)</h3>
            <p style="margin:6px 0 0 0; color:#155724;">
                <b>{primary_model}</b> estimated attack probability at <b>{prob:.2%}</b> 
                (below threshold of {thresh:.4f}).
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.subheader("Model Ensemble Breakdown")
    c1, c2, c3 = st.columns(3)
    for col, m_name in zip([c1, c2, c3], ["XGBoost", "RandomForest", "LogReg"]):
        row = preds[preds["model"] == m_name].iloc[0]
        status = "ALERT" if row["attack_predicted"] else "SAFE"
        col.metric(
            label=f"{m_name} (Threshold: {row['threshold']:.4f})",
            value=f"{row['probability']:.2%}",
            delta=status,
            delta_color="inverse" if row["attack_predicted"] else "normal",
        )

    chart_df = preds[["model", "probability"]].set_index("model")
    st.bar_chart(chart_df)


def render_batch_predictions(
    preds: pd.DataFrame, input_df: pd.DataFrame, primary_model: str
):
    st.subheader(f"Batch Forecasting Analysis — {preds['row'].nunique()} Windows")

    primary_preds = preds[preds["model"] == primary_model].copy()
    total_windows = len(primary_preds)
    alert_count = int(primary_preds["attack_predicted"].sum())
    alert_rate = (alert_count / max(total_windows, 1)) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Analyzed 10-Second Windows", f"{total_windows:,}")
    col2.metric(
        f"Early Alerts ({primary_model})",
        f"{alert_count:,}",
        delta="Attacks Detected" if alert_count > 0 else "All Normal",
        delta_color="inverse" if alert_count > 0 else "normal",
    )
    col3.metric("Alert Ratio", f"{alert_rate:.1f}%")

    st.markdown("---")

    pivot = preds.pivot(index="row", columns="model", values="probability")
    pivot.columns = [f"{c}_prob" for c in pivot.columns]
    pivot["forecast_alert"] = primary_preds["attack_predicted"].values

    if "window_start" in input_df.columns:
        pivot.insert(0, "window_start", input_df["window_start"].values)
    if "Future_Attack_Target" in input_df.columns:
        pivot["actual_target"] = input_df["Future_Attack_Target"].values

    st.dataframe(
        pivot.style.format({col: "{:.4f}" for col in pivot.columns if "_prob" in col}),
        use_container_width=True,
    )

    st.line_chart(pivot[[col for col in pivot.columns if "_prob" in col]])


def main():
    # Load calibrated thresholds from disk
    _, _, thresholds = get_artifacts()

    # Sidebar Controls
    st.sidebar.title("🛡️ VectorFlow Controls")
    st.sidebar.caption("SIH 2026 — Problem Statement 26153")

    primary_model = st.sidebar.selectbox(
        "Primary Decision Model", ["XGBoost", "RandomForest", "LogReg"], index=0
    )

    use_calibration = st.sidebar.checkbox(
        "Use Calibrated Alert Thresholds",
        value=True,
        help="Applies optimal F1-tuned decision thresholds from the validation set instead of the default 0.50.",
    )

    custom_thresh = None
    if not use_calibration:
        custom_thresh = st.sidebar.slider(
            "Manual Decision Threshold", 0.001, 0.999, 0.500, step=0.005
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""
    **Current Operational Thresholds:**
    - **XGBoost:** `{thresholds.get("XGBoost", 0.5):.4f}`
    - **Random Forest:** `{thresholds.get("RandomForest", 0.5):.4f}`
    - **Logistic Regression:** `{thresholds.get("LogReg", 0.5):.4f}`
    """
    )

    st.title("VectorFlow — Network Attack Forecasting Dashboard")
    st.write(
        "Anticipate network intrusions **120 seconds in advance** before attack payloads land, "
        "using 10-second flow statistical windows, temporal lags, and velocity deltas."
    )

    mode = st.radio(
        "Data Ingestion Mode",
        ["📂 Batch File Upload (CSV)", "⚙️ Live Traffic Simulator (Manual)"],
        horizontal=True,
    )

    st.markdown("---")

    if mode.startswith("📂 Batch"):
        st.markdown(
            "Upload any preprocessed CSV window file (or an export from our Polars aggregation pipeline)."
        )
        file = st.file_uploader("Upload CSV", type=["csv"])

        # Quick Load Sample Button
        sample_path = "data/sample_for_upload.csv"
        if os.path.exists(sample_path) and st.button(
            "Load Built-In Test Sample (50 Windows)"
        ):
            sample_df = pd.read_csv(sample_path)
            with st.spinner("Executing real-time inference..."):
                preds = predict(
                    sample_df,
                    use_calibrated_threshold=use_calibration,
                    custom_threshold=custom_thresh,
                )
            render_batch_predictions(preds, sample_df, primary_model)

        elif file is not None:
            try:
                df = pd.read_csv(file)
                st.info(f"Loaded {len(df):,} windows from `{file.name}`")
                with st.spinner("Computing early warning probabilities..."):
                    preds = predict(
                        df,
                        use_calibrated_threshold=use_calibration,
                        custom_threshold=custom_thresh,
                    )
                render_batch_predictions(preds, df, primary_model)
            except Exception as e:
                st.error(f"Inference error: {e}")

    else:
        st.markdown(
            "Simulate custom network conditions. Defaults reflect typical network baseline states."
        )

        sim_type = st.selectbox(
            "Pre-fill Traffic Scenario",
            [
                "Baseline Normal Traffic",
                "Port Scan / Recon Phase",
                "RST Flood Attack Precursor",
            ],
        )

        inputs = FEATURE_DEFAULTS.copy()
        if sim_type == "Port Scan / Recon Phase":
            inputs["Dst Port_nunique"] = 450.0
            inputs["Dst Port_nunique_lag1"] = 320.0
            inputs["Dst Port_nunique_delta1"] = 130.0
            inputs["flow_count"] = 1200.0
        elif sim_type == "RST Flood Attack Precursor":
            inputs["RST Flag Cnt_sum"] = 850.0
            inputs["RST Flag Cnt_sum_lag1"] = 720.0
            inputs["RST Flag Cnt_sum_delta1"] = 130.0
            inputs["Tot Fwd Pkts_sum"] = 8500.0

        with st.expander("Base Features (Current 10-Second Window)", expanded=True):
            cols = st.columns(3)
            for i, f in enumerate(FEATURE_COLS[:15]):
                inputs[f] = cols[i % 3].number_input(
                    f, value=float(inputs[f]), format="%.2f"
                )

        with st.expander("Historical Lag Features (Lag 1-3)", expanded=False):
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

        if st.button(
            "🚀 Forecast Imminent Attack Risk", type="primary", use_container_width=True
        ):
            with st.spinner("Evaluating multi-model ensemble..."):
                preds = predict(
                    inputs,
                    use_calibrated_threshold=use_calibration,
                    custom_threshold=custom_thresh,
                )
            render_single_prediction(preds, primary_model)


if __name__ == "__main__":
    main()
