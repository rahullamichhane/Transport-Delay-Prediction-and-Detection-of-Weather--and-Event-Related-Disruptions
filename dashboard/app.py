from __future__ import annotations
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.data_generator import TransportDataGenerator
from src.models import MODEL_CONFIGS, train_all_models
from src.preprocessing import build_preprocessor, prepare_features, split_data
from src.streaming import StreamingPredictor, simulate_stream

OUTPUT_DIR = Path("output")
MODEL_DIR = OUTPUT_DIR / "models"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"

VEHICLE_CLASS_COLORS = {
    "bus": "#1f77b4",
    "streetcar": "#ff7f0e",
    "subway": "#2ca02c",
}

st.set_page_config(
    page_title="TTC Delay Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Professional CSS

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f3b57;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #5a6b7d;
        margin-bottom: 2rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, #1f3b57 0%, #2a527e 100%);
        padding: 1.25rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .kpi-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .section-title {
        font-size: 1.4rem;
        font-weight: 600;
        color: #1f3b57;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
        padding-left: 0.75rem;
    }
    .vehicle-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        color: white;
        margin-right: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _ensure_models() -> None:
    """Train + save models if no saved pipelines exist in output/models/."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not PREPROCESSOR_PATH.exists():
        with st.spinner("Training models for the first time (this may take a minute)..."):
            gen = TransportDataGenerator(seed=42)
            df = gen.generate_historical(n_records=6000)
            X, y = prepare_features(df)
            X_train, X_test, y_train, y_test = split_data(X, y)
            preprocessor = build_preprocessor()
            pipelines, _ = train_all_models(
                preprocessor, X_train, y_train, X_test, y_test
            )
            joblib.dump(preprocessor, PREPROCESSOR_PATH)
            for name, pipe in pipelines.items():
                safe = name.lower().replace(" ", "_")
                joblib.dump(pipe, MODEL_DIR / f"{safe}.pkl")


def _load_pipelines() -> dict:
    pipelines = {}
    for name in MODEL_CONFIGS:
        safe = name.lower().replace(" ", "_")
        p = MODEL_DIR / f"{safe}.pkl"
        if p.exists():
            pipelines[name] = joblib.load(p)
    return pipelines


def _vehicle_badge(vehicle_class: str) -> str:
    color = VEHICLE_CLASS_COLORS.get(vehicle_class, "#95a5a6")
    return (
        f'<span class="vehicle-badge" style="background-color:{color}">'
        f"{vehicle_class.title()}</span>"
    )


def main() -> None:
    st.markdown('<div class="main-header"> TTC Delay Detection</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Public Transport Delay Detection with Weather & Events | Live Streaming | Vehicle Classification</div>',
        unsafe_allow_html=True,
    )

    _ensure_models()
    pipelines = _load_pipelines()
    preprocessor = (
        joblib.load(PREPROCESSOR_PATH) if PREPROCESSOR_PATH.exists() else None
    )

    if not pipelines or preprocessor is None:
        st.error("Models not available. Run `python main.py` to train first.")
        return

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------
    st.sidebar.header("⚙️ Stream Controls")
    n_records = st.sidebar.slider("Records to stream", 50, 2000, 500, step=50)
    speed = st.sidebar.selectbox("Streaming speed", ["Fast", "Medium", "Slow"], index=0)
    delay_map = {"Fast": 0.0, "Medium": 0.01, "Slow": 0.05}
    run_stream = st.sidebar.button("▶ Start / Restart Live Stream", type="primary")

    model_names = list(pipelines.keys())
    selected_models = st.sidebar.multiselect(
        "Models to display", model_names, default=model_names
    )

    show_vehicle_class = st.sidebar.multiselect(
        "Filter by vehicle class",
        ["Bus", "Streetcar", "Subway"],
        default=["Bus", "Streetcar", "Subway"],
    )

   
    # Session state
  
    if "history" not in st.session_state or run_stream:
        st.session_state.history = []
        st.session_state.predictor = StreamingPredictor(window_size=200)
        for name in model_names:
            st.session_state.predictor.register_model(name)
        st.session_state.streamed = False

    predictor: StreamingPredictor = st.session_state.predictor

   
    # Run live stream

    if run_stream or not st.session_state.streamed:
        gen = TransportDataGenerator(seed=123)
        records = gen.stream_records(n_records=n_records)
        results = simulate_stream(
            records, pipelines, preprocessor, predictor, delay_seconds=delay_map[speed]
        )
        st.session_state.history = results
        st.session_state.streamed = True

    history = predictor.to_dataframe()
    if history.empty:
        st.info("Click '▶ Start / Restart Live Stream' to begin live detection.")
        return

    # Apply vehicle class filter
    if show_vehicle_class and "vehicle_class" in history.columns:
        history = history[history["vehicle_class"].str.title().isin(show_vehicle_class)]

  
    # KPI row
  
    st.markdown('<div class="section-title"> Live Stream KPIs</div>', unsafe_allow_html=True)
    total = len(history)
    true_delayed = int(history["delay_label"].sum()) if "delay_label" in history else 0
    delay_rate = (true_delayed / total * 100) if total else 0.0
    avg_delay = history["delay_minutes"].mean() if "delay_minutes" in history else 0.0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-value">{total}</div>'
            f'<div class="kpi-label">Records Streamed</div></div>',
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-value">{true_delayed}</div>'
            f'<div class="kpi-label">Delayed Vehicles</div></div>',
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-value">{delay_rate:.1f}%</div>'
            f'<div class="kpi-label">Delay Rate</div></div>',
            unsafe_allow_html=True,
        )
    with k4:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-value">{avg_delay:.1f}</div>'
            f'<div class="kpi-label">Avg Delay (min)</div></div>',
            unsafe_allow_html=True,
        )

  
    # Per-model delay rates
   
    st.markdown('<div class="section-title"> Model Delay Rates (rolling window)</div>', unsafe_allow_html=True)
    if selected_models:
        cols = st.columns(min(len(selected_models), 4))
        for i, name in enumerate(selected_models):
            rate = predictor.recent_delay_rate(name)
            with cols[i % len(cols)]:
                st.metric(name, f"{rate:.2%}")

  
    # Charts
  
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title"> Delay Minutes Over Time</div>', unsafe_allow_html=True)
        if "timestamp" in history.columns and "delay_minutes" in history.columns:
            h = history.copy()
            h["timestamp"] = pd.to_datetime(h["timestamp"])
            fig = px.line(
                h.sort_values("timestamp"),
                x="timestamp",
                y="delay_minutes",
                color="vehicle_class" if "vehicle_class" in h.columns else None,
                color_discrete_map=VEHICLE_CLASS_COLORS,
                title="Live Delay Minutes by Vehicle Class",
            )
            fig.update_layout(hovermode="x unified", height=350)
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title"> Predicted Delay Share by Model</div>', unsafe_allow_html=True)
        if selected_models:
            shares = {name: predictor.recent_delay_rate(name) for name in selected_models}
            fig2 = go.Figure(
                data=[
                    go.Bar(
                        x=list(shares.keys()),
                        y=list(shares.values()),
                        marker_color="#1f77b4",
                    )
                ]
            )
            fig2.update_layout(
                yaxis_tickformat=".0%",
                title="",
                height=350,
                xaxis_tickangle=-15,
            )
            st.plotly_chart(fig2, use_container_width=True)

   
    # Vehicle class breakdown
   
    st.markdown('<div class="section-title"> Vehicle Class Breakdown</div>', unsafe_allow_html=True)
    if "vehicle_class" in history.columns:
        vc_counts = history["vehicle_class"].value_counts()
        vc_delayed = history[history["delay_label"] == 1]["vehicle_class"].value_counts()
        vc_df = pd.DataFrame(
            {
                "Vehicle Class": vc_counts.index,
                "Total Records": vc_counts.values,
                "Delayed": [
                    int(vc_delayed.get(cls, 0)) for cls in vc_counts.index
                ],
            }
        )
        vc_df["Delay Rate"] = (vc_df["Delayed"] / vc_df["Total Records"] * 100).round(1)

        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(vc_df, use_container_width=True, hide_index=True)
        with c2:
            fig3 = px.pie(
                vc_df,
                names="Vehicle Class",
                values="Total Records",
                color="Vehicle Class",
                color_discrete_map=VEHICLE_CLASS_COLORS,
                title="Records by Vehicle Class",
                height=350,
            )
            st.plotly_chart(fig3, use_container_width=True)

   
    # Recent records table
   
    st.markdown('<div class="section-title"> Recent Detections</div>', unsafe_allow_html=True)
    display_cols = [
        c
        for c in [
            "timestamp",
            "vehicle_id",
            "vehicle_class",
            "route_id",
            "stop_id",
            "weather_condition",
            "event_type",
            "is_peak_hour",
            "delay_minutes",
            "delay_label",
        ]
        if c in history.columns
    ] + [f"pred_{m.replace(' ', '_')}" for m in selected_models if f"pred_{m.replace(' ', '_')}" in history.columns]

    show_df = history.tail(15)[display_cols].copy()
    if "delay_label" in show_df.columns:
        show_df["delay_label"] = show_df["delay_label"].map({0: "On Time", 1: "DELAYED"})
    if "vehicle_class" in show_df.columns:
        show_df["vehicle_class"] = show_df["vehicle_class"].apply(_vehicle_badge)
    st.dataframe(show_df, use_container_width=True, hide_index=True)

    st.caption(
        "This is a simulated live stream of TTC records with vehicle classification and delay detection. The models are trained on synthetic data and may not reflect real-world performance."
    )


if __name__ == "__main__":
    main()
