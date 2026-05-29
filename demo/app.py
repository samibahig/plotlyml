"""
plotlyml — Interactive Demo
Hugging Face Spaces · Streamlit

Run locally:
    pip install plotlyml[demo]
    streamlit run demo/app.py
"""

import math
import random

import streamlit as st
import plotly.graph_objects as go

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from plotlyml import distribution_drift, model_disagreement, quantile_evolution

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="plotlyml — ML Visualization Primitives",
    page_icon="📊",
    layout="wide",
)

# ── helpers ───────────────────────────────────────────────────────────────────
def _gaussian(mean=0.0, std=1.0):
    """Box-Muller Gaussian sample (no numpy dependency)."""
    u, v = 0.0, 0.0
    while u == 0:
        u = random.random()
    while v == 0:
        v = random.random()
    return math.sqrt(-2.0 * math.log(u)) * math.cos(2 * math.pi * v) * std + mean

def generate_samples(mean, std, n=800):
    return [_gaussian(mean, std) for _ in range(n)]

def generate_disagreement_data(n=300):
    xs, ys, preds_list = [], [], []
    for _ in range(n):
        cluster = 1 if random.random() > 0.5 else -1
        x = _gaussian(cluster * 2, 1.5)
        y = _gaussian(cluster * 2, 1.5)
        dist = math.sqrt(x * x + y * y)
        unc = max(0.0, 1 - dist / 3)
        true_label = 1 if cluster > 0 else 0
        preds = [
            max(0, min(1, true_label + _gaussian(0, unc * 0.5 + 0.1)))
            for _ in range(5)
        ]
        xs.append(x)
        ys.append(y)
        preds_list.append(preds)
    return xs, ys, preds_list

def generate_quantile_data(weeks=52):
    timestamps, p10s, p25s, p50s, p75s, p90s, means = [], [], [], [], [], [], []
    current_mean, current_var = 100.0, 10.0
    from datetime import datetime, timedelta
    start = datetime.now() - timedelta(weeks=weeks)
    for w in range(weeks):
        current_mean += _gaussian(0.5, 1)
        if 20 < w < 30:
            current_var += _gaussian(2, 0.5)
        elif w >= 30:
            current_var = max(10.0, current_var - 1)
        samples = sorted([_gaussian(current_mean, current_var) for _ in range(100)])
        ts = (start + timedelta(weeks=w)).strftime("%Y-%m-%d")
        timestamps.append(ts)
        p10s.append(samples[10])
        p25s.append(samples[25])
        p50s.append(samples[50])
        p75s.append(samples[75])
        p90s.append(samples[90])
        means.append(sum(samples) / len(samples))
    return timestamps, p10s, p25s, p50s, p75s, p90s, means


# ── sidebar nav ───────────────────────────────────────────────────────────────
st.sidebar.title("plotlyml")
st.sidebar.caption("ML Monitoring Visualization Primitives")
page = st.sidebar.radio(
    "Navigate",
    ["Gallery", "Distribution Drift", "Model Disagreement", "Quantile Evolution"],
)
st.sidebar.divider()
st.sidebar.markdown(
    "[![GitHub](https://img.shields.io/badge/GitHub-plotlyml-181717?logo=github)](https://github.com/samibahig/plotlyml)"
)

# ─────────────────────────────────────────────────────────────────────────────
# GALLERY
# ─────────────────────────────────────────────────────────────────────────────
if page == "Gallery":
    st.title("Visual Primitives for ML Monitoring")
    st.markdown(
        "Three high-level **Plotly** functions designed in the spirit of `plotly.express`, "
        "filling the gaps in ML observability workflows."
    )
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Distribution Drift")
        st.markdown(
            "Compare training vs. live distributions. "
            "KL divergence score surfaces feature drift before it degrades model performance."
        )
        st.code("from plotlyml import distribution_drift\nfig = distribution_drift(ref, cur)\nfig.show()", language="python")
    with col2:
        st.subheader("Model Disagreement")
        st.markdown(
            "Map ensemble variance per sample in a 2-D feature space. "
            "High-disagreement regions highlight uncertainty hot-spots for active learning or auditing."
        )
        st.code("from plotlyml import model_disagreement\nfig = model_disagreement(x, y, preds)\nfig.show()", language="python")
    with col3:
        st.subheader("Quantile Evolution")
        st.markdown(
            "Track P10/P50/P90 bands over time. "
            "Catches regime changes and volatility spikes that the mean alone would miss."
        )
        st.code("from plotlyml import quantile_evolution\nfig = quantile_evolution(dates, p50=med)\nfig.show()", language="python")

    st.divider()
    st.markdown("**Select a chart from the sidebar to explore the interactive demo.**")


# ─────────────────────────────────────────────────────────────────────────────
# DISTRIBUTION DRIFT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Distribution Drift":
    st.title("Distribution Drift Plot")
    st.caption("Compare two distributions and surface divergence — training vs. inference, A/B test groups, pre/post deployment.")

    with st.sidebar:
        st.subheader("Simulation Controls")
        mean_shift  = st.slider("Current Mean Shift",  -2.0, 3.0, 0.8, 0.05)
        std_scale   = st.slider("Current Std Scale",    0.5, 3.0, 1.2, 0.05)
        threshold   = st.slider("KL Divergence Threshold", 0.01, 1.0, 0.1, 0.01)
        n_samples   = st.slider("Samples per distribution", 200, 2000, 800, 100)

    ref = generate_samples(0, 1, n_samples)
    cur = generate_samples(mean_shift, std_scale, n_samples)

    fig = distribution_drift(
        ref, cur,
        divergence_threshold=threshold,
        reference_name="Reference (Training)",
        current_name="Current (Inference)",
        title="Feature Distribution: Age",
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Component API"):
        st.code("""
from plotlyml import distribution_drift

fig = distribution_drift(
    reference,               # list | np.ndarray | pd.Series
    current,                 # list | np.ndarray | pd.Series
    bins=50,                 # histogram bins
    divergence_threshold=0.1,# KL alert threshold
    reference_name="Reference",
    current_name="Current",
    x_label="Feature Value",
    title=None,              # auto-generated if omitted
    template="plotly_dark",
)
fig.show()
""", language="python")

    with st.expander("Use cases"):
        st.markdown("""
- **Feature store monitoring** — detect when a feature's live distribution drifts from the training snapshot
- **A/B test sanity check** — confirm groups are comparable before reading metrics
- **Post-deployment audit** — flag model inputs that fall outside the training domain
- **Data pipeline QA** — catch upstream ETL regressions that shift numerical columns
""")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DISAGREEMENT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Model Disagreement":
    st.title("Model Disagreement Map")
    st.caption("Visualise per-sample ensemble variance in a 2-D feature space. High variance = high uncertainty.")

    with st.sidebar:
        st.subheader("Controls")
        threshold = st.slider("Alert Threshold (Variance)", 0.005, 0.2, 0.05, 0.005, format="%.3f")
        n_samples = st.slider("Number of samples", 100, 500, 300, 50)

    xs, ys, preds_list = generate_disagreement_data(n_samples)

    fig = model_disagreement(
        xs, ys, preds_list,
        threshold=threshold,
        x_label="UMAP Dim 1",
        y_label="UMAP Dim 2",
        title="Ensemble Disagreement Map",
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Component API"):
        st.code("""
from plotlyml import model_disagreement

fig = model_disagreement(
    x,                    # 1-D array: reduced feature dim 1
    y,                    # 1-D array: reduced feature dim 2
    predictions,          # 2-D array: shape (n_samples, n_models)
    sample_ids=None,      # optional list of string IDs
    true_labels=None,     # optional ground-truth labels
    threshold=0.05,       # variance alert threshold
    x_label="Dim 1",
    y_label="Dim 2",
    colorscale="Viridis",
    template="plotly_dark",
)
fig.show()
""", language="python")

    with st.expander("Use cases"):
        st.markdown("""
- **Active learning** — prioritise labelling samples where the ensemble disagrees most
- **Uncertainty-aware deployment** — flag high-variance samples for human review at inference time
- **Model debugging** — identify clusters where the ensemble is systematically confused
- **Distribution shift detection** — high-disagreement regions often correlate with OOD inputs
""")


# ─────────────────────────────────────────────────────────────────────────────
# QUANTILE EVOLUTION
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Quantile Evolution":
    st.title("Quantile Evolution Plot")
    st.caption("Track P10/P50/P90 bands over time — catch regime changes and volatility spikes that the mean alone misses.")

    with st.sidebar:
        st.subheader("Controls")
        show_mean = st.toggle("Overlay Mean Line", value=True)
        show_inner = st.toggle("Show P25–P75 Band", value=True)
        weeks = st.slider("Number of weeks", 12, 104, 52, 4)

    timestamps, p10s, p25s, p50s, p75s, p90s, means = generate_quantile_data(weeks)

    fig = quantile_evolution(
        timestamps,
        p50=p50s,
        p10=p10s,
        p90=p90s,
        p25=p25s if show_inner else None,
        p75=p75s if show_inner else None,
        mean=means,
        show_mean=show_mean,
        y_label="Metric Value",
        title="Metric Quantile Evolution",
    )
    st.plotly_chart(fig, use_container_width=True)

    if weeks >= 12:
        st.subheader("Distribution Snapshots")
        snap_cols = st.columns(3)
        indices = [weeks // 6, weeks // 2, int(weeks * 0.85)]
        for col, idx in zip(snap_cols, indices):
            with col:
                ts = timestamps[idx]
                st.markdown(f"**{ts}**")
                snap = go.Figure(go.Box(
                    q1=[p25s[idx]], median=[p50s[idx]],
                    q3=[p75s[idx]], lowerfence=[p10s[idx]],
                    upperfence=[p90s[idx]],
                    name=ts,
                    marker_color="#3b82f6",
                    fillcolor="rgba(59,130,246,0.3)",
                ))
                snap.update_layout(
                    template="plotly_dark",
                    height=200,
                    margin=dict(t=10, b=10, l=10, r=10),
                    showlegend=False,
                )
                st.plotly_chart(snap, use_container_width=True)

    with st.expander("Component API"):
        st.code("""
from plotlyml import quantile_evolution

fig = quantile_evolution(
    timestamps,            # list of ISO strings, labels, or numbers
    p50=median_values,     # required: median at each timestamp
    p10=p10_values,        # optional: outer lower band
    p90=p90_values,        # optional: outer upper band
    p25=p25_values,        # optional: IQR lower bound
    p75=p75_values,        # optional: IQR upper bound
    mean=mean_values,      # optional: mean line
    show_mean=False,
    volatility_multiplier=1.5,  # spread ratio to annotate as high-volatility
    template="plotly_dark",
)
fig.show()
""", language="python")

    with st.expander("Use cases"):
        st.markdown("""
- **Service reliability** — detect when P99 latency spikes while P50 stays flat (tail latency problem)
- **ML prediction confidence** — track how certainty evolves post-deployment
- **Clinical trials** — compare endpoint distributions across cohorts over time
- **Financial risk** — monitor VaR and CVaR evolution under changing market conditions
- **Data quality** — early warning when quantile spread increases (upstream instability)
""")
