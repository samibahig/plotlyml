"""
Quantile Evolution Plot — worked example
==========================================
Demonstrates usage with pandas DataFrames and pre-computed quantiles.
"""
import numpy as np
import pandas as pd
from plotlyml import quantile_evolution

rng = np.random.default_rng(7)

# ── 1. Generate weekly data with a simulated regime change ────────────────────
weeks = 52
dates = pd.date_range("2024-01-01", periods=weeks, freq="W").strftime("%Y-%m-%d").tolist()

mean_path = np.cumsum(rng.normal(0.5, 1, weeks)) + 100
spread = np.abs(rng.normal(8, 2, weeks))
# Volatility spike weeks 20-28
spread[20:28] *= 2.5

p50 = mean_path
p10 = p50 - 2.0 * spread
p90 = p50 + 2.0 * spread
p25 = p50 - spread
p75 = p50 + spread
mean_line = mean_path + rng.normal(0, 0.5, weeks)

fig = quantile_evolution(
    dates,
    p50=p50,
    p10=p10, p90=p90,
    p25=p25, p75=p75,
    mean=mean_line,
    show_mean=True,
    volatility_multiplier=1.5,
    y_label="Prediction Confidence Score",
    title="Model Confidence — Weekly Quantile Evolution",
)
fig.show()

# ── 2. From a DataFrame with raw observations ─────────────────────────────────
# (More realistic: each row is a raw observation, then aggregate)
records = []
for w, date in enumerate(dates):
    current_mean = 100 + w * 0.5
    current_std  = 10 + (5 if 20 <= w < 28 else 0)
    obs = rng.normal(current_mean, current_std, 200)
    records.append({
        "week":  date,
        "p10":   np.percentile(obs, 10),
        "p25":   np.percentile(obs, 25),
        "p50":   np.percentile(obs, 50),
        "p75":   np.percentile(obs, 75),
        "p90":   np.percentile(obs, 90),
        "mean":  obs.mean(),
    })

df = pd.DataFrame(records)

fig2 = quantile_evolution(
    df["week"].tolist(),
    p50=df["p50"].tolist(),
    p10=df["p10"].tolist(),
    p90=df["p90"].tolist(),
    p25=df["p25"].tolist(),
    p75=df["p75"].tolist(),
    mean=df["mean"].tolist(),
    show_mean=True,
    title="Service Latency Quantiles (from raw observations)",
    y_label="Latency (ms)",
)
fig2.show()
