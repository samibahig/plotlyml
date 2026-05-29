"""
Distribution Drift — worked example
====================================
Demonstrates the full API including numpy arrays and pandas Series inputs.
"""
import numpy as np
import pandas as pd
from plotlyml import distribution_drift

rng = np.random.default_rng(42)

# ── 1. Basic usage ────────────────────────────────────────────────────────────
reference = rng.normal(0, 1, 1000)          # training-time samples
current   = rng.normal(0.8, 1.3, 1000)     # inference-time samples (drifted)

fig = distribution_drift(
    reference, current,
    divergence_threshold=0.1,
    title="Age Feature — Training vs. Live",
    x_label="Age (normalised)",
)
fig.show()

# ── 2. Using pandas Series ────────────────────────────────────────────────────
df_train = pd.DataFrame({"age": rng.normal(35, 10, 500)})
df_live  = pd.DataFrame({"age": rng.normal(42, 12, 500)})

fig2 = distribution_drift(
    df_train["age"],
    df_live["age"],
    reference_name="Training (2023)",
    current_name="Live (2024)",
    bins=40,
    title="User Age Distribution Shift",
)
fig2.show()

# ── 3. No-drift baseline (expect low KL score) ────────────────────────────────
fig3 = distribution_drift(
    rng.normal(0, 1, 1000),
    rng.normal(0, 1, 1000),
    divergence_threshold=0.05,
    title="Stable Feature — No Drift Expected",
)
fig3.show()
