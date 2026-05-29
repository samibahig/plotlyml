# plotlyml

**ML Monitoring Visualization Primitives for Plotly**

Three high-level functions — designed in the spirit of `plotly.express` — that fill real gaps in the modern ML observability toolkit.

[![CI](https://github.com/samibahig/plotlyml/actions/workflows/ci.yml/badge.svg)](https://github.com/samibahig/plotlyml/actions)
[![PyPI](https://img.shields.io/pypi/v/plotlyml)](https://pypi.org/project/plotlyml/)
[![HuggingFace Demo](https://img.shields.io/badge/demo-HuggingFace%20Spaces-orange)](https://huggingface.co/spaces/samibahig/plotlyml-demo)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## The primitives

| Function | What it does | When to use it |
|---|---|---|
| `distribution_drift` | Overlays two distributions and scores their divergence (KL) | Detect feature drift between training and live data |
| `model_disagreement` | Maps per-sample ensemble variance in a 2-D feature space | Identify uncertainty hot-spots, prioritise active learning |
| `quantile_evolution` | Tracks P10/P50/P90 bands over time with volatility annotations | Monitor metric distributions — not just the mean — over time |

---

## Installation

```bash
pip install plotlyml
```

For the Streamlit demo locally:

```bash
pip install plotlyml[demo]
streamlit run demo/app.py
```

---

## Quick start

```python
import numpy as np
from plotlyml import distribution_drift, model_disagreement, quantile_evolution

# Distribution Drift ─────────────────────────────────────────────────────────
ref = np.random.normal(0, 1, 1000)       # training distribution
cur = np.random.normal(0.8, 1.3, 1000)  # live distribution (drifted)

fig = distribution_drift(ref, cur, divergence_threshold=0.1)
fig.show()

# Model Disagreement ─────────────────────────────────────────────────────────
rng = np.random.default_rng(0)
n   = 300
x, y   = rng.normal(size=n), rng.normal(size=n)
preds  = rng.random((n, 5))   # 5 model predictions per sample

fig = model_disagreement(x, y, preds, threshold=0.05)
fig.show()

# Quantile Evolution ─────────────────────────────────────────────────────────
import pandas as pd
dates = pd.date_range("2024-01-01", periods=52, freq="W").strftime("%Y-%m-%d").tolist()
base  = np.cumsum(rng.normal(0.5, 1, 52)) + 100
spread = np.abs(rng.normal(8, 2, 52))

fig = quantile_evolution(
    dates,
    p50=base,
    p10=base - 2*spread, p90=base + 2*spread,
    p25=base - spread,   p75=base + spread,
    show_mean=True, mean=base + rng.normal(0, 0.5, 52),
)
fig.show()
```

---

## API reference

### `distribution_drift(reference, current, *, ...)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `reference` | array-like | — | Baseline / training-time samples |
| `current` | array-like | — | Live / inference-time samples |
| `bins` | int | 50 | Histogram bins |
| `divergence_threshold` | float | 0.1 | KL score that triggers alert colouring |
| `reference_name` | str | `"Reference"` | Legend label |
| `current_name` | str | `"Current"` | Legend label |
| `template` | str | `"plotly_dark"` | Plotly template |

Returns `plotly.graph_objects.Figure`.

---

### `model_disagreement(x, y, predictions, *, ...)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `x` | array-like | — | Reduced feature dim 1 per sample |
| `y` | array-like | — | Reduced feature dim 2 per sample |
| `predictions` | 2-D array-like | — | Shape `(n_samples, n_models)` |
| `sample_ids` | list[str] | auto | Hover tooltip identifiers |
| `true_labels` | array-like | None | Shown in hover when provided |
| `threshold` | float | 0.05 | Variance alert threshold |
| `colorscale` | str | `"Viridis"` | Plotly colorscale name |

Returns `plotly.graph_objects.Figure`.

---

### `quantile_evolution(timestamps, p50, *, ...)`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `timestamps` | array-like | — | ISO strings, labels, or numbers |
| `p50` | array-like | — | Median at each timestamp (**required**) |
| `p10` / `p90` | array-like | None | Outer ribbon edges |
| `p25` / `p75` | array-like | None | IQR ribbon edges |
| `mean` | array-like | None | Mean line (dashed) |
| `show_mean` | bool | False | Whether to render the mean |
| `volatility_multiplier` | float | 1.5 | Spread ratio triggering annotation |

Returns `plotly.graph_objects.Figure`.

---

## Why these primitives?

The standard Plotly/Express library covers individual distributions (`px.histogram`),
individual time series (`px.line`), and scatter plots (`px.scatter`).  What it lacks
is **first-class support for comparison and uncertainty** — the two things that matter
most in production ML systems.

These three functions are domain-agnostic.  They are useful in:

- **ML monitoring** — drift detection, ensemble uncertainty, confidence evolution
- **Healthcare analytics** — cohort distribution comparisons, clinical endpoint tracking
- **Finance** — VaR/CVaR evolution, return distribution comparison
- **Platform reliability** — latency quantile tracking, error rate distribution shifts

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to:

- Add a new primitive to this library
- Propose these functions for inclusion in **`plotly/plotly.py`** core
  (step-by-step guide, proposed function signatures, and PR template included)

---

## Publishing to Hugging Face Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space), select **Streamlit**.
2. Push this repository to the Space:

```bash
git remote add space https://huggingface.co/spaces/samibahig/plotlyml-demo
git subtree push --prefix python-package space main
```

3. Set the Space's `app_file` to `demo/app.py` in the Space settings
   (or add the HF YAML header to `demo/app.py`).

The Space will auto-install from `demo/requirements.txt`.

---

## License

MIT — see [LICENSE](LICENSE).
