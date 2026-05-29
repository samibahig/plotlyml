# Contributing — and the path toward Plotly core

This document covers two audiences:

1. **Contributors to this repository** — feature additions, bug fixes, new primitives
2. **Anyone wanting to propose these primitives for inclusion in `plotly/plotly.py`**

---

## Contributing to this repo

```bash
git clone https://github.com/samibahig/plotlyml
cd plotlyml
pip install -e ".[dev]"
pytest tests/
```

Each primitive lives in `plotlyml/<name>.py`.  The public function must:

- Return a `plotly.graph_objects.Figure`
- Accept both `list`, `numpy.ndarray`, and `pandas.Series` inputs (handled by `_utils._to_list`)
- Include a full NumPy-style docstring with a working example
- Have no mandatory dependencies beyond `plotly`

---

## Proposing these primitives for Plotly core

Plotly maintains two Python entry points:

| Entry point | Where |
|---|---|
| `plotly.graph_objects` (low-level) | `packages/python/plotly/plotly/graph_objs/` |
| `plotly.express` (high-level) | `packages/python/plotly/plotly/express/_chart_types.py` |

The most realistic contribution path is **`plotly.express`**, since these are
high-level convenience functions that return a `go.Figure`.

### Step-by-step

#### 1. Open a GitHub Discussion first

Go to [github.com/plotly/plotly.py/discussions](https://github.com/plotly/plotly.py/discussions)
and open a **"New Feature"** discussion.  Include:

- A clear one-paragraph description of each primitive
- The proposed function signature (see below)
- 1–2 real-world use cases
- A link to this repository as a working prototype

This is the most important step — the core team decides whether a new
`px` function is in scope before code review begins.

#### 2. Proposed function signatures for plotly.express

```python
# plotly/express/_chart_types.py

def distribution_drift(
    data_frame=None,
    *,
    reference: str | Sequence[float] | None = None,
    current:   str | Sequence[float] | None = None,
    bins: int = 50,
    divergence_threshold: float = 0.1,
    reference_name: str = "Reference",
    current_name: str = "Current",
    x: str | None = None,
    color: str | None = None,
    title: str | None = None,
    template=None,
    **kwargs,
) -> go.Figure:
    ...

def model_disagreement(
    data_frame=None,
    *,
    x: str | Sequence[float] | None = None,
    y: str | Sequence[float] | None = None,
    predictions: str | Sequence[Sequence[float]] | None = None,
    threshold: float = 0.05,
    color_continuous_scale: str = "Viridis",
    title: str | None = None,
    template=None,
    **kwargs,
) -> go.Figure:
    ...

def quantile_evolution(
    data_frame=None,
    *,
    x: str | Sequence | None = None,
    p50: str | Sequence[float] | None = None,
    p10: str | Sequence[float] | None = None,
    p90: str | Sequence[float] | None = None,
    p25: str | Sequence[float] | None = None,
    p75: str | Sequence[float] | None = None,
    mean: str | Sequence[float] | None = None,
    title: str | None = None,
    template=None,
    **kwargs,
) -> go.Figure:
    ...
```

#### 3. Fork and implement

```bash
git clone https://github.com/plotly/plotly.py
cd plotly.py
pip install -e "packages/python/plotly[dev]"
```

Add the functions to `packages/python/plotly/plotly/express/_chart_types.py`
and export them in `packages/python/plotly/plotly/express/__init__.py`.

Add tests in `packages/python/plotly/tests/test_core/test_px/`.

#### 4. Open the Pull Request

Use the PR template.  Reference the GitHub Discussion.  The core team requires:

- Full docstring
- Unit tests
- An entry in `CHANGELOG.md`
- No new mandatory dependencies

---

## Prior art in Plotly worth referencing

When writing the Discussion or PR, link to these existing functions as
style guides:

- `px.histogram()` — overlapping histograms with `barmode="overlay"`
- `px.scatter()` with `color=` — continuous colourscale on a scatter
- `px.line()` with `error_y=` / fill-between pattern — closest to quantile bands

These show the team that your additions are stylistically consistent with
the existing API.
