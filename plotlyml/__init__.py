"""
plotlyml — ML Monitoring Visualization Primitives
==================================================

Three high-level Plotly functions for modern ML observability workflows,
designed in the spirit of ``plotly.express``.

Quick start
-----------
>>> from plotlyml import distribution_drift, model_disagreement, quantile_evolution

Each function returns a ``plotly.graph_objects.Figure`` that you can
``.show()``, embed in a Dash layout, or export with ``.write_html()``.

Functions
---------
distribution_drift      Compare two distributions and surface KL divergence.
model_disagreement      Map per-sample ensemble variance in a feature space.
quantile_evolution      Track P10/P50/P90 (and more) over time or cohorts.
"""

from .distribution_drift import distribution_drift
from .model_disagreement import model_disagreement
from .quantile_evolution import quantile_evolution

__all__ = [
    "distribution_drift",
    "model_disagreement",
    "quantile_evolution",
]

__version__ = "0.1.0"
