"""
distribution_drift
==================
Compare two distributions and surface divergence visually.

Typical use-cases
-----------------
- Training data vs. live inference feature drift
- A/B test response distributions
- Pre/post deployment output shift detection
"""
from __future__ import annotations

from typing import Optional, Sequence

import plotly.graph_objects as go

from ._utils import _to_list, compute_drift_stats


def distribution_drift(
    reference: Sequence[float],
    current: Sequence[float],
    *,
    bins: int = 50,
    divergence_threshold: float = 0.1,
    reference_name: str = "Reference",
    current_name: str = "Current",
    x_label: str = "Feature Value",
    title: Optional[str] = None,
    template: str = "plotly_dark",
) -> go.Figure:
    """
    Create a Distribution Drift Plot comparing two samples.

    The chart overlays two normalised histograms and annotates a KL-divergence
    score.  When the score exceeds *divergence_threshold* the current
    distribution is coloured red to signal an alert state.

    Parameters
    ----------
    reference : sequence of float
        Baseline / training-time samples (e.g. feature values at training).
    current : sequence of float
        Live / inference-time samples to compare against the baseline.
    bins : int, default 50
        Number of histogram bins used for both distributions.
    divergence_threshold : float, default 0.1
        KL-divergence value above which the current distribution is flagged.
    reference_name : str, default "Reference"
        Legend label for the reference distribution.
    current_name : str, default "Current"
        Legend label for the current distribution.
    x_label : str, default "Feature Value"
        X-axis title.
    title : str, optional
        Figure title.  Defaults to "Distribution Drift — <current_name>".
    template : str, default "plotly_dark"
        Plotly template name.

    Returns
    -------
    plotly.graph_objects.Figure
        A Plotly figure ready for ``fig.show()`` or embedding.

    Examples
    --------
    >>> import numpy as np
    >>> from plotlyml import distribution_drift
    >>> ref = np.random.normal(0, 1, 1000)
    >>> cur = np.random.normal(0.8, 1.2, 1000)
    >>> fig = distribution_drift(ref, cur, divergence_threshold=0.1)
    >>> fig.show()
    """
    ref = _to_list(reference)
    cur = _to_list(current)

    stats = compute_drift_stats(ref, cur, bins=bins)
    is_drifting = stats["kl_divergence"] > divergence_threshold
    current_color = "#ef4444" if is_drifting else "#f97316"   # red : orange
    reference_color = "#3b82f6"                                 # blue

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=ref,
        name=reference_name,
        histnorm="probability density",
        opacity=0.75,
        marker_color=reference_color,
        nbinsx=bins,
    ))

    fig.add_trace(go.Histogram(
        x=cur,
        name=current_name,
        histnorm="probability density",
        opacity=0.75,
        marker_color=current_color,
        nbinsx=bins,
    ))

    kl = stats["kl_divergence"]
    alert_symbol = " ⚠" if is_drifting else ""
    annotation_text = (
        f"KL Divergence: <b>{kl:.4f}</b>{alert_symbol}<br>"
        f"{reference_name} μ={stats['ref_mean']:.3f}  σ={stats['ref_std']:.3f}<br>"
        f"{current_name} μ={stats['cur_mean']:.3f}  σ={stats['cur_std']:.3f}"
    )

    fig.update_layout(
        title=title or f"Distribution Drift — {current_name}",
        xaxis_title=x_label,
        yaxis_title="Density",
        barmode="overlay",
        template=template,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        annotations=[dict(
            x=1, y=1,
            xref="paper", yref="paper",
            xanchor="right", yanchor="top",
            text=annotation_text,
            showarrow=False,
            align="right",
            bgcolor="rgba(0,0,0,0.4)",
            bordercolor=current_color,
            borderwidth=1,
            font=dict(size=12),
        )],
    )

    return fig
