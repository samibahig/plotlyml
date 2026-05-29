"""
model_disagreement
==================
Visualise per-sample prediction disagreement across a model ensemble.

Typical use-cases
-----------------
- Active-learning candidate selection (high disagreement = most informative)
- Identifying distribution-shift hot-spots in a reduced feature space
- Audit of ensemble calibration quality
"""
from __future__ import annotations

from typing import Optional, Sequence

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ._utils import _to_list, compute_variance


def model_disagreement(
    x: Sequence[float],
    y: Sequence[float],
    predictions: Sequence[Sequence[float]],
    *,
    sample_ids: Optional[Sequence[str]] = None,
    true_labels: Optional[Sequence[float]] = None,
    threshold: float = 0.05,
    x_label: str = "Dim 1",
    y_label: str = "Dim 2",
    colorscale: str = "Viridis",
    title: Optional[str] = None,
    template: str = "plotly_dark",
) -> go.Figure:
    """
    Create a Model Disagreement Map.

    Each sample is plotted in a 2-D reduced feature space (UMAP, t-SNE, PCA, …)
    and coloured by its ensemble variance — the disagreement across model
    predictions.  Samples above *threshold* are fully opaque; others are
    dimmed, focusing attention on the uncertain region.

    Parameters
    ----------
    x : sequence of float
        X-coordinates in the reduced feature space (one value per sample).
    y : sequence of float
        Y-coordinates in the reduced feature space (one value per sample).
    predictions : sequence of sequences of float
        For each sample, a list of scalar predictions from each ensemble
        member (e.g. predicted probabilities from 5 models).
    sample_ids : sequence of str, optional
        Human-readable identifiers shown in hover tooltips.
    true_labels : sequence of float, optional
        Ground-truth labels; included in hover text when provided.
    threshold : float, default 0.05
        Variance level above which a sample is considered high-disagreement.
    x_label : str, default "Dim 1"
        X-axis title (e.g. "UMAP 1", "PC 1").
    y_label : str, default "Dim 2"
        Y-axis title.
    colorscale : str, default "Viridis"
        Plotly colorscale name applied to the disagreement scores.
    title : str, optional
        Figure title.
    template : str, default "plotly_dark"
        Plotly template name.

    Returns
    -------
    plotly.graph_objects.Figure

    Examples
    --------
    >>> import numpy as np
    >>> from plotlyml import model_disagreement
    >>> rng = np.random.default_rng(42)
    >>> n = 200
    >>> x = rng.normal(size=n)
    >>> y = rng.normal(size=n)
    >>> preds = rng.random((n, 5))          # 5 model predictions per sample
    >>> fig = model_disagreement(x, y, preds, threshold=0.04)
    >>> fig.show()
    """
    xs = _to_list(x)
    ys = _to_list(y)
    preds_list = [_to_list(p) for p in predictions]
    n = len(xs)

    ids = list(sample_ids) if sample_ids is not None else [f"S-{i:04d}" for i in range(n)]
    labels = _to_list(true_labels) if true_labels is not None else [None] * n

    variances = [compute_variance(preds_list[i]) for i in range(n)]
    mean_var = sum(variances) / len(variances)
    pct_above = 100 * sum(1 for v in variances if v >= threshold) / n

    hover_texts = []
    for i in range(n):
        pstr = ", ".join(f"{p:.3f}" for p in preds_list[i])
        label_line = f"<br>True label: {labels[i]}" if labels[i] is not None else ""
        hover_texts.append(
            f"<b>{ids[i]}</b><br>"
            f"Disagreement: {variances[i]:.5f}<br>"
            f"Predictions: [{pstr}]"
            f"{label_line}"
        )

    colors = [v if v >= threshold else 0.0 for v in variances]
    opacities = [1.0 if v >= threshold else 0.15 for v in variances]

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.72, 0.28],
        subplot_titles=("Feature Space — Ensemble Disagreement", "Disagreement Distribution"),
    )

    fig.add_trace(go.Scatter(
        x=xs,
        y=ys,
        mode="markers",
        text=hover_texts,
        hoverinfo="text",
        marker=dict(
            color=colors,
            colorscale=colorscale,
            showscale=True,
            size=7,
            opacity=opacities,
            colorbar=dict(title="Variance", x=0.68, thickness=12),
        ),
        name="Samples",
        showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Histogram(
        x=variances,
        name="Variance distribution",
        marker_color="#818cf8",
        showlegend=False,
    ), row=1, col=2)

    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text=f"threshold={threshold}",
        annotation_position="top right",
        row=1, col=2,
    )

    fig.update_layout(
        title=title or "Model Disagreement Map",
        template=template,
        xaxis_title=x_label,
        yaxis_title=y_label,
        annotations=[
            *fig.layout.annotations,
            dict(
                x=1, y=1,
                xref="paper", yref="paper",
                xanchor="right", yanchor="top",
                text=(
                    f"Mean variance: <b>{mean_var:.5f}</b><br>"
                    f"Above threshold: <b>{pct_above:.1f}%</b><br>"
                    f"Samples: {n}"
                ),
                showarrow=False,
                bgcolor="rgba(0,0,0,0.4)",
                bordercolor="#818cf8",
                borderwidth=1,
                font=dict(size=11),
            ),
        ],
    )

    return fig
