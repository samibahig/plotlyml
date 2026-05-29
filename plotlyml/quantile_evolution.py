"""
quantile_evolution
==================
Track how the full distribution of a metric evolves over time or cohorts.

Typical use-cases
-----------------
- Service latency monitoring: detect when P99 spikes while P50 stays flat
- ML prediction confidence over deployment lifetime
- Clinical trial endpoint distributions across cohorts
- Gradual data drift visible in quantile spread before it appears in the mean
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import plotly.graph_objects as go

from ._utils import _to_list


def quantile_evolution(
    timestamps: Sequence[Union[str, float]],
    p50: Sequence[float],
    *,
    p10: Optional[Sequence[float]] = None,
    p90: Optional[Sequence[float]] = None,
    p25: Optional[Sequence[float]] = None,
    p75: Optional[Sequence[float]] = None,
    mean: Optional[Sequence[float]] = None,
    show_mean: bool = False,
    volatility_multiplier: float = 1.5,
    x_label: str = "Time",
    y_label: str = "Metric Value",
    title: Optional[str] = None,
    template: str = "plotly_dark",
) -> go.Figure:
    """
    Create a Quantile Evolution Plot.

    Renders stacked, filled ribbon bands for P10–P90 and P25–P75 with P50
    as a bold centre line.  Automatically annotates timestamps where the
    P10–P90 spread is *volatility_multiplier* times wider than the series
    average, flagging high-variance windows.

    Parameters
    ----------
    timestamps : sequence of str or float
        X-axis values — ISO-8601 strings, cohort labels, or numeric indices.
    p50 : sequence of float
        Median (50th percentile) at each timestamp.  Required.
    p10 : sequence of float, optional
        10th percentile.  Outer lower ribbon edge.
    p90 : sequence of float, optional
        90th percentile.  Outer upper ribbon edge.
    p25 : sequence of float, optional
        25th percentile.  Inner lower ribbon edge (IQR lower bound).
    p75 : sequence of float, optional
        75th percentile.  Inner upper ribbon edge (IQR upper bound).
    mean : sequence of float, optional
        Arithmetic mean.  Shown as a dashed line when *show_mean* is True.
    show_mean : bool, default False
        Overlay the mean line on the chart.
    volatility_multiplier : float, default 1.5
        Spread ratio above which a timestamp is annotated as high-volatility.
        Only applied when both p10 and p90 are provided.
    x_label : str, default "Time"
        X-axis title.
    y_label : str, default "Metric Value"
        Y-axis title.
    title : str, optional
        Figure title.
    template : str, default "plotly_dark"
        Plotly template name.

    Returns
    -------
    plotly.graph_objects.Figure

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from plotlyml import quantile_evolution
    >>> rng = np.random.default_rng(0)
    >>> dates = pd.date_range("2024-01-01", periods=52, freq="W").astype(str).tolist()
    >>> base = np.cumsum(rng.normal(0.5, 1, 52)) + 100
    >>> spread = np.abs(rng.normal(5, 2, 52))
    >>> fig = quantile_evolution(
    ...     dates,
    ...     p50=base,
    ...     p10=base - 2 * spread,
    ...     p90=base + 2 * spread,
    ...     p25=base - spread,
    ...     p75=base + spread,
    ...     show_mean=True,
    ...     mean=base + rng.normal(0, 0.5, 52),
    ... )
    >>> fig.show()
    """
    ts = _to_list(timestamps)
    med = _to_list(p50)
    traces: list[go.BaseTraceType] = []

    outer_color = "rgba(59, 130, 246, 0.15)"
    inner_color = "rgba(59, 130, 246, 0.30)"
    median_color = "#3b82f6"
    mean_color = "#f97316"

    if p10 is not None and p90 is not None:
        p10l = _to_list(p10)
        p90l = _to_list(p90)

        traces.append(go.Scatter(
            x=ts, y=p10l,
            mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        traces.append(go.Scatter(
            x=ts, y=p90l,
            mode="lines",
            fill="tonexty",
            fillcolor=outer_color,
            line=dict(width=0),
            name="P10–P90 Band",
        ))

    if p25 is not None and p75 is not None:
        p25l = _to_list(p25)
        p75l = _to_list(p75)

        traces.append(go.Scatter(
            x=ts, y=p25l,
            mode="lines", line=dict(width=0),
            showlegend=False, hoverinfo="skip",
        ))
        traces.append(go.Scatter(
            x=ts, y=p75l,
            mode="lines",
            fill="tonexty",
            fillcolor=inner_color,
            line=dict(width=0),
            name="P25–P75 Band",
        ))

    traces.append(go.Scatter(
        x=ts, y=med,
        mode="lines",
        line=dict(color=median_color, width=3),
        name="Median (P50)",
    ))

    if show_mean and mean is not None:
        traces.append(go.Scatter(
            x=ts, y=_to_list(mean),
            mode="lines",
            line=dict(color=mean_color, width=2, dash="dash"),
            name="Mean",
        ))

    annotations: list[dict] = []
    if p10 is not None and p90 is not None:
        spreads = [p90l[i] - p10l[i] for i in range(len(ts))]
        avg_spread = sum(spreads) / len(spreads)
        for i, s in enumerate(spreads):
            if s > avg_spread * volatility_multiplier:
                annotations.append(dict(
                    x=ts[i], y=p90l[i],
                    text="High Volatility",
                    showarrow=True,
                    arrowhead=2,
                    ax=0, ay=-36,
                    font=dict(color="#ef4444", size=10),
                    arrowcolor="#ef4444",
                ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title or "Quantile Evolution",
        xaxis_title=x_label,
        yaxis_title=y_label,
        template=template,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        annotations=annotations if annotations else None,
    )

    return fig
