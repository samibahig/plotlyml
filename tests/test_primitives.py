"""Smoke tests — verify each primitive returns a valid go.Figure."""
import math
import random

import plotly.graph_objects as go
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from plotlyml import distribution_drift, model_disagreement, quantile_evolution


def _gaussian(mean=0.0, std=1.0):
    u, v = 0.0, 0.0
    while u == 0: u = random.random()
    while v == 0: v = random.random()
    return math.sqrt(-2.0 * math.log(u)) * math.cos(2 * math.pi * v) * std + mean


def _samples(mean=0.0, std=1.0, n=200):
    return [_gaussian(mean, std) for _ in range(n)]


# ── distribution_drift ────────────────────────────────────────────────────────

def test_distribution_drift_returns_figure():
    fig = distribution_drift(_samples(0, 1), _samples(0.8, 1.2))
    assert isinstance(fig, go.Figure)

def test_distribution_drift_has_two_traces():
    fig = distribution_drift(_samples(), _samples(1, 1.5))
    assert len(fig.data) == 2

def test_distribution_drift_no_drift_color():
    """When KL < threshold the current trace should NOT be red."""
    fig = distribution_drift(_samples(0, 1), _samples(0, 1), divergence_threshold=10.0)
    assert fig.data[1].marker.color != "#ef4444"

def test_distribution_drift_drift_color():
    """When KL > threshold the current trace should be red."""
    fig = distribution_drift(_samples(0, 1), _samples(5, 3), divergence_threshold=0.001)
    assert fig.data[1].marker.color == "#ef4444"

def test_distribution_drift_custom_names():
    fig = distribution_drift(_samples(), _samples(1), reference_name="A", current_name="B")
    names = [t.name for t in fig.data]
    assert "A" in names and "B" in names


# ── model_disagreement ────────────────────────────────────────────────────────

def _make_disagreement_data(n=50):
    x = [_gaussian() for _ in range(n)]
    y = [_gaussian() for _ in range(n)]
    preds = [[random.random() for _ in range(5)] for _ in range(n)]
    return x, y, preds

def test_model_disagreement_returns_figure():
    x, y, p = _make_disagreement_data()
    fig = model_disagreement(x, y, p)
    assert isinstance(fig, go.Figure)

def test_model_disagreement_has_scatter_and_histogram():
    x, y, p = _make_disagreement_data()
    fig = model_disagreement(x, y, p)
    trace_types = {type(t).__name__ for t in fig.data}
    assert "Scatter" in trace_types
    assert "Histogram" in trace_types

def test_model_disagreement_custom_ids():
    n = 20
    x, y, p = _make_disagreement_data(n)
    ids = [f"item_{i}" for i in range(n)]
    fig = model_disagreement(x, y, p, sample_ids=ids)
    assert isinstance(fig, go.Figure)


# ── quantile_evolution ────────────────────────────────────────────────────────

def _make_quantile_data(n=20):
    ts = [f"2024-W{i:02d}" for i in range(n)]
    base = [100.0 + i * 0.5 for i in range(n)]
    spread = [5.0] * n
    return ts, base, spread

def test_quantile_evolution_returns_figure():
    ts, base, spread = _make_quantile_data()
    fig = quantile_evolution(ts, p50=base)
    assert isinstance(fig, go.Figure)

def test_quantile_evolution_with_all_bands():
    ts, base, spread = _make_quantile_data()
    fig = quantile_evolution(
        ts, p50=base,
        p10=[b - 2 * s for b, s in zip(base, spread)],
        p90=[b + 2 * s for b, s in zip(base, spread)],
        p25=[b - s for b, s in zip(base, spread)],
        p75=[b + s for b, s in zip(base, spread)],
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 5  # p10, p90 (2), p25, p75 (2), p50 (1)

def test_quantile_evolution_show_mean():
    ts, base, spread = _make_quantile_data()
    fig = quantile_evolution(ts, p50=base, mean=base, show_mean=True)
    names = [t.name for t in fig.data]
    assert "Mean" in names

def test_quantile_evolution_p50_only():
    """Should work with only p50 — all other bands are optional."""
    ts = ["2024-01", "2024-02", "2024-03"]
    fig = quantile_evolution(ts, p50=[100, 101, 99])
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
