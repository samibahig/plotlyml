from __future__ import annotations

import math
import random
from typing import Sequence


def _to_list(x) -> list:
    """Accept list, tuple, numpy array, or pandas Series."""
    try:
        return x.tolist()
    except AttributeError:
        return list(x)


def _compute_histogram(
    data: list[float], bins: int, global_min: float, global_max: float
) -> list[float]:
    """Return normalised probability per bin (sums to 1)."""
    bin_width = (global_max - global_min) / bins
    counts = [0] * bins
    for v in data:
        idx = min(int((v - global_min) / bin_width), bins - 1)
        if 0 <= idx < bins:
            counts[idx] += 1
    total = len(data)
    return [c / total for c in counts]


def kl_divergence(p: list[float], q: list[float]) -> float:
    """Approximate KL divergence D_KL(P || Q) from histogram bins."""
    eps = 1e-10
    return sum(
        (pi + eps) * math.log((pi + eps) / (qi + eps))
        for pi, qi in zip(p, q)
    )


def compute_drift_stats(
    reference: list[float],
    current: list[float],
    bins: int = 50,
) -> dict:
    global_min = min(min(reference), min(current))
    global_max = max(max(reference), max(current))

    ref_hist = _compute_histogram(reference, bins, global_min, global_max)
    cur_hist = _compute_histogram(current, bins, global_min, global_max)

    def _mean(d):
        return sum(d) / len(d)

    def _std(d):
        m = _mean(d)
        return math.sqrt(sum((x - m) ** 2 for x in d) / len(d))

    return {
        "ref_mean": _mean(reference),
        "ref_std": _std(reference),
        "cur_mean": _mean(current),
        "cur_std": _std(current),
        "kl_divergence": kl_divergence(ref_hist, cur_hist),
    }


def compute_variance(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)
