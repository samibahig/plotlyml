import math
import random
from datetime import datetime, timedelta


def _gaussian(mean=0.0, std=1.0):
    u, v = 0.0, 0.0
    while u == 0:
        u = random.random()
    while v == 0:
        v = random.random()
    return math.sqrt(-2.0 * math.log(u)) * math.cos(2 * math.pi * v) * std + mean


def generate_samples(mean, std, n=800):
    return [_gaussian(mean, std) for _ in range(n)]


def generate_disagreement_data(n=300):
    xs, ys, preds_list = [], [], []

    for _ in range(n):
        cluster = 1 if random.random() > 0.5 else -1

        x = _gaussian(cluster * 2, 1.5)
        y = _gaussian(cluster * 2, 1.5)

        dist = math.sqrt(x * x + y * y)
        unc = max(0.0, 1 - dist / 3)

        true_label = 1 if cluster > 0 else 0

        preds = [
            max(0, min(1, true_label + _gaussian(0, unc * 0.5 + 0.1))) for _ in range(5)
        ]

        xs.append(x)
        ys.append(y)
        preds_list.append(preds)

    return xs, ys, preds_list


def generate_quantile_data(weeks=52):
    timestamps, p10s, p25s, p50s, p75s, p90s, means = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    current_mean = 100.0
    current_var = 10.0

    start = datetime.now() - timedelta(weeks=weeks)

    for w in range(weeks):
        current_mean += _gaussian(0.5, 1)

        if 20 < w < 30:
            current_var += _gaussian(2, 0.5)
        elif w >= 30:
            current_var = max(10.0, current_var - 1)

        samples = sorted([_gaussian(current_mean, current_var) for _ in range(100)])

        timestamps.append((start + timedelta(weeks=w)).strftime("%Y-%m-%d"))

        p10s.append(samples[10])
        p25s.append(samples[25])
        p50s.append(samples[50])
        p75s.append(samples[75])
        p90s.append(samples[90])
        means.append(sum(samples) / len(samples))

    return timestamps, p10s, p25s, p50s, p75s, p90s, means
