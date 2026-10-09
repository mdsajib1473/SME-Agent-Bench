"""Statistics: task bootstrap, pass^k, majority vote, exact McNemar, Holm, Cohen's kappa."""

import math

import numpy as np
from scipy.stats import binomtest

N_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0


def bootstrap_ci(task_values, n_resamples=N_RESAMPLES, seed=BOOTSTRAP_SEED, level=0.95):
    """Percentile interval of the mean, resampling tasks with replacement."""
    values = np.asarray([v for v in task_values if v == v], dtype=float)
    if len(values) == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_resamples, len(values)))].mean(axis=1)
    tail = (1.0 - level) / 2.0 * 100.0
    low, high = np.percentile(means, [tail, 100.0 - tail])
    return float(low), float(high)


def pass_hat_k(per_task_successes, k):
    """Fraction of tasks solved in all of their first k seeds, over tasks with at least k seeds.

    Returns (value, number of tasks used).
    """
    eligible = [list(s)[:k] for s in per_task_successes if len(s) >= k]
    if not eligible or k < 1:
        return math.nan, 0
    return sum(all(s) for s in eligible) / len(eligible), len(eligible)


def majority_success(successes):
    """True when more than half of the seeds succeeded; a tie counts as failure."""
    successes = list(successes)
    return sum(bool(s) for s in successes) * 2 > len(successes)


def discordant_counts(a, b):
    """Paired booleans to (both, a only, b only, neither)."""
    both = sum(x and y for x, y in zip(a, b))
    a_only = sum(x and not y for x, y in zip(a, b))
    b_only = sum(y and not x for x, y in zip(a, b))
    neither = sum(not x and not y for x, y in zip(a, b))
    return both, a_only, b_only, neither


def mcnemar_exact(a_only, b_only):
    """Two sided exact McNemar p value: binomial test of the discordant pairs at 0.5."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    return float(binomtest(a_only, n, 0.5).pvalue)


def discordant_odds_ratio(a_only, b_only):
    if b_only == 0:
        return math.inf if a_only > 0 else math.nan
    return a_only / b_only


def holm(pvalues):
    """Holm step down adjusted p values; NaN entries are skipped and stay NaN."""
    pvalues = list(pvalues)
    valid = [i for i, p in enumerate(pvalues) if p == p]
    order = sorted(valid, key=lambda i: pvalues[i])
    adjusted = [math.nan] * len(pvalues)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(valid) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def cohen_kappa(first, second):
    """Cohen's kappa for two equally long label lists."""
    first, second = list(first), list(second)
    if len(first) != len(second):
        raise ValueError("label lists differ in length")
    n = len(first)
    if n == 0:
        return math.nan
    observed = sum(x == y for x, y in zip(first, second)) / n
    labels = set(first) | set(second)
    expected = sum((first.count(label) / n) * (second.count(label) / n) for label in labels)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else math.nan
    return (observed - expected) / (1.0 - expected)
