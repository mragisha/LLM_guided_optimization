"""
Scott-Knott ranking and Cliff's Delta effect size.
Used to rank treatments per task and aggregate across tasks.
"""
import numpy as np
from itertools import combinations


def cliffs_delta(a, b):
    """Cliff's Delta effect size between two samples."""
    a, b = np.array(a), np.array(b)
    n = len(a) * len(b)
    more = sum(1 for x in a for y in b if x > y)
    less = sum(1 for x in a for y in b if x < y)
    return (more - less) / n


def bootstrap_test(a, b, n_boot=1000, alpha=0.05):
    """Two-sample bootstrap test. Returns True if difference is significant."""
    a, b = np.array(a), np.array(b)
    observed = abs(np.median(a) - np.median(b))
    combined = np.concatenate([a, b])
    count = 0
    for _ in range(n_boot):
        s1 = np.random.choice(combined, size=len(a), replace=True)
        s2 = np.random.choice(combined, size=len(b), replace=True)
        count += abs(np.median(s1) - np.median(s2)) >= observed
    return (count / n_boot) < alpha


def scott_knott(groups, n_boot=1000, alpha=0.05):
    """
    Scott-Knott recursive bi-clustering.
    groups: dict of {name: [values]}
    Returns dict of {name: rank} where 0 = best (lowest median Chebyshev).
    """
    names = list(groups.keys())
    medians = {n: np.median(groups[n]) for n in names}
    sorted_names = sorted(names, key=lambda n: medians[n])

    ranks = {n: 0 for n in names}
    _sk_split(sorted_names, groups, ranks, n_boot, alpha, current_rank=0)
    return ranks


def _sk_split(names, groups, ranks, n_boot, alpha, current_rank):
    if len(names) <= 1:
        for n in names:
            ranks[n] = current_rank
        return

    best_split, best_var = None, float('inf')
    for i in range(1, len(names)):
        left = names[:i]
        right = names[i:]
        var = _weighted_var(left, groups) + _weighted_var(right, groups)
        if var < best_var:
            best_var = var
            best_split = i

    left = names[:best_split]
    right = names[best_split:]

    left_vals = [v for n in left for v in groups[n]]
    right_vals = [v for n in right for v in groups[n]]

    if bootstrap_test(left_vals, right_vals, n_boot, alpha):
        _sk_split(left, groups, ranks, n_boot, alpha, current_rank)
        _sk_split(right, groups, ranks, n_boot, alpha, current_rank + 1)
    else:
        for n in names:
            ranks[n] = current_rank


def _weighted_var(names, groups):
    all_vals = [v for n in names for v in groups[n]]
    total_n = len(all_vals)
    if total_n == 0:
        return 0
    result = 0
    for n in names:
        vals = np.array(groups[n])
        result += len(vals) * np.var(vals)
    return result / total_n
