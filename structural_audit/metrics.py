"""
Leakage metric computations.

Each function takes a list of ExecutionTraces and returns a scalar metric.
All functions are stateless and can be composed freely.
"""

from __future__ import annotations
import math
import numpy as np
from collections import defaultdict, Counter
from typing import Sequence
from .core import ExecutionTrace, LeakageMetrics


# ------------------------------------------------------------------ #
#  Individual metric functions
# ------------------------------------------------------------------ #

def _infer_degree_from_rotation_count(n_rot: int) -> int:
    """Invert the FicGCN rotation formula f(d) = d * ceil(log2(d)) to recover d.

    f is strictly increasing for d >= 1, so d is uniquely recoverable.
    The attacker observes the rotation count and inverts it exactly.
    Returns the closest d when n_rot falls between two f-values (should
    not occur in practice since n_rot is always set to f(d) for some d).
    """
    if n_rot <= 0:
        return 0
    prev_d = 1
    for d in range(1, n_rot + 1):
        f = int(d * math.ceil(math.log2(max(d, 2))))
        if f == n_rot:
            return d
        if f > n_rot:
            return prev_d
        prev_d = d
    return prev_d


def degree_mae(traces: Sequence[ExecutionTrace]) -> float:
    """Mean absolute error between true degree and attacker's best estimate.

    n_rotations stores the actual HE rotation count: f(d) = d * ceil(log2(d))
    in the unprotected sparse path, and f(B) for bucket target B otherwise.
    The attacker inverts f (which is bijective) to recover the degree exactly
    in sparse mode (MAE = 0) or the bucket target in protected modes (MAE > 0).
    """
    errors = []
    for t in traces:
        estimated = _infer_degree_from_rotation_count(t.n_rotations)
        errors.append(abs(estimated - t.true_degree))
    return float(np.mean(errors)) if errors else 0.0


def reid_accuracy(traces: Sequence[ExecutionTrace]) -> tuple[float, float, int]:
    """Re-identification accuracy and anonymity set statistics.

    Groups users by their observable fingerprint (chunk_indices, n_rotations).
    For each user, re-ID probability = 1 / |group|.

    Returns:
        (mean_reid_acc, mean_anon_set, min_anon_set)
    """
    fingerprint_groups: dict[tuple, list[int]] = defaultdict(list)
    for t in traces:
        # Fingerprint: what the server observes
        fp = (t.n_rotations, t.n_chunks, t.chunk_indices)
        fingerprint_groups[fp].append(t.user_id)

    reid_acc_per_user = []
    anon_sizes = []
    for fp, users in fingerprint_groups.items():
        group_size = len(users)
        for _ in users:
            reid_acc_per_user.append(1.0 / group_size)
            anon_sizes.append(group_size)

    if not reid_acc_per_user:
        return 0.0, 0.0, 0

    return (
        float(np.mean(reid_acc_per_user)),
        float(np.mean(anon_sizes)),
        int(np.min(anon_sizes)),
    )


def timing_cv(traces: Sequence[ExecutionTrace],
              bucket_fn=None) -> float:
    """Mean within-bucket coefficient of variation of wall-clock time.

    If bucket_fn is provided, groups by bucket assignment.
    Otherwise groups by n_rotations (unprotected regime).

    CV = std / mean for each group; returns average over groups.
    """
    if all(t.wall_time_ms == 0.0 for t in traces):
        return float("nan")

    group_fn = bucket_fn if bucket_fn is not None else (lambda t: t.n_rotations)
    groups: dict = defaultdict(list)
    for t in traces:
        groups[group_fn(t)].append(t.wall_time_ms)

    cvs = []
    for key, times in groups.items():
        if len(times) < 2:
            continue
        times_arr = np.array(times)
        mu = times_arr.mean()
        if mu > 0:
            cvs.append(times_arr.std() / mu)
    return float(np.mean(cvs)) if cvs else float("nan")


def chunk_pattern_unique_pct(traces: Sequence[ExecutionTrace],
                              bucket_fn=None) -> float:
    """Fraction of users with a unique chunk-index pattern within their bucket.

    If bucket_fn is provided, evaluates uniqueness within each bucket.
    Otherwise treats all users as one group (unprotected regime).
    """
    if bucket_fn is None:
        counts = Counter(t.chunk_indices for t in traces)
        n_unique = sum(1 for t in traces if counts[t.chunk_indices] == 1)
        return n_unique / len(traces) if traces else 0.0

    buckets: dict = defaultdict(list)
    for t in traces:
        buckets[bucket_fn(t)].append(t.chunk_indices)

    total_unique = 0
    total = 0
    for bucket_id, patterns in buckets.items():
        counts = Counter(patterns)
        total_unique += sum(1 for p in patterns if counts[p] == 1)
        total += len(patterns)
    return total_unique / total if total > 0 else 0.0


def popularity_prior_f1(train_dict: dict, K: int = 20) -> float:
    """Interaction-recovery F1 using a popularity-prior attacker.

    Recommends the top-K globally most popular items to every user.
    This is a dataset-level floor independent of the HE execution regime.
    Returns mean F1@K across all users with at least one training item.
    """
    item_counts: Counter = Counter()
    for items in train_dict.values():
        item_counts.update(items)

    top_k = set(item for item, _ in item_counts.most_common(K))

    f1s = []
    for items in train_dict.values():
        actual = set(items)
        if not actual:
            continue
        tp = len(top_k & actual)
        if tp == 0:
            f1s.append(0.0)
            continue
        precision = tp / K
        recall = tp / len(actual)
        f1s.append(2 * precision * recall / (precision + recall))

    return float(np.mean(f1s)) if f1s else 0.0


# ------------------------------------------------------------------ #
#  Composite metric computation
# ------------------------------------------------------------------ #

def compute_leakage_metrics(
    traces: Sequence[ExecutionTrace],
    bucket_fn=None,
    baseline_mean_time_ms: float = None,
) -> LeakageMetrics:
    """Compute all leakage metrics from a list of traces.

    Args:
        traces:                 list of ExecutionTrace for each user
        bucket_fn:              function(ExecutionTrace) -> bucket_id,
                                used for within-bucket metrics.
                                Pass None for unprotected (sparse) regime.
        baseline_mean_time_ms:  mean latency of unprotected sparse baseline,
                                used to compute overhead ratio.
    """
    mae = degree_mae(traces)
    mean_reid, mean_anon, min_anon = reid_accuracy(traces)
    cv = timing_cv(traces, bucket_fn=bucket_fn)
    chunk_unique = chunk_pattern_unique_pct(traces, bucket_fn=bucket_fn)

    overhead = 1.0
    if baseline_mean_time_ms and baseline_mean_time_ms > 0:
        mean_time = np.mean([t.wall_time_ms for t in traces if t.wall_time_ms > 0])
        if mean_time > 0:
            overhead = mean_time / baseline_mean_time_ms

    return LeakageMetrics(
        degree_mae=mae,
        reid_accuracy=mean_reid,
        mean_anonymity_set=mean_anon,
        min_anonymity_set=min_anon,
        timing_cv=float("nan") if math.isnan(cv) else cv,
        chunk_pattern_unique_pct=chunk_unique,
        overhead=overhead,
    )
