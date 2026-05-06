"""
Multi-session composition experiment for ObliRec.

Simulates how re-identification accuracy accumulates over multiple sessions
as users acquire new interactions and potentially transition between buckets.
Tests sticky bucketing as a mitigation strategy.

Results used in Table multisession of the paper.
"""

import math
import numpy as np
from collections import defaultdict
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def assign_bucket_log2(degree, max_degree=None):
    """Assign user to log-base-2 bucket."""
    if degree <= 0:
        return 0
    return math.ceil(math.log2(max(degree, 1)))


def bucket_range(bid):
    """Return (lo, hi) for bucket bid."""
    if bid == 0:
        return (0, 1)
    return (2 ** (bid - 1) + 1, 2 ** bid)


def reid_accuracy_from_traces(user_traces, user_ids):
    """
    Compute re-identification accuracy given accumulated traces.

    user_traces: dict uid -> list of observed values (degree or bucket) across sessions
    user_ids: list of user IDs

    Metric: fraction of users with a unique trajectory fingerprint
    (a user is re-identifiable iff no other user shares their full trace vector).
    """
    trace_groups = defaultdict(int)
    for uid in user_ids:
        trace_groups[tuple(user_traces[uid])] += 1

    n = len(user_ids)
    unique_count = sum(
        1 for uid in user_ids if trace_groups[tuple(user_traces[uid])] == 1
    )
    return float(unique_count) / n


def run_multisession_experiment(
    train_dict,
    n_sessions=10,
    sticky_thresholds=(0, 1, 3, 5),
    seed=42
):
    """
    Run multi-session composition experiment.

    Models T sequential queries as a temporal-growth process: each interaction
    is assigned to a uniformly random session, so cumulative degree at session t
    grows from ~d_u/T (session 1) to d_u (session T). Users diverge in their
    growth trajectories, making multi-session traces increasingly unique.

    Sticky bucketing uses a delta-threshold: rebucket only when cumulative degree
    has grown by more than delta since the last bucket assignment. This resists
    micro-increment tracking across sessions.

    Args:
        train_dict: dict uid -> list of item IDs
        n_sessions: number of sessions (T)
        sticky_thresholds: list of delta values for sticky bucketing (0 = no sticky)
        seed: random seed

    Returns:
        dict with results per strategy per session count
    """
    rng = np.random.RandomState(seed)
    user_ids = sorted(train_dict.keys())

    # Temporal growth model: assign each interaction to a uniformly random session.
    # degree_history[uid][t] = cumulative interactions through session t (0-indexed).
    # By session T-1 the cumulative count equals d_u (the full training degree).
    degree_history = {}
    for uid in user_ids:
        d_u = len(train_dict[uid])
        session_assignments = rng.randint(0, n_sessions, size=d_u)
        counts = np.bincount(session_assignments, minlength=n_sessions)
        cumulative = np.cumsum(counts)
        degree_history[uid] = list(cumulative)

    results = {}
    checkpoints = [1, 3, 5, 10]
    checkpoints = [c for c in checkpoints if c <= n_sessions]

    # --- No ObliRec: attacker sees exact cumulative degree each session ---
    no_oblipack_traces = {uid: [] for uid in user_ids}
    no_oblipack_results = {}
    for session in range(n_sessions):
        for uid in user_ids:
            deg = degree_history[uid][session]
            no_oblipack_traces[uid].append(deg)

        if (session + 1) in checkpoints:
            acc = reid_accuracy_from_traces(no_oblipack_traces, user_ids)
            no_oblipack_results[session + 1] = acc

    results['no_oblipack'] = no_oblipack_results

    # --- ObliRec with various delta-threshold sticky bucketing ---
    for delta in sticky_thresholds:
        traces = {uid: [] for uid in user_ids}
        # For delta-threshold sticky: track degree at last bucket assignment.
        # Rebucket only when cumulative degree grows by more than delta.
        last_assigned_degree = {uid: degree_history[uid][0] for uid in user_ids}
        prev_bucket = {uid: assign_bucket_log2(degree_history[uid][0])
                       for uid in user_ids}

        strategy_results = {}

        for session in range(n_sessions):
            for uid in user_ids:
                deg = degree_history[uid][session]

                if delta == 0:
                    # No sticky: always use true bucket for current cumulative degree
                    effective_bucket = assign_bucket_log2(deg)
                    prev_bucket[uid] = effective_bucket
                else:
                    degree_increase = deg - last_assigned_degree[uid]
                    if degree_increase > delta:
                        # Degree grew enough: reassign bucket
                        effective_bucket = assign_bucket_log2(deg)
                        prev_bucket[uid] = effective_bucket
                        last_assigned_degree[uid] = deg
                    else:
                        # Below threshold: keep old bucket
                        effective_bucket = prev_bucket[uid]

                traces[uid].append(effective_bucket)

            if (session + 1) in checkpoints:
                acc = reid_accuracy_from_traces(traces, user_ids)
                strategy_results[session + 1] = acc

        label = f'oblipack_sticky_{delta}' if delta > 0 else 'oblipack_no_sticky'
        results[label] = strategy_results

    return results


DATASET_CONFIGS = {
    'gowalla':     {'path': 'data/gowalla',     'rating_threshold': 0},
    'amazon-book': {'path': 'data/amazon-book', 'rating_threshold': 0},
}


def load_train_dict(dataset_name):
    from dataset import RecDataset
    cfg = DATASET_CONFIGS[dataset_name]
    ds = RecDataset(cfg['path'], dataset_name=dataset_name,
                    rating_threshold=cfg['rating_threshold'])
    return {u: list(items) for u, items in ds.train_dict.items() if len(items) > 0}


def run_dataset(dataset_name, seed=42):
    rng = np.random.RandomState(seed)
    try:
        train_dict = load_train_dict(dataset_name)
        print(f"Loaded {dataset_name}: {len(train_dict)} users")
    except Exception as e:
        print(f"Loading {dataset_name} failed ({e}), skipping.")
        return None

    degrees = [len(v) for v in train_dict.values()]
    print(f"  Degree range: [{min(degrees)}, {max(degrees)}], mean: {np.mean(degrees):.1f}")

    results = run_multisession_experiment(
        train_dict,
        n_sessions=10,
        sticky_thresholds=(0, 1, 3, 5),
        seed=seed,
    )

    strategies = [
        ('no_oblipack',        'No ObliRec     '),
        ('oblipack_no_sticky', 'ObliRec (no st.)'),
        ('oblipack_sticky_1',  'Sticky (δ=1)   '),
        ('oblipack_sticky_3',  'Sticky (δ=3)   '),
        ('oblipack_sticky_5',  'Sticky (δ=5)   '),
    ]
    print(f"\n  {'Strategy':<20} {'T=1':>8} {'T=3':>8} {'T=5':>8} {'T=10':>8}")
    print("  " + "-" * 48)
    for key, label in strategies:
        if key in results:
            v = results[key]
            print(f"  {label:<20} "
                  f"{v.get(1, float('nan')):>8.4f} "
                  f"{v.get(3, float('nan')):>8.4f} "
                  f"{v.get(5, float('nan')):>8.4f} "
                  f"{v.get(10, float('nan')):>8.4f}")

    return {'n_users': len(train_dict), 'results': results}


def main():
    all_output = {}
    for ds_name in DATASET_CONFIGS:
        print(f"\n{'='*60}")
        print(f"  {ds_name}")
        print(f"{'='*60}")
        out = run_dataset(ds_name, seed=42)
        if out is not None:
            all_output[ds_name] = out

    output = {
        'experiment': 'multi_session_composition',
        'n_sessions': 10,
        'session_model': 'temporal_growth_uniform_assignment',
        'sticky_model': 'delta_threshold',
        'datasets': all_output,
    }
    with open('results_multisession.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to results_multisession.json")


if __name__ == '__main__':
    main()
