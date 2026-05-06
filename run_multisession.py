"""
Multi-session composition experiment for ObliPack.

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

    Attacker strategy: group users by their full trace vector,
    then re-ID accuracy = 1 / |group| for each user.
    """
    trace_groups = defaultdict(list)
    for uid in user_ids:
        trace_key = tuple(user_traces[uid])
        trace_groups[trace_key].append(uid)

    reid_accs = []
    for uid in user_ids:
        trace_key = tuple(user_traces[uid])
        group_size = len(trace_groups[trace_key])
        reid_accs.append(1.0 / group_size)

    return float(np.mean(reid_accs))


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

    # --- No ObliPack: attacker sees exact cumulative degree each session ---
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

    # --- ObliPack with various delta-threshold sticky bucketing ---
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


def main():
    # Load Gowalla dataset
    rng = np.random.RandomState(42)
    try:
        from dataset import RecDataset
        dataset = RecDataset('data/gowalla', 'gowalla')
        train_dict = {u: list(items) for u, items in dataset.train_dict.items()
                      if len(items) > 0}
        print(f"Loaded real Gowalla data: {len(train_dict)} users")
    except Exception as e:
        print(f"Loading real data failed ({e}), using simulated power-law distribution")
        n_users = 2000
        # Power-law with alpha ~2.17 (Gowalla-like)
        raw = rng.pareto(1.17, n_users) + 1
        degrees = np.clip(raw * 5, 1, 500).astype(int)
        train_dict = {i: list(range(int(d))) for i, d in enumerate(degrees)}

    # Sample 2000 users for experiment
    all_uids = sorted(train_dict.keys())
    if len(all_uids) > 2000:
        sampled = rng.choice(all_uids, 2000, replace=False)
        train_dict = {u: train_dict[u] for u in sampled}

    degrees = [len(v) for v in train_dict.values()]
    print(f"Running multi-session experiment with {len(train_dict)} users")
    print(f"Degree range: [{min(degrees)}, {max(degrees)}]")
    print(f"Mean degree: {np.mean(degrees):.1f}")
    print()

    results = run_multisession_experiment(
        train_dict,
        n_sessions=10,
        sticky_thresholds=(0, 1, 3, 5),
        seed=42
    )

    # Print results table (matching paper format)
    print("=" * 70)
    print("Multi-Session Re-Identification Accuracy")
    print("=" * 70)
    print(f"{'Strategy':<25} {'1 sess':>10} {'3 sess':>10} {'5 sess':>10} {'10 sess':>10}")
    print("-" * 70)

    strategies = [
        ('no_oblipack', 'No ObliPack'),
        ('oblipack_no_sticky', 'ObliPack (no sticky)'),
        ('oblipack_sticky_1', 'Sticky (δ=1)'),
        ('oblipack_sticky_3', 'Sticky (δ=3)'),
        ('oblipack_sticky_5', 'Sticky (δ=5)'),
    ]

    for key, label in strategies:
        if key in results:
            vals = results[key]
            row = f"{label:<25}"
            for s in [1, 3, 5, 10]:
                if s in vals:
                    row += f" {vals[s]:>10.3f}"
                else:
                    row += f" {'--':>10}"
            print(row)

    print("=" * 70)

    # Save results
    output = {
        'experiment': 'multi_session_composition',
        'n_users': len(train_dict),
        'n_sessions': 10,
        'session_model': 'temporal_growth_uniform_assignment',
        'sticky_model': 'delta_threshold',
        'results': results
    }

    with open('results_multisession.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to results_multisession.json")


if __name__ == '__main__':
    main()
