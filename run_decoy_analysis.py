"""
Decoy chunk analysis: measuring chunk-index leakage reduction.

For each dataset, simulates decoy chunk assignment at varying budgets
and measures within-bucket distinguishability (no HE needed).
"""

import json
import math
import numpy as np
from collections import defaultdict
from itertools import combinations

from src.dataset import RecDataset
from src.oblirec import ObliRec


DATASETS = {
    "ml-100k":     {"rating_threshold": 4},
    "ml-1m":       {"rating_threshold": 4},
    "gowalla":     {"rating_threshold": 0},
    "amazon-book": {"rating_threshold": 0},
}

SLOT_CAPACITY = 4096  # poly_degree=8192 → S=4096


def get_user_active_chunks(dataset, oblipack, S):
    """For each user, compute which chunk indices have non-zero indicator entries."""
    M = dataset.n_items
    C = math.ceil(M / S)
    user_chunks = {}

    for uid in range(dataset.n_users):
        items = dataset.train_dict.get(uid, [])
        if not items:
            user_chunks[uid] = set()
            continue
        # Find which chunks contain interacted items
        active = set()
        for item_id in items:
            chunk_id = item_id // S
            active.add(chunk_id)
        user_chunks[uid] = active

    return user_chunks, C


def simulate_decoys(user_chunks, bucket_users, L_b, C, n_decoys, rng):
    """Simulate decoy assignment for all users in a bucket.

    Per Definition B.1 (bucketed mode): D_u = ∅, so no pre-padding to L_b.
    Each user submits their active chunks plus n_decoys random inactive chunks.

    Returns dict: uid -> frozenset of submitted chunk indices.
    """
    submitted = {}
    all_chunks = set(range(C))

    for uid in bucket_users:
        active = user_chunks.get(uid, set())
        inactive = list(all_chunks - active)
        rng.shuffle(inactive)

        n_take = min(n_decoys, len(inactive))
        extra = set(inactive[:n_take])

        submitted[uid] = frozenset(active | extra)

    return submitted


def analyze_bucket(submitted_sets):
    """Compute distinguishability metrics for a bucket's submitted chunk sets."""
    uids = list(submitted_sets.keys())
    n = len(uids)
    if n <= 1:
        return {"n_users": n, "unique_patterns": 1, "distinguishable_frac": 0.0,
                "avg_jaccard": 0.0, "avg_chunks": 0}

    # Unique patterns: fraction of users whose chunk-index set is unique (not shared)
    patterns = [submitted_sets[uid] for uid in uids]
    counts = {}
    for p in patterns:
        counts[p] = counts.get(p, 0) + 1
    unique = len(counts)  # number of distinct pattern types
    distinguishable = sum(1 for p in patterns if counts[p] == 1) / n

    # Average Jaccard distance (sample if large)
    if n <= 50:
        pairs = list(combinations(range(n), 2))
    else:
        pairs = [(i, j) for i, j in zip(
            np.random.randint(0, n, 200), np.random.randint(0, n, 200)) if i != j]

    jaccard_dists = []
    for i, j in pairs:
        a, b = patterns[i], patterns[j]
        if len(a | b) == 0:
            jaccard_dists.append(0.0)
        else:
            jaccard_dists.append(1.0 - len(a & b) / len(a | b))

    avg_jaccard = float(np.mean(jaccard_dists)) if jaccard_dists else 0.0
    avg_chunks = float(np.mean([len(s) for s in patterns]))

    return {
        "n_users": n,
        "unique_patterns": unique,
        "distinguishable_frac": distinguishable,
        "avg_jaccard": avg_jaccard,
        "avg_chunks": avg_chunks,
    }


def main():
    results = {}

    for ds_name, ds_cfg in DATASETS.items():
        print(f"\n{'='*65}")
        print(f"  {ds_name}")
        print(f"{'='*65}")

        dataset = RecDataset(f"data/{ds_name}", dataset_name=ds_name,
                             rating_threshold=ds_cfg["rating_threshold"], seed=42)
        oblipack = ObliRec(dataset, seed=42)

        S = SLOT_CAPACITY
        user_chunks, C = get_user_active_chunks(dataset, oblipack, S)

        # Compute L_b per bucket (max active chunks in bucket)
        bucket_Lb = {}
        for bid, users in oblipack.user_buckets.items():
            max_active = 0
            for uid in users:
                max_active = max(max_active, len(user_chunks.get(uid, set())))
            bucket_Lb[bid] = max_active

        print(f"  Items M={dataset.n_items}, Slots S={S}, Chunks C={C}")
        print(f"  Buckets: {len(oblipack.user_buckets)}")
        for bid in sorted(oblipack.user_buckets.keys()):
            print(f"    B{bid}: {len(oblipack.user_buckets[bid])} users, L_b={bucket_Lb[bid]}")

        # Decoy budgets to test
        max_Lb = max(bucket_Lb.values()) if bucket_Lb else 1
        decoy_budgets = sorted(set([0, 1, 2, 3, 5, 8, max(0, C - max_Lb)]))
        # Remove negative or zero-only duplicates
        decoy_budgets = [k for k in decoy_budgets if k >= 0]

        ds_results = {"C": C, "S": S, "M": dataset.n_items, "buckets": {}}

        print(f"\n  {'k':>4} {'TotalChunks':>12} {'UniquePatFrac':>15} "
              f"{'AvgJaccard':>12} {'Cost(avg)':>10}")
        print(f"  {'-'*55}")

        for k in decoy_budgets:
            rng = np.random.RandomState(42)
            # User-weighted accumulators: weight each bucket by its user count
            # so avg_unique = "fraction of users with unique pattern" (not bucket average)
            total_unique_users = 0.0
            total_jaccard_w = 0.0
            total_cost_w = 0.0
            total_users = 0

            # Dense: k covers all inactive slots → every user submits all C chunks.
            # Per Definition B.1, dense mode is I_u = [C] for all users, not
            # active | min(k, inactive), which may fall short for low-degree users.
            is_dense = (k >= C - max_Lb)

            bucket_detail = {}
            for bid in sorted(oblipack.user_buckets.keys()):
                users = oblipack.user_buckets[bid]
                L_b = bucket_Lb[bid]

                if is_dense:
                    all_chunks_set = frozenset(range(C))
                    submitted = {uid: all_chunks_set for uid in users}
                else:
                    submitted = simulate_decoys(
                        user_chunks, users, L_b, C, k, rng)
                metrics = analyze_bucket(submitted)

                bucket_detail[str(bid)] = {
                    "L_b": L_b,
                    "total_target": L_b + k,
                    **metrics,
                }

                if metrics["n_users"] > 1:
                    n_b = metrics["n_users"]
                    total_unique_users += metrics["distinguishable_frac"] * n_b
                    total_jaccard_w += metrics["avg_jaccard"] * n_b
                    total_cost_w += metrics["avg_chunks"] * n_b
                    total_users += n_b

            avg_unique = total_unique_users / total_users if total_users > 0 else 0.0
            avg_jaccard = total_jaccard_w / total_users if total_users > 0 else 0.0
            avg_cost = total_cost_w / total_users if total_users > 0 else 0.0

            label = f"k={k}" if k < C - max_Lb else f"k={k} (dense)"
            print(f"  {label:>4} {avg_cost:>12.1f} {avg_unique:>15.3f} "
                  f"{avg_jaccard:>12.3f} {avg_cost:>10.1f}")

            ds_results[f"decoys_{k}"] = {
                "n_decoys": k,
                "avg_distinguishable_frac": avg_unique,
                "avg_jaccard_distance": avg_jaccard,
                "avg_total_chunks": avg_cost,
                "buckets": bucket_detail,
            }

        results[ds_name] = ds_results

    # Save
    with open("results_decoy_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to results_decoy_analysis.json")


if __name__ == "__main__":
    main()
