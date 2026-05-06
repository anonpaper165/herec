"""
Reproduce Table 3: Structural Leakage Under Four Deployment Regimes.

Runs the structural leakage audit on all training-set users and prints
leakage metrics across four ObliRec deployment modes:
  sparse   - unprotected FicGCN-style execution
  bucketed - degree bucketing only (channel 1 defense)
  decoy    - bucketing + k decoy chunk indices (partial channel 2)
  dense    - bucketing + all C chunks submitted (full channel 2 defense)

Five metrics are computed per regime:
  Degree MAE          - |attacker_estimate - true_degree|; 0 = perfect leak
  Re-ID accuracy      - mean 1/|fingerprint_group|; lower = safer
  Mean anonymity set  - average group size; larger = safer
  Min anonymity set   - smallest group (worst-case user)
  Chunk unique %      - fraction of users with unique chunk pattern in bucket

Usage:
    python audit/run_structural_audit.py --dataset gowalla
    python audit/run_structural_audit.py --dataset amazon-book
    python audit/run_structural_audit.py --dataset ml-100k
    python audit/run_structural_audit.py --dataset ml-1m
    python audit/run_structural_audit.py --dataset all

Prerequisites:
    python train.py --dataset <name>   # train LightGCN first
"""

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import RecDataset
from src.oblirec import ObliRec
from structural_audit.core import ExecutionTrace
from structural_audit.metrics import compute_leakage_metrics, popularity_prior_f1

DATASETS = {
    "ml-100k":     {"rating_threshold": 4},
    "ml-1m":       {"rating_threshold": 4},
    "gowalla":     {"rating_threshold": 0},
    "amazon-book": {"rating_threshold": 0},
}

SLOT_CAPACITY = 4096  # CKKS n_poly=8192 → S = n_poly/2 = 4096 slots


def rot_count(d):
    """Rotation count for degree d: f(d) = d * ceil(log2(d)) (per-item HE schedule)."""
    return int(d * math.ceil(math.log2(max(d, 2))))


def load_dataset(name, data_dir="data"):
    cfg = DATASETS[name]
    return RecDataset(
        os.path.join(data_dir, name),
        dataset_name=name,
        rating_threshold=cfg["rating_threshold"],
    )


def build_oblirec(dataset, privacy_weight=0.0, seed=42):
    """Build ObliRec with default log-scale bucketing (Table 3 configuration).

    Log-scale bucketing is the default used in the main structural audit.
    Pass --privacy-weight > 0 with bucketing="adaptive" to reproduce the
    adaptive bucketing variants reported in Appendix E.
    """
    return ObliRec(
        dataset,
        seed=seed,
        bucketing="log",
        privacy_weight=privacy_weight,
    )


def collect_traces(dataset, oblipack, regime, decoy_k=1):
    """Build ExecutionTrace objects for all users under a given regime.

    Each ExecutionTrace models the server-observable metadata for one
    user's encrypted query:
      - n_rotations:   observable rotation count (degree proxy under sparse)
      - chunk_indices: frozenset of submitted chunk indices
      - n_chunks:      |chunk_indices|
      - true_degree:   ground-truth user degree (not observable by server;
                       used only for MAE metric computation)

    Regime semantics (§4.2):
      sparse   : n_rotations = d*ceil(log2(d)), chunk_indices = active_chunks
                 → channel 1 + channel 2 both exposed
                 (n_rotations reveals exact degree d via f-inversion; MAE = 0)
      bucketed : n_rotations = B*ceil(log2(B)) for bucket target B,
                 chunk_indices = active_chunks
                 → channel 1 hidden (all bucket members share same n_rotations),
                   channel 2 still exposed
      decoy    : n_rotations = B*ceil(log2(B)),
                 chunk_indices = active_chunks ∪ {k random inactive chunks}
                 Static decoy (per-user deterministic seed) models the
                 fixed-RNG problem described in §6.5: the server can
                 fingerprint users by their fixed decoy pattern.
      dense    : n_rotations = B*ceil(log2(B)), chunk_indices = {0,...,C-1}
                 → modeled chunk-index channel removed; degree coarsened to
                   bucket assignment (Channel 1 residual remains)
    """
    C = math.ceil(dataset.n_items / SLOT_CAPACITY)
    item_to_chunk = np.arange(dataset.n_items) // SLOT_CAPACITY

    traces = []
    for user_id, items_set in dataset.train_dict.items():
        items = list(items_set)
        if not items:
            continue
        d_u = len(items)
        active_chunks = frozenset(
            int(item_to_chunk[i]) for i in items if i < dataset.n_items
        )

        if regime == "sparse":
            # Channel 1: n_rotations = d*ceil(log2(d)) → degree recoverable by
            # inverting f; MAE = 0 (degree fully exposed).
            # Channel 2: chunk_indices reveals item-range access pattern.
            n_rot = rot_count(d_u)
            submitted = active_chunks

        else:
            # Bucket target pads neighbor list to uniform size (§4.2)
            padded = oblipack.user_padded_neighbors.get(user_id)
            if padded is not None:
                bucket_target = len(padded)
            else:
                bid = oblipack.adaptive_bucket_id(d_u)
                bucket_target = oblipack.adaptive_bucket_max(bid)
            # All users in a bucket share the same rotation count → channel 1 hidden.
            n_rot = rot_count(bucket_target)

            if regime == "bucketed":
                # Only channel 1 defended: chunk_indices still leaks item ranges
                submitted = active_chunks

            elif regime == "decoy":
                # Adds k static decoy chunks per user (fixed per-user seed).
                # Static assignment means the server can still fingerprint users
                # by their (active_chunks ∪ fixed_decoy) pattern (§6.5).
                inactive = list(set(range(C)) - active_chunks)
                k = min(decoy_k, len(inactive))
                if k > 0 and inactive:
                    user_rng = np.random.RandomState(user_id)
                    decoy = frozenset(
                        user_rng.choice(inactive, k, replace=False).tolist()
                    )
                else:
                    decoy = frozenset()
                submitted = active_chunks | decoy

            elif regime == "dense":
                # All C chunks submitted: chunk_indices same for all users
                # → channel 2 completely hidden
                submitted = frozenset(range(C))

            else:
                raise ValueError(f"Unknown regime: {regime!r}")

        traces.append(ExecutionTrace(
            user_id=user_id,
            true_degree=d_u,
            n_chunks=len(submitted),
            chunk_indices=submitted,
            n_rotations=n_rot,
            wall_time_ms=0.0,
        ))

    return traces


def neighbor_blowup(dataset, oblipack):
    """Compute average padded/true degree ratio across all users (Nbr. Blowup)."""
    total_real = total_pad = 0
    for u, items in dataset.train_dict.items():
        d = len(items)
        if d == 0:
            continue
        padded = oblipack.user_padded_neighbors.get(u)
        if padded is not None:
            pad_size = len(padded)
        else:
            bid = oblipack.adaptive_bucket_id(d)
            pad_size = oblipack.adaptive_bucket_max(bid)
        total_real += d
        total_pad += pad_size
    return total_pad / max(total_real, 1)


def run_audit(dataset_name, data_dir="data", privacy_weight=10.0,
              decoy_k=1, seed=42):
    """Run full structural leakage audit for one dataset. Returns result dict."""
    dataset = load_dataset(dataset_name, data_dir)
    C = math.ceil(dataset.n_items / SLOT_CAPACITY)
    n_train = sum(len(v) for v in dataset.train_dict.values())
    print(f"  [{dataset_name}] {dataset.n_users} users, {dataset.n_items} items, "
          f"{n_train} train interactions, C={C} chunks")

    print("  Building ObliRec ...", end=" ", flush=True)
    oblipack = build_oblirec(dataset, privacy_weight=privacy_weight, seed=seed)
    nbr_blowup = neighbor_blowup(dataset, oblipack)
    n_buckets = len(oblipack.user_buckets)
    print(f"done. {n_buckets} adaptive buckets, mean neighbor blowup={nbr_blowup:.2f}x")

    def bucket_fn(trace):
        return oblipack.adaptive_bucket_id(trace.true_degree)

    # Interaction-recovery F1: popularity-prior floor, regime-independent
    pop_f1 = popularity_prior_f1(dataset.train_dict, K=20)
    print(f"  Popularity-prior F1@20 = {pop_f1:.4f} (regime-independent floor)")

    # Mean degree for normalizing Deg.MAE to % of mean degree (Table 3 column)
    degrees = [len(v) for v in dataset.train_dict.values() if v]
    mean_degree = float(np.mean(degrees)) if degrees else 1.0

    results = {}
    regimes = ["sparse", "bucketed", "decoy", "dense"]

    for regime in regimes:
        traces = collect_traces(dataset, oblipack, regime, decoy_k)
        bfn = None if regime == "sparse" else bucket_fn
        m = compute_leakage_metrics(traces, bucket_fn=bfn)
        blowup = 1.0 if regime == "sparse" else nbr_blowup
        mae_pct = 100.0 * m.degree_mae / mean_degree
        results[regime] = {
            "degree_mae":               round(m.degree_mae, 4),
            "degree_mae_pct":           round(mae_pct, 2),
            "reid_accuracy":            round(m.reid_accuracy, 6),
            "mean_anonymity_set":       round(m.mean_anonymity_set, 2),
            "min_anonymity_set":        m.min_anonymity_set,
            "chunk_unique_pct":         round(m.chunk_pattern_unique_pct, 4),
            "nbr_blowup":               round(blowup, 3),
            "interaction_recovery_f1":  round(pop_f1, 4),
        }
        print(f"  {regime:<10} Re-ID={m.reid_accuracy:.4f}  "
              f"Deg.MAE={m.degree_mae:.2f} ({mae_pct:.1f}% of mean)  "
              f"AnonymSet={m.mean_anonymity_set:.0f}  "
              f"MinAnon={m.min_anonymity_set}  "
              f"F1@20={pop_f1:.4f}")

    return results, C


def print_table(dataset_name, results, C):
    """Print Table 3-style structural leakage summary."""
    print(f"\n{'='*74}")
    print(f"  Structural Leakage Audit — {dataset_name}  (C={C}, all users)")
    print(f"{'='*74}")
    hdr = (f"  {'Regime':<12} {'Deg.MAE%':>9} {'Re-ID↓':>9} "
           f"{'MeanAnon↑':>10} {'MinAnon↑':>9} {'ChunkUniq':>10} {'Blowup':>8}")
    print(hdr)
    print("  " + "-" * 70)
    for regime, m in results.items():
        print(
            f"  {regime:<12} "
            f"{m['degree_mae_pct']:>8.1f}% "
            f"{m['reid_accuracy']:>9.4f} "
            f"{m['mean_anonymity_set']:>10.1f} "
            f"{m['min_anonymity_set']:>9} "
            f"{m['chunk_unique_pct']:>10.4f} "
            f"{m['nbr_blowup']:>7.2f}x"
        )
    print(f"{'='*74}")

    # Highlight the key finding
    sparse_reid = results["sparse"]["reid_accuracy"]
    dense_reid  = results["dense"]["reid_accuracy"]
    if dense_reid > 0:
        reduction = sparse_reid / dense_reid
        print(f"\n  Key finding: sparse→dense Re-ID reduction = {reduction:.0f}x "
              f"({sparse_reid:.3f} → {dense_reid:.4f})")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Structural leakage audit (reproduces Table 3)"
    )
    parser.add_argument("--dataset", required=True,
                        choices=list(DATASETS) + ["all"],
                        help="Dataset to audit")
    parser.add_argument("--data-dir", default="data",
                        help="Root directory containing dataset folders")
    parser.add_argument("--privacy-weight", type=float, default=0.0,
                        help="λ for adaptive bucketing variant (Appendix E); "
                             "default 0.0 uses log-scale bucketing as in Table 3")
    parser.add_argument("--decoy-k", type=int, default=1,
                        help="Number of static decoy chunks in decoy mode")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    datasets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    all_results = {}

    for ds_name in datasets:
        print(f"\n{'='*74}")
        print(f"  Auditing: {ds_name}")
        print(f"{'='*74}")
        results, C = run_audit(
            ds_name,
            data_dir=args.data_dir,
            privacy_weight=args.privacy_weight,
            decoy_k=args.decoy_k,
            seed=args.seed,
        )
        print_table(ds_name, results, C)
        all_results[ds_name] = {"C": C, "regimes": results}

    out = "results_structural_audit.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved → {out}")


if __name__ == "__main__":
    main()
