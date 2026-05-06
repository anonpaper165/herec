"""
Compare log-scale vs adaptive (DP-optimal) bucketing.

Metrics:
1. Total padding cost (sum of padding dummies across all users)
2. Padding blowup ratio
3. Anonymity set sizes
4. Number of buckets

Usage:
    python benchmark_adaptive.py --dataset gowalla
    python benchmark_adaptive.py --dataset amazon-book
"""

import argparse
import json
import math
import os
import time
import numpy as np
from collections import defaultdict

from src.dataset import RecDataset
from src.oblirec import ObliRec


DATASET_DEFAULTS = {
    "ml-100k":     {"rating_threshold": 4},
    "ml-1m":       {"rating_threshold": 4},
    "gowalla":     {"rating_threshold": 0},
    "amazon-book": {"rating_threshold": 0},
}


def compute_padding_stats(oblipack, label, is_adaptive=False):
    """Compute padding cost statistics for a given ObliPack instance."""
    total_real = 0
    total_padded = 0
    bucket_stats = []

    for bid in sorted(oblipack.user_buckets.keys()):
        users = oblipack.user_buckets[bid]
        if is_adaptive:
            pad_size = oblipack.adaptive_bucket_max(bid)
            lo, hi = oblipack.adaptive_bucket_range(bid)
        else:
            pad_size = oblipack.bucket_max(bid)
            lo, hi = oblipack.bucket_range(bid)

        real = sum(oblipack.user_degrees[u] for u in users)
        padded = len(users) * pad_size
        total_real += real
        total_padded += padded

        bucket_stats.append({
            "bucket": bid,
            "range": (lo, hi),
            "pad_target": pad_size,
            "n_users": len(users),
            "real_edges": real,
            "padded_edges": padded,
            "overhead": padded / max(real, 1),
        })

    blowup = total_padded / max(total_real, 1)
    anon_sizes = [s["n_users"] for s in bucket_stats]

    return {
        "label": label,
        "total_real": total_real,
        "total_padded": total_padded,
        "blowup": blowup,
        "n_buckets": len(bucket_stats),
        "min_anon_set": min(anon_sizes) if anon_sizes else 0,
        "mean_anon_set": np.mean(anon_sizes) if anon_sizes else 0,
        "median_anon_set": np.median(anon_sizes) if anon_sizes else 0,
        "buckets": bucket_stats,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="gowalla",
                        choices=list(DATASET_DEFAULTS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    defaults = DATASET_DEFAULTS[args.dataset]
    dataset = RecDataset(f"data/{args.dataset}", dataset_name=args.dataset,
                         rating_threshold=defaults["rating_threshold"],
                         seed=args.seed)

    print(f"\n{'='*70}")
    print(f"  Adaptive vs Log Bucketing — {args.dataset}")
    print(f"  Users: {dataset.n_users}, Items: {dataset.n_items}")
    print(f"{'='*70}")

    # --- Log bucketing (default) ---
    t0 = time.time()
    op_log = ObliRec(dataset, seed=args.seed, bucketing="log")
    t_log = time.time() - t0
    stats_log = compute_padding_stats(op_log, "Log (2^k)", is_adaptive=False)

    # --- Adaptive bucketing with same number of buckets ---
    n_log_buckets = stats_log["n_buckets"]

    # Try different bucket counts
    configs = [
        ("Adaptive (same K)", n_log_buckets),
        ("Adaptive (K+2)", n_log_buckets + 2),
        ("Adaptive (K+5)", n_log_buckets + 5),
        ("Adaptive (2K)", n_log_buckets * 2),
    ]

    all_stats = [stats_log]

    for label, n_b in configs:
        t0 = time.time()
        op_adaptive = ObliRec(dataset, seed=args.seed, bucketing="adaptive",
                               n_buckets=n_b, min_anon_set=10)
        t_adaptive = time.time() - t0
        stats = compute_padding_stats(op_adaptive, f"{label}, K={n_b}",
                                      is_adaptive=True)
        stats["build_time"] = t_adaptive
        all_stats.append(stats)

    # --- Print comparison ---
    print(f"\n  {'Strategy':<30} {'K':>4} {'Blowup':>8} {'Total Pad':>12} "
          f"{'Min Anon':>9} {'Mean Anon':>10}")
    print(f"  {'-'*75}")
    for s in all_stats:
        print(f"  {s['label']:<30} {s['n_buckets']:>4} {s['blowup']:>8.3f}x "
              f"{s['total_padded']:>12,} {s['min_anon_set']:>9} "
              f"{s['mean_anon_set']:>10.0f}")

    # --- Detailed bucket comparison: log vs best adaptive ---
    print(f"\n{'='*70}")
    print(f"  Detailed Bucket Breakdown")
    print(f"{'='*70}")

    for s in all_stats:
        print(f"\n  [{s['label']}] (blowup={s['blowup']:.3f}x, "
              f"K={s['n_buckets']})")
        print(f"  {'Bucket':>7} {'Range':>15} {'Pad→':>6} {'#Users':>8} "
              f"{'Overhead':>9}")
        print(f"  {'-'*50}")
        for b in s["buckets"]:
            lo, hi = b["range"]
            print(f"  B{b['bucket']:<5d} [{lo:>5d}-{hi:<5d}] {b['pad_target']:>5d} "
                  f"{b['n_users']:>8d} {b['overhead']:>8.2f}x")

    # --- Savings ---
    print(f"\n{'='*70}")
    print(f"  Padding Savings vs Log Bucketing")
    print(f"{'='*70}")
    log_total = stats_log["total_padded"]
    for s in all_stats[1:]:
        saving = (1 - s["total_padded"] / log_total) * 100
        print(f"  {s['label']:<35} saving: {saving:>+.1f}%")

    # Save results
    os.makedirs("models", exist_ok=True)
    out_path = f"models/{args.dataset}_adaptive_bucketing.json"
    # Simplify for JSON
    save_data = {}
    for s in all_stats:
        key = s["label"]
        save_data[key] = {
            "n_buckets": s["n_buckets"],
            "blowup": s["blowup"],
            "total_real": s["total_real"],
            "total_padded": s["total_padded"],
            "min_anon_set": s["min_anon_set"],
            "mean_anon_set": float(s["mean_anon_set"]),
        }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
