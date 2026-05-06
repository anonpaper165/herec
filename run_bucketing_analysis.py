"""
Bucketing strategy comparison: log-scale vs padding-optimal vs privacy-aware.

For each dataset, compares bucketing strategies at the same K (number of
buckets) on padding cost, anonymity set sizes, and re-ID risk.
"""

import json
import math
import numpy as np
from collections import Counter

from src.dataset import RecDataset
from src.oblirec import ObliRec


DATASETS = {
    "ml-100k":     {"rating_threshold": 4},
    "ml-1m":       {"rating_threshold": 4},
    "gowalla":     {"rating_threshold": 0},
    "amazon-book": {"rating_threshold": 0},
}

PRIVACY_WEIGHTS = [0, 0.1, 1.0, 10.0, 100.0]


def compute_log_buckets(degrees):
    """Compute log-scale bucket assignments."""
    buckets = {}
    for d in degrees:
        if d <= 0:
            continue
        bid = ObliRec.bucket_id(d)
        buckets.setdefault(bid, []).append(d)
    return buckets


def compute_bucket_metrics(buckets, degrees):
    """Compute metrics for a bucketing strategy."""
    total_padding = 0
    total_real = 0
    bucket_sizes = []

    for bid, degs in sorted(buckets.items()):
        pad_target = max(degs)
        real = sum(degs)
        padded = len(degs) * pad_target
        total_real += real
        total_padding += (padded - real)
        bucket_sizes.append(len(degs))

    blowup = (total_real + total_padding) / max(total_real, 1)
    min_anon = min(bucket_sizes) if bucket_sizes else 0
    max_anon = max(bucket_sizes) if bucket_sizes else 0
    mean_anon = float(np.mean(bucket_sizes)) if bucket_sizes else 0
    worst_reid = 1.0 / max(min_anon, 1)

    return {
        "n_buckets": len(buckets),
        "total_padding": total_padding,
        "total_real": total_real,
        "blowup": blowup,
        "min_anon_set": min_anon,
        "mean_anon_set": mean_anon,
        "max_anon_set": max_anon,
        "worst_reid_prob": worst_reid,
        "bucket_sizes": bucket_sizes,
    }


def dp_buckets_to_dict(degrees, boundaries, pad_targets, unique_degs):
    """Convert DP solution to bucket dict format."""
    from collections import Counter
    deg_counts = Counter(degrees)

    buckets = {}
    for bid, ((start, end), pad_target) in enumerate(zip(boundaries, pad_targets)):
        degs_in_bucket = []
        for idx in range(start, end + 1):
            d = unique_degs[idx]
            degs_in_bucket.extend([d] * deg_counts[d])
        if degs_in_bucket:
            buckets[bid] = degs_in_bucket
    return buckets


def main():
    results = {}

    for ds_name, ds_cfg in DATASETS.items():
        print(f"\n{'='*70}")
        print(f"  {ds_name}")
        print(f"{'='*70}")

        dataset = RecDataset(f"data/{ds_name}", dataset_name=ds_name,
                             rating_threshold=ds_cfg["rating_threshold"], seed=42)

        # Compute degrees
        degrees = []
        for uid in range(dataset.n_users):
            items = dataset.train_dict.get(uid, [])
            if items:
                degrees.append(len(items))

        N = len(degrees)
        deg_counter = Counter(degrees)
        unique_degs = sorted(deg_counter.keys())

        print(f"  Users: {N}, Unique degrees: {len(unique_degs)}, "
              f"Max degree: {max(degrees)}")

        # 1. Log-scale bucketing
        log_buckets = compute_log_buckets(degrees)
        K = len(log_buckets)
        log_metrics = compute_bucket_metrics(log_buckets, degrees)

        print(f"\n  Log-scale (K={K}):")
        print(f"    Padding: {log_metrics['total_padding']:,}, "
              f"Blowup: {log_metrics['blowup']:.3f}x")
        print(f"    Anon set: min={log_metrics['min_anon_set']}, "
              f"mean={log_metrics['mean_anon_set']:.0f}, "
              f"max={log_metrics['max_anon_set']}")
        print(f"    Worst re-ID: {log_metrics['worst_reid_prob']:.4f}")

        ds_results = {
            "n_users": N,
            "n_unique_degrees": len(unique_degs),
            "max_degree": max(degrees),
            "K": K,
            "log_scale": log_metrics,
            "strategies": {},
        }

        # 2. DP-optimal strategies at same K
        print(f"\n  {'Strategy':<25} {'Padding':>10} {'Blowup':>8} "
              f"{'MinAnon':>8} {'MeanAnon':>9} {'ReID':>8}")
        print(f"  {'-'*70}")

        # Log-scale row
        print(f"  {'Log-scale':<25} {log_metrics['total_padding']:>10,} "
              f"{log_metrics['blowup']:>8.3f} "
              f"{log_metrics['min_anon_set']:>8} "
              f"{log_metrics['mean_anon_set']:>9.0f} "
              f"{log_metrics['worst_reid_prob']:>8.4f}")

        for lam in PRIVACY_WEIGHTS:
            label = f"λ={lam}" if lam > 0 else "Padding-only"

            if lam == 0:
                boundaries, pad_targets, cost = ObliRec.optimal_buckets(
                    degrees, K, min_anon_set=1)
            else:
                boundaries, pad_targets, cost = ObliRec.optimal_buckets_privacy(
                    degrees, K, min_anon_set=1, privacy_weight=lam)

            buckets = dp_buckets_to_dict(degrees, boundaries, pad_targets,
                                         unique_degs)
            metrics = compute_bucket_metrics(buckets, degrees)

            print(f"  {label:<25} {metrics['total_padding']:>10,} "
                  f"{metrics['blowup']:>8.3f} "
                  f"{metrics['min_anon_set']:>8} "
                  f"{metrics['mean_anon_set']:>9.0f} "
                  f"{metrics['worst_reid_prob']:>8.4f}")

            ds_results["strategies"][f"lambda_{lam}"] = {
                "privacy_weight": lam,
                "boundaries": [(int(s), int(e)) for s, e in boundaries],
                "pad_targets": [int(p) for p in pad_targets],
                **metrics,
                "bucket_sizes": [int(x) for x in metrics["bucket_sizes"]],
            }

        results[ds_name] = ds_results

    # Save
    with open("results_bucketing_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to results_bucketing_analysis.json")

    # Summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY: Privacy-Efficiency Pareto")
    print(f"{'='*70}")
    for ds, r in results.items():
        print(f"\n  {ds} (K={r['K']}):")
        log = r["log_scale"]
        print(f"    Log-scale:    blowup={log['blowup']:.3f}x, "
              f"min_anon={log['min_anon_set']}, "
              f"worst_reID={log['worst_reid_prob']:.4f}")
        for key, s in r["strategies"].items():
            lam = s["privacy_weight"]
            label = f"λ={lam}" if lam > 0 else "Padding-only"
            print(f"    {label:<14}: blowup={s['blowup']:.3f}x, "
                  f"min_anon={s['min_anon_set']}, "
                  f"worst_reID={s['worst_reid_prob']:.4f}")


if __name__ == "__main__":
    main()
