"""
Generate Figure: Correlation Paradox — Degree vs Observable Side-Channel.

Shows:
- Left: Sparse HE (noisy cloud) — degree leaks through chunk/rotation count with noise
- Right: ObliRec (step function) — zero within-bucket variance, only bucket identity leaks

Usage:
    python generate_correlation_figure.py [--dataset gowalla]
"""

import argparse
import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

from src.dataset import RecDataset
from src.oblirec import ObliRec

DATASET_DEFAULTS = {
    "ml-100k":     {"rating_threshold": 4},
    "ml-1m":       {"rating_threshold": 4},
    "gowalla":     {"rating_threshold": 0},
    "amazon-book": {"rating_threshold": 0},
}

DATASET_LABELS = {
    "ml-100k": "ML-100K",
    "ml-1m": "ML-1M",
    "gowalla": "Gowalla",
    "amazon-book": "Amazon-Book",
}


def simulate_sparse_he_observable(degrees, n_slots=8192, noise_std=0.0):
    """Simulate chunk-count observable for sparse HE (proportional to degree + noise)."""
    chunks = np.array([max(1, math.ceil(d / n_slots)) for d in degrees], dtype=float)
    # Add timing noise (simulates network/system jitter)
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, len(degrees))
        chunks = chunks + noise
    return chunks


def simulate_rotation_observable(degrees, noise_std=0.5):
    """Simulate FicGCN-style rotation count (= degree, but with system noise)."""
    rots = np.array(degrees, dtype=float)
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, len(degrees))
        rots = rots + noise
    return rots


def get_oblipack_observable(oblipack, user_ids):
    """Get ObliPack bucket pad target for each user (the only thing observable)."""
    observables = []
    for u in user_ids:
        deg = oblipack.user_degrees[u]
        bid = oblipack.bucket_id(deg)
        pad = oblipack.bucket_max(bid)
        observables.append(pad)
    return np.array(observables, dtype=float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="gowalla",
                        choices=list(DATASET_DEFAULTS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-sample", type=int, default=2000)
    args = parser.parse_args()

    defaults = DATASET_DEFAULTS[args.dataset]
    dataset = RecDataset(f"data/{args.dataset}", dataset_name=args.dataset,
                         rating_threshold=defaults["rating_threshold"],
                         seed=args.seed)

    oblipack = ObliRec(dataset, seed=args.seed, bucketing="log")

    # Sample users
    rng = np.random.RandomState(args.seed)
    all_users = sorted(oblipack.user_degrees.keys())
    sample_users = rng.choice(all_users, size=min(args.n_sample, len(all_users)),
                              replace=False)
    degrees = np.array([oblipack.user_degrees[u] for u in sample_users])

    # Observables
    sparse_obs = simulate_rotation_observable(degrees, noise_std=1.0)
    oblipack_obs = get_oblipack_observable(oblipack, sample_users)

    # Metrics
    r_sparse = np.corrcoef(degrees, sparse_obs)[0, 1]
    r_oblipack = np.corrcoef(degrees, oblipack_obs)[0, 1]
    mae_sparse = np.mean(np.abs(degrees - sparse_obs))
    mae_oblipack = np.mean(np.abs(degrees - oblipack_obs))

    # Anonymity: unique observable values = number of distinguishable groups
    n_unique_sparse = len(set(np.round(sparse_obs, 0).astype(int)))
    n_unique_oblipack = len(set(oblipack_obs.astype(int)))
    avg_anon_sparse = len(degrees) / n_unique_sparse
    avg_anon_oblipack = len(degrees) / n_unique_oblipack

    ds_label = DATASET_LABELS[args.dataset]

    # --- Figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)

    # Left: Sparse HE (rotation-count attack)
    ax1.scatter(degrees, sparse_obs, s=4, alpha=0.3, c="#d62728", edgecolors="none")
    dmin, dmax = degrees.min(), degrees.max()
    ax1.plot([dmin, dmax], [dmin, dmax], 'k--', lw=1, alpha=0.5, label="y = x")
    ax1.set_xlabel("True Degree", fontsize=12)
    ax1.set_ylabel("Observable (Rotation Count)", fontsize=12)
    ax1.set_title(f"(a) Sparse HE — No Protection", fontsize=12, fontweight="bold")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.legend(fontsize=9, loc="upper left")

    # Stats box for sparse
    stats_text_a = (f"MAE = {mae_sparse:.1f}\n"
                    f"Unique obs. = {n_unique_sparse}\n"
                    f"Avg. anon. set = {avg_anon_sparse:.1f}")
    ax1.text(0.97, 0.03, stats_text_a, transform=ax1.transAxes,
             fontsize=8.5, verticalalignment="bottom", horizontalalignment="right",
             bbox=dict(boxstyle="round,pad=0.4", fc="#ffe0e0", ec="#d62728", alpha=0.9))

    # Right: ObliRec (bucket pad target)
    unique_pads = sorted(set(oblipack_obs))
    cmap = plt.cm.Set2
    for idx, pad in enumerate(unique_pads):
        mask = oblipack_obs == pad
        n_in_bucket = mask.sum()
        ax2.scatter(degrees[mask], oblipack_obs[mask], s=4, alpha=0.4,
                    c=[cmap(idx % 8)], edgecolors="none")
        # Annotate bucket anonymity set size
        if n_in_bucket > 20:
            x_mid = np.median(degrees[mask])
            ax2.annotate(f"n={n_in_bucket}", xy=(x_mid, pad),
                         fontsize=6.5, alpha=0.7, ha="center", va="bottom",
                         xytext=(0, 3), textcoords="offset points")

    for pad in unique_pads:
        ax2.axhline(y=pad, color="gray", lw=0.5, alpha=0.3)

    ax2.set_xlabel("True Degree", fontsize=12)
    ax2.set_ylabel("Observable (Padded Degree)", fontsize=12)
    ax2.set_title(f"(b) ObliPack — Bucket Padding", fontsize=12, fontweight="bold")
    ax2.set_xscale("log")
    ax2.set_yscale("log")

    # Stats box for ObliPack
    stats_text_b = (f"MAE = {mae_oblipack:.1f}\n"
                    f"Unique obs. = {n_unique_oblipack}\n"
                    f"Avg. anon. set = {avg_anon_oblipack:.0f}")
    ax2.text(0.97, 0.03, stats_text_b, transform=ax2.transAxes,
             fontsize=8.5, verticalalignment="bottom", horizontalalignment="right",
             bbox=dict(boxstyle="round,pad=0.4", fc="#e0ffe0", ec="#2ca02c", alpha=0.9))

    fig.suptitle(f"Degree vs. Side-Channel Observable — {ds_label}\n"
                 f"High correlation does not imply fine-grained leakage",
                 fontsize=12, fontweight="bold", y=1.04)
    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    out_pdf = f"figures/fig_correlation_paradox_{args.dataset}.pdf"
    out_png = f"figures/fig_correlation_paradox_{args.dataset}.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=300)
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")
    print(f"\n  {'Metric':<25} {'Sparse HE':>12} {'ObliPack':>12}")
    print(f"  {'-'*50}")
    print(f"  {'Pearson r':<25} {r_sparse:>12.4f} {r_oblipack:>12.4f}")
    print(f"  {'MAE':<25} {mae_sparse:>12.1f} {mae_oblipack:>12.1f}")
    print(f"  {'Unique observables':<25} {n_unique_sparse:>12} {n_unique_oblipack:>12}")
    print(f"  {'Avg anonymity set':<25} {avg_anon_sparse:>12.1f} {avg_anon_oblipack:>12.0f}")
    print(f"\n  Key insight: r ≈ {r_oblipack:.2f} for ObliPack, but MAE = {mae_oblipack:.1f}")
    print(f"  and only {n_unique_oblipack} distinguishable groups (vs {n_unique_sparse} for sparse).")


if __name__ == "__main__":
    main()
