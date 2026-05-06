"""
Item-layout sensitivity audit (Appendix experiment).

Tests whether chunk-index leakage (Channel 2) is a dataset artifact or
a deployment-layout effect.  We re-map item IDs under three orderings:
  original   : item_to_chunk = item_id // slot_capacity  (default)
  random     : items randomly permuted before chunk assignment
  popularity : most-popular items assigned to lowest chunk IDs

For each layout, every user's interaction set is mapped to chunk indices
and we compute the standard audit metrics without running HE.

Datasets: gowalla, amazon-book
Output  : results_item_reorder.json + printed table
"""

import json
import math
import sys, os
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from src.dataset import RecDataset

SLOT_CAPACITY = 4096
SEED          = 42


# ------------------------------------------------------------------ #
#  Metric helpers (inline; no HE dependency)
# ------------------------------------------------------------------ #

def chunk_unique_pct(fingerprints):
    counts = Counter(fingerprints)
    n_unique = sum(1 for fp in fingerprints if counts[fp] == 1)
    return n_unique / len(fingerprints) if fingerprints else 0.0


def anon_set_stats(fingerprints):
    counts = Counter(fingerprints)
    sizes = [counts[fp] for fp in fingerprints]
    return float(np.mean(sizes)), int(np.min(sizes))


def reid_exact_match(fp_s1, fp_s2):
    """Fraction of users whose session-1 fingerprint matches exactly one user
    in session-2 (minimum-distance attacker).  Both lists must be co-indexed."""
    correct = 0
    s2_counts = Counter(fp_s2)
    for fp in fp_s1:
        if fp in s2_counts:
            # attacker picks uniformly among all session-2 users with same fp
            correct += 1.0 / s2_counts[fp]
    return correct / len(fp_s1) if fp_s1 else 0.0


# ------------------------------------------------------------------ #
#  Layout builders
# ------------------------------------------------------------------ #

def make_item_to_chunk(n_items, layout, train_dict, rng):
    """Return array of length n_items: item_id -> chunk_id."""
    if layout == "original":
        return np.arange(n_items) // SLOT_CAPACITY

    elif layout == "random":
        perm = rng.permutation(n_items)
        rank = np.empty(n_items, dtype=np.int64)
        rank[perm] = np.arange(n_items)
        return rank // SLOT_CAPACITY

    elif layout == "popularity":
        # Count how many users interacted with each item
        item_freq = np.zeros(n_items, dtype=np.int64)
        for items in train_dict.values():
            for i in items:
                if i < n_items:
                    item_freq[i] += 1
        # Most popular items → lowest chunk IDs
        order = np.argsort(-item_freq)   # descending popularity
        rank = np.empty(n_items, dtype=np.int64)
        rank[order] = np.arange(n_items)
        return rank // SLOT_CAPACITY

    else:
        raise ValueError(f"Unknown layout: {layout}")


# ------------------------------------------------------------------ #
#  Per-layout audit
# ------------------------------------------------------------------ #

def audit_layout(dataset, users, layout, rng):
    item_to_chunk = make_item_to_chunk(
        dataset.n_items, layout, dataset.train_dict, rng)
    n_chunks_total = math.ceil(dataset.n_items / SLOT_CAPACITY)

    # Build chunk-index fingerprints from the full training set (single session).
    # This matches the main-paper protocol: fp = frozenset of chunk IDs accessed
    # when submitting ALL training items.  Re-ID = mean(1/group_size).
    fps = []
    for u in users:
        items = list(dataset.train_dict.get(u, []))
        fp = frozenset(int(item_to_chunk[i]) for i in items if i < dataset.n_items)
        fps.append(fp)

    counts    = Counter(fps)
    uniq_pct  = chunk_unique_pct(fps)
    mean_anon, min_anon = anon_set_stats(fps)
    reid_acc  = float(np.mean([1.0 / counts[fp] for fp in fps]))

    return {
        "layout":           layout,
        "n_chunks_total":   int(n_chunks_total),
        "chunk_unique_pct": round(uniq_pct, 4),
        "mean_anon_set":    round(mean_anon, 2),
        "min_anon_set":     min_anon,
        "reid_acc":         round(reid_acc, 4),
    }


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

DATASETS = ["gowalla", "amazon-book"]
LAYOUTS  = ["original", "random", "popularity"]

def run_dataset(ds_name):
    dataset = RecDataset(f"data/{ds_name}", dataset_name=ds_name)
    users = [u for u, its in dataset.train_dict.items() if len(its) >= 2]

    results = {}
    for layout in LAYOUTS:
        r = audit_layout(dataset, users, layout, np.random.RandomState(SEED))
        results[layout] = r
        print(f"  {ds_name:12s}  {layout:12s}  "
              f"unique%={r['chunk_unique_pct']*100:5.1f}  "
              f"mean-anon={r['mean_anon_set']:6.1f}  "
              f"min-anon={r['min_anon_set']:4d}  "
              f"reid={r['reid_acc']*100:5.2f}%")
    return results


def main():
    print(f"\n{'='*75}")
    print("  Item-Layout Sensitivity Audit")
    print(f"  slot_capacity={SLOT_CAPACITY}, all eligible users")
    print(f"{'='*75}")
    print(f"  {'Dataset':12s}  {'Layout':12s}  {'Unique%':>8}  "
          f"{'MeanAnon':>9}  {'MinAnon':>8}  {'Re-ID%':>7}")
    print("  " + "-" * 65)

    all_results = {}
    for ds_name in DATASETS:
        all_results[ds_name] = run_dataset(ds_name)

    out = "results_item_reorder.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved → {out}")


if __name__ == "__main__":
    main()
