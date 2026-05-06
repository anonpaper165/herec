"""
GraphSAGE-style fixed-fanout trace simulation (Appendix experiment).

We do NOT implement a full HE GraphSAGE protocol.
Goal: test whether the structural audit observables (Channel 1: support-size
leakage via operation-volume signals; Channel 2: support-location leakage via
chunk-index set) transfer to a
different sparse graph-serving pattern --- fixed-fanout neighborhood sampling.

Under fixed fanout k:
  - Each user submits exactly min(k, d_u) encrypted item slots.
  - n_rotations is capped at k for high-degree users → Channel 1 leakage
    shrinks to zero for users with d_u >= k.
  - chunk-index pattern is determined by WHICH k items were sampled
    → Channel 2 leakage persists if items spread across chunks.

Fanout values tested: 10, 25 (single-hop)
Re-ID metric: single-session mean(1/group_size), following the structural audit protocol.
  Full-support baseline uses all training items (deterministic fingerprint).
  Fanout rows sample k items once per user; Re-ID measures identifiability
  from a single query observation.

Datasets: gowalla, amazon-book
Output  : results_graphsage_trace.json + printed table
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
FANOUTS       = [10, 25]


# ------------------------------------------------------------------ #
#  Metric helpers
# ------------------------------------------------------------------ #

def chunk_unique_pct(fingerprints):
    counts = Counter(fingerprints)
    n_unique = sum(1 for fp in fingerprints if counts[fp] == 1)
    return n_unique / len(fingerprints) if fingerprints else 0.0


def anon_set_stats(fingerprints):
    counts = Counter(fingerprints)
    sizes = [counts[fp] for fp in fingerprints]
    return float(np.mean(sizes)), int(np.min(sizes))


def degree_mae(true_degs, effective_degs):
    """MAE between true degree and what the server observes (effective = min(k, d)).
    Server observes the capped value; MAE > 0 iff some users have d > k."""
    errors = [abs(ed - td) for td, ed in zip(true_degs, effective_degs)]
    return float(np.mean(errors))


# ------------------------------------------------------------------ #
#  Fanout simulation
# ------------------------------------------------------------------ #

def simulate_fanout(dataset, users, fanout, seed):
    """Simulate one round of fixed-fanout sampling per user.

    Samples k items per user (all items if d_u <= k), maps to chunk indices,
    and computes single-session Re-ID = mean(1/group_size).
    """
    item_to_chunk = np.arange(dataset.n_items) // SLOT_CAPACITY
    rng_s1 = np.random.RandomState(seed)

    fp_s1          = []
    true_degs      = []
    effective_degs = []
    n_capped       = 0   # users whose true degree > fanout (Channel 1 hidden)

    for u in users:
        items = list(dataset.train_dict.get(u, []))
        true_d = len(items)
        true_degs.append(true_d)
        effective_d = min(fanout, true_d)
        effective_degs.append(effective_d)
        if true_d > fanout:
            n_capped += 1

        # Sample without replacement (use all items if d <= fanout)
        if true_d <= fanout:
            s1 = items
        else:
            idx1 = rng_s1.choice(true_d, size=fanout, replace=False)
            s1 = [items[i] for i in idx1]

        fp1 = frozenset(int(item_to_chunk[i]) for i in s1 if i < dataset.n_items)
        fp_s1.append(fp1)

    ch1_mae        = degree_mae(true_degs, effective_degs)
    uniq_pct       = chunk_unique_pct(fp_s1)
    mean_anon, min_anon = anon_set_stats(fp_s1)
    counts         = Counter(fp_s1)
    reid_acc       = float(np.mean([1.0 / counts[fp] for fp in fp_s1]))
    pct_capped     = n_capped / len(users)

    return {
        "fanout":            fanout,
        "n_users":           len(users),
        "pct_degree_gt_k":   round(pct_capped, 4),
        "ch1_mae":           round(ch1_mae, 2),
        "ch2_chunk_unique":  round(uniq_pct, 4),
        "mean_anon_set":     round(mean_anon, 2),
        "min_anon_set":      min_anon,
        "reid_acc":          round(reid_acc, 4),
    }


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

DATASETS = ["gowalla", "amazon-book"]


def run_dataset(ds_name):
    dataset = RecDataset(f"data/{ds_name}", dataset_name=ds_name)
    users   = [u for u, its in dataset.train_dict.items() if len(its) >= 1]

    # Baseline: no-fanout (full support, i.e. LightGCN sparse regime).
    # All training items are always submitted → fingerprint is deterministic.
    # Re-ID = mean(1/group_size), single-session structural audit protocol.
    print(f"\n  {ds_name}  [baseline: full support / no fanout]")
    base_item_to_chunk = np.arange(dataset.n_items) // SLOT_CAPACITY
    fp_base = []
    for u in users:
        items = list(dataset.train_dict.get(u, []))
        fp_base.append(frozenset(int(base_item_to_chunk[i]) for i in items if i < dataset.n_items))
    base_counts = Counter(fp_base)
    baseline = {
        "fanout":           "full",
        "ch1_mae":          0.0,
        "ch2_chunk_unique": round(chunk_unique_pct(fp_base), 4),
        "mean_anon_set":    round(anon_set_stats(fp_base)[0], 2),
        "min_anon_set":     anon_set_stats(fp_base)[1],
        "reid_acc":         round(float(np.mean([1.0 / base_counts[fp] for fp in fp_base])), 4),
        "pct_degree_gt_k":  0.0,
    }
    print(f"    fanout=full  unique%={baseline['ch2_chunk_unique']*100:5.1f}  "
          f"mean-anon={baseline['mean_anon_set']:6.1f}  reid={baseline['reid_acc']*100:5.2f}%")

    results = {"baseline_full_support": baseline}
    for k in FANOUTS:
        r = simulate_fanout(dataset, users, fanout=k, seed=SEED)
        results[f"fanout_{k}"] = r
        print(f"    fanout={k:3d}   unique%={r['ch2_chunk_unique']*100:5.1f}  "
              f"mean-anon={r['mean_anon_set']:6.1f}  reid={r['reid_acc']*100:5.2f}%  "
              f"ch1_mae={r['ch1_mae']:.1f}  pct_capped={r['pct_degree_gt_k']*100:.1f}%")
    return results


def main():
    print(f"\n{'='*75}")
    print("  GraphSAGE-Style Fixed-Fanout Trace Simulation")
    print(f"  slot_capacity={SLOT_CAPACITY}, all eligible users")
    print(f"  NOTE: no HE protocol implemented; trace-level simulation only")
    print(f"{'='*75}")

    all_results = {}
    for ds_name in DATASETS:
        all_results[ds_name] = run_dataset(ds_name)

    out = "results_graphsage_trace.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved → {out}")


if __name__ == "__main__":
    main()
