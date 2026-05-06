"""
HE timing benchmark — measures per-user chunked HE cost
for P2 (depth-2) and P2-MultiHop (depth-1) across protocols.

Measures per-mode latency across datasets at varying decoy budgets.

Usage:
    python benchmark_he_timing.py --dataset gowalla
    python benchmark_he_timing.py --dataset amazon-book
"""

import argparse
import json
import math
import os
import sys
import time
import numpy as np
import torch
import tenseal as ts

sys.path.insert(0, os.path.dirname(__file__))

from src.dataset import RecDataset
from src.model import LightGCN
from src.oblirec import ObliRec


DATASET_DEFAULTS = {
    "gowalla":     {"rating_threshold": 0, "emb_dim": 64, "n_layers": 3},
    "amazon-book": {"rating_threshold": 0, "emb_dim": 64, "n_layers": 3},
}

# Paper protocol: depth-1 uses n_poly=8192 (S=4096), depth-2 uses n_poly=16384 (S=8192)
POLY_DEPTH1 = 8192
POLY_DEPTH2 = 16384
SLOTS_DEPTH1 = POLY_DEPTH1 // 2   # 4096
SLOTS_DEPTH2 = POLY_DEPTH2 // 2   # 8192


def p(msg, **kw):
    """Print with immediate flush."""
    print(msg, flush=True, **kw)


def make_ckks_context(poly_modulus_degree, depth):
    """Create CKKS context for the experiment (128-bit security, scale 2^40)."""
    scale = 2 ** 40
    if depth == 1:
        coeff_mod = [60, 40, 40, 60]
    else:  # depth == 2
        coeff_mod = [60, 40, 40, 40, 60]
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod,
    )
    ctx.global_scale = scale
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    return ctx


def chunked_p2mh(ctx, indicator, E_agg, slots, n_trials=3):
    """
    Chunked P2-MultiHop: enc(indicator_chunk) @ E_agg_rows -> sum -> enc_user_emb
    Only submits non-zero chunks (sparse mode).
    Returns timing dict.
    """
    n_items = len(indicator)
    timings = []

    active_chunks = []
    for start in range(0, n_items, slots):
        end = min(start + slots, n_items)
        chunk = indicator[start:end]
        if np.any(chunk != 0):
            active_chunks.append((chunk, E_agg[start:end, :]))

    for trial in range(n_trials):
        t0 = time.perf_counter()
        enc_u = None
        for chunk, E_chunk in active_chunks:
            enc_chunk = ts.ckks_vector(ctx, chunk.tolist())
            enc_partial = enc_chunk.mm(E_chunk.tolist())
            enc_u = enc_partial if enc_u is None else enc_u + enc_partial
        timings.append((time.perf_counter() - t0) * 1000)

    return {
        "mean_ms": float(np.mean(timings)),
        "std_ms": float(np.std(timings)),
        "n_chunks": len(active_chunks),
    }


def chunked_p2mh_padded(ctx, indicator, E_agg, slots, n_trials=3):
    """Same but submits ALL chunks (dense/padded mode)."""
    n_items = len(indicator)
    timings = []

    all_chunks = []
    for start in range(0, n_items, slots):
        end = min(start + slots, n_items)
        all_chunks.append((indicator[start:end], E_agg[start:end, :]))

    for trial in range(n_trials):
        t0 = time.perf_counter()
        enc_u = None
        for chunk, E_chunk in all_chunks:
            enc_chunk = ts.ckks_vector(ctx, chunk.tolist())
            enc_partial = enc_chunk.mm(E_chunk.tolist())
            enc_u = enc_partial if enc_u is None else enc_u + enc_partial
        timings.append((time.perf_counter() - t0) * 1000)

    return {
        "mean_ms": float(np.mean(timings)),
        "std_ms": float(np.std(timings)),
        "n_chunks": len(all_chunks),
    }


def make_indicator(uid, train_dict, user_degrees, item_degrees, n_items):
    """Build weighted interaction indicator vector for user uid."""
    v = np.zeros(n_items, dtype=np.float64)
    d_u = user_degrees.get(uid, 1)
    for iid in train_dict.get(uid, []):
        d_i = item_degrees.get(iid, 1)
        w = 1.0 / math.sqrt(d_u * d_i)
        if iid < n_items:
            v[iid] = w
    return v


def run(dataset_name, n_sample=5, seed=42, n_trials=2):
    cfg = DATASET_DEFAULTS[dataset_name]
    device = torch.device("cpu")

    p(f"\n{'='*65}")
    p(f"  HE Timing Benchmark: {dataset_name}")
    p(f"{'='*65}")

    p("  Loading dataset...", end=" ")
    dataset = RecDataset(f"data/{dataset_name}", dataset_name=dataset_name,
                         rating_threshold=cfg["rating_threshold"], seed=seed)
    n_users, n_items = dataset.n_users, dataset.n_items
    C1 = math.ceil(n_items / SLOTS_DEPTH1)
    C2 = math.ceil(n_items / SLOTS_DEPTH2)
    p(f"done. Items={n_items}, C1={C1}, C2={C2}")

    p("  Loading model...", end=" ")
    ckpt = torch.load(f"models/{dataset_name}_best.pt",
                      map_location=device, weights_only=False)
    model = LightGCN(n_users, n_items,
                     ckpt["args"]["emb_dim"],
                     ckpt["args"]["n_layers"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    p("done.")

    p("  Building propagated embeddings...", end=" ")
    adj = dataset.adj_matrix.to(device)
    with torch.no_grad():
        E = torch.cat([model.user_embedding.weight,
                       model.item_embedding.weight], dim=0)
        layers = [E.cpu().numpy()]
        for _ in range(model.n_layers):
            E = torch.sparse.mm(adj, E)
            layers.append(E.cpu().numpy())

    item_layers = [l[n_users:] for l in layers]
    # P2-MultiHop: E_I_agg = mean of layers 0..K-1
    E_I_agg = np.mean(item_layers[:-1], axis=0)
    p("done.")

    # Degree maps
    user_degrees = {u: len(v) for u, v in dataset.train_dict.items()}
    item_degrees = {}
    for items in dataset.train_dict.values():
        for i in items:
            item_degrees[i] = item_degrees.get(i, 0) + 1

    # Sample users
    rng = np.random.RandomState(seed)
    test_users = [u for u in dataset.test_dict if dataset.test_dict[u]]
    rng.shuffle(test_users)
    sample_users = test_users[:n_sample]
    avg_deg = np.mean([user_degrees.get(u, 0) for u in sample_users])
    p(f"  Sampled {n_sample} users, avg degree={avg_deg:.1f}")

    # ObliPack for bucketed/dense tests
    p("  Building ObliPack...", end=" ")
    oblipack = ObliRec(dataset, bucketing="adaptive",
                        privacy_weight=10.0, seed=seed)
    p("done.")

    results = {}

    # ── (A) P2 depth-2, Sparse ──────────────────────────────────────────
    p(f"\n  [A] P2 depth-2 Sparse (n_poly={POLY_DEPTH2}, S={SLOTS_DEPTH2})...")
    p("      Creating CKKS context (depth-2)...", end=" ")
    ctx2 = make_ckks_context(POLY_DEPTH2, depth=2)
    p("done.")
    t_list, nc_list = [], []
    for i, uid in enumerate(sample_users):
        v = make_indicator(uid, dataset.train_dict, user_degrees, item_degrees, n_items)
        r = chunked_p2mh(ctx2, v, E_I_agg, SLOTS_DEPTH2, n_trials=n_trials)
        t_list.append(r["mean_ms"])
        nc_list.append(r["n_chunks"])
        p(f"      user {i+1}/{n_sample}: {r['mean_ms']:.1f}ms  ({r['n_chunks']}/{C2} chunks)")
    results["p2_depth2_sparse"] = {
        "mean_ms": float(np.mean(t_list)), "std_ms": float(np.std(t_list)),
        "avg_n_chunks": float(np.mean(nc_list)),
    }
    p(f"    → {np.mean(t_list):.1f} ± {np.std(t_list):.1f} ms")

    # ── (B) P2 depth-2, ObliRec bucketed ────────────────────────────────
    p(f"\n  [B] P2 depth-2 Bucketed...")
    t_list, nc_list = [], []
    for i, uid in enumerate(sample_users):
        padded = oblipack.user_padded_neighbors.get(uid) or list(dataset.train_dict.get(uid, []))
        real_items = set(dataset.train_dict.get(uid, []))
        d_pad = len(padded)
        v = np.zeros(n_items, dtype=np.float64)
        for iid in padded:
            d_i = item_degrees.get(iid, 1)
            w = 1.0 / math.sqrt(d_pad * d_i) if iid in real_items else 0.0
            if iid < n_items:
                v[iid] = w
        r = chunked_p2mh(ctx2, v, E_I_agg, SLOTS_DEPTH2, n_trials=n_trials)
        t_list.append(r["mean_ms"])
        nc_list.append(r["n_chunks"])
        p(f"      user {i+1}/{n_sample}: {r['mean_ms']:.1f}ms  ({r['n_chunks']}/{C2} chunks)")
    results["p2_depth2_bucketed"] = {
        "mean_ms": float(np.mean(t_list)), "std_ms": float(np.std(t_list)),
        "avg_n_chunks": float(np.mean(nc_list)),
    }
    p(f"    → {np.mean(t_list):.1f} ± {np.std(t_list):.1f} ms")

    del ctx2  # free memory

    # ── (C) P2-MH depth-1, Sparse ───────────────────────────────────────
    p(f"\n  [C] P2-MH depth-1 Sparse (n_poly={POLY_DEPTH1}, S={SLOTS_DEPTH1})...")
    p("      Creating CKKS context (depth-1)...", end=" ")
    ctx1 = make_ckks_context(POLY_DEPTH1, depth=1)
    p("done.")
    t_list, nc_list = [], []
    for i, uid in enumerate(sample_users):
        v = make_indicator(uid, dataset.train_dict, user_degrees, item_degrees, n_items)
        r = chunked_p2mh(ctx1, v, E_I_agg, SLOTS_DEPTH1, n_trials=n_trials)
        t_list.append(r["mean_ms"])
        nc_list.append(r["n_chunks"])
        p(f"      user {i+1}/{n_sample}: {r['mean_ms']:.1f}ms  ({r['n_chunks']}/{C1} chunks)")
    results["p2mh_depth1_sparse"] = {
        "mean_ms": float(np.mean(t_list)), "std_ms": float(np.std(t_list)),
        "avg_n_chunks": float(np.mean(nc_list)),
    }
    p(f"    → {np.mean(t_list):.1f} ± {np.std(t_list):.1f} ms")

    # ── (D) P2-MH depth-1, Bucketed ─────────────────────────────────────
    p(f"\n  [D] P2-MH depth-1 Bucketed...")
    t_list, nc_list = [], []
    for i, uid in enumerate(sample_users):
        padded = oblipack.user_padded_neighbors.get(uid) or list(dataset.train_dict.get(uid, []))
        real_items = set(dataset.train_dict.get(uid, []))
        d_pad = len(padded)
        v = np.zeros(n_items, dtype=np.float64)
        for iid in padded:
            d_i = item_degrees.get(iid, 1)
            w = 1.0 / math.sqrt(d_pad * d_i) if iid in real_items else 0.0
            if iid < n_items:
                v[iid] = w
        r = chunked_p2mh(ctx1, v, E_I_agg, SLOTS_DEPTH1, n_trials=n_trials)
        t_list.append(r["mean_ms"])
        nc_list.append(r["n_chunks"])
        p(f"      user {i+1}/{n_sample}: {r['mean_ms']:.1f}ms  ({r['n_chunks']}/{C1} chunks)")
    results["p2mh_depth1_bucketed"] = {
        "mean_ms": float(np.mean(t_list)), "std_ms": float(np.std(t_list)),
        "avg_n_chunks": float(np.mean(nc_list)),
    }
    p(f"    → {np.mean(t_list):.1f} ± {np.std(t_list):.1f} ms")

    # ── (E) P2-MH depth-1, Decoy k=1 ────────────────────────────────────
    p(f"\n  [E] P2-MH depth-1 Decoy k=1...")
    t_list, nc_list = [], []
    for i, uid in enumerate(sample_users):
        padded = oblipack.user_padded_neighbors.get(uid) or list(dataset.train_dict.get(uid, []))
        real_items = set(dataset.train_dict.get(uid, []))
        d_pad = len(padded)
        v = np.zeros(n_items, dtype=np.float64)
        for iid in padded:
            d_i = item_degrees.get(iid, 1)
            w = 1.0 / math.sqrt(d_pad * d_i) if iid in real_items else 0.0
            if iid < n_items:
                v[iid] = w
        chunk_starts = list(range(0, n_items, SLOTS_DEPTH1))
        nonzero = set(s for s in chunk_starts if np.any(v[s:s+SLOTS_DEPTH1] != 0))
        inactive = [s for s in chunk_starts if s not in nonzero]
        if inactive:
            v[inactive[0]] = 1e-10
        r = chunked_p2mh(ctx1, v, E_I_agg, SLOTS_DEPTH1, n_trials=n_trials)
        t_list.append(r["mean_ms"])
        nc_list.append(r["n_chunks"])
        p(f"      user {i+1}/{n_sample}: {r['mean_ms']:.1f}ms  ({r['n_chunks']}/{C1} chunks)")
    results["p2mh_depth1_decoy1"] = {
        "mean_ms": float(np.mean(t_list)), "std_ms": float(np.std(t_list)),
        "avg_n_chunks": float(np.mean(nc_list)),
    }
    p(f"    → {np.mean(t_list):.1f} ± {np.std(t_list):.1f} ms")

    # ── (F) P2-MH depth-1, Dense ────────────────────────────────────────
    p(f"\n  [F] P2-MH depth-1 Dense (all {C1} chunks)...")
    t_list = []
    for i, uid in enumerate(sample_users):
        padded = oblipack.user_padded_neighbors.get(uid) or list(dataset.train_dict.get(uid, []))
        real_items = set(dataset.train_dict.get(uid, []))
        d_pad = len(padded)
        v = np.zeros(n_items, dtype=np.float64)
        for iid in padded:
            d_i = item_degrees.get(iid, 1)
            w = 1.0 / math.sqrt(d_pad * d_i) if iid in real_items else 0.0
            if iid < n_items:
                v[iid] = w
        r = chunked_p2mh_padded(ctx1, v, E_I_agg, SLOTS_DEPTH1, n_trials=n_trials)
        t_list.append(r["mean_ms"])
        p(f"      user {i+1}/{n_sample}: {r['mean_ms']:.1f}ms  ({C1}/{C1} chunks)")
    results["p2mh_depth1_dense"] = {
        "mean_ms": float(np.mean(t_list)), "std_ms": float(np.std(t_list)),
        "avg_n_chunks": float(C1),
    }
    p(f"    → {np.mean(t_list):.1f} ± {np.std(t_list):.1f} ms")

    # ── Summary ──────────────────────────────────────────────────────────
    p(f"\n  {'='*65}")
    p(f"  Summary (per-user ms, mean ± std, {n_sample} users, {n_trials} trials):")
    p(f"  {'Protocol':<35} {'n_poly':>7} {'Chunks':>8} {'ms':>9}")
    p(f"  {'-'*63}")
    rows = [
        ("P2 (depth-2) Sparse",         POLY_DEPTH2, "p2_depth2_sparse"),
        ("P2 (depth-2) Bucketed",        POLY_DEPTH2, "p2_depth2_bucketed"),
        ("P2-MH (depth-1) Sparse",       POLY_DEPTH1, "p2mh_depth1_sparse"),
        ("P2-MH (depth-1) Bucketed",     POLY_DEPTH1, "p2mh_depth1_bucketed"),
        ("P2-MH (depth-1) Decoy k=1",    POLY_DEPTH1, "p2mh_depth1_decoy1"),
        ("P2-MH (depth-1) Dense",        POLY_DEPTH1, "p2mh_depth1_dense"),
    ]
    for label, poly, key in rows:
        r = results[key]
        p(f"  {label:<35} {poly:>7} {r['avg_n_chunks']:>8.1f} "
          f"{r['mean_ms']:>7.1f} ± {r['std_ms']:>5.1f}")

    results["dataset"] = dataset_name
    results["n_items"] = n_items
    results["C_depth1"] = C1
    results["C_depth2"] = C2
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="gowalla",
                        choices=list(DATASET_DEFAULTS.keys()))
    parser.add_argument("--n_sample", type=int, default=5)
    parser.add_argument("--n_trials", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    results = run(args.dataset, n_sample=args.n_sample,
                  seed=args.seed, n_trials=args.n_trials)

    out = args.out or f"results_he_timing_{args.dataset}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    p(f"\n  Saved to {out}")


if __name__ == "__main__":
    main()
