"""
Decoy mode end-to-end latency benchmark.

Measures HE inference latency at varying decoy budgets (k=0, 3, 5, dense)
to show the actual cost of the recommended decoy operating point.

Complements the per-mode latency table by reporting decoy-mode overhead
at intermediate budgets (k=1,3,5) between bucketed and dense extremes.

Usage:
    python benchmark_decoy_latency.py --dataset gowalla --gpu 0
    python benchmark_decoy_latency.py --dataset amazon-book --gpu 0
"""

import argparse
import json
import math
import os
import time
import numpy as np
import torch
import tenseal as ts

from src.dataset import RecDataset
from src.model import LightGCN
from src.he_inference import HERecommender
from src.oblirec import ObliRec


DATASET_DEFAULTS = {
    "ml-100k":     {"rating_threshold": 4},
    "ml-1m":       {"rating_threshold": 4},
    "gowalla":     {"rating_threshold": 0},
    "amazon-book": {"rating_threshold": 0},
}


def benchmark_chunked_with_decoys(he, indicator, target_n_chunks, n_decoys,
                                   matrix, n_items, n_trials=3, warmup=1,
                                   rng=None):
    """Benchmark chunked HE with ObliPack + decoy chunks.

    Returns latency breakdown and chunk statistics.
    """
    S = he._get_slot_capacity()
    d = matrix.shape[1]
    timings = {"encrypt": [], "server": [], "decrypt": [], "total": []}

    for trial in range(warmup + n_trials):
        # -- Encrypt: ObliRec-padded + decoy chunks --
        t0 = time.perf_counter()
        enc_chunks = he.sparse_chunked_encrypt_oblirec(
            indicator, target_n_chunks, n_decoys=n_decoys, rng=rng)
        t_enc = time.perf_counter() - t0

        # -- Server: chunked matmul + sum --
        t0 = time.perf_counter()
        enc_user_emb = he.sparse_chunked_server_compute(enc_chunks, matrix)
        t_srv = time.perf_counter() - t0

        # -- Client: decrypt --
        t0 = time.perf_counter()
        u_plain = np.array(enc_user_emb.decrypt()[:d])
        t_dec = time.perf_counter() - t0

        if trial >= warmup:
            timings["encrypt"].append(t_enc)
            timings["server"].append(t_srv)
            timings["decrypt"].append(t_dec)
            timings["total"].append(t_enc + t_srv + t_dec)

    # Communication
    upload_kb = sum(HERecommender.ciphertext_size_kb(enc)
                    for _, enc in enc_chunks)
    download_kb = HERecommender.ciphertext_size_kb(enc_user_emb)

    return {
        "encrypt_ms": np.mean(timings["encrypt"]) * 1000,
        "server_ms": np.mean(timings["server"]) * 1000,
        "decrypt_ms": np.mean(timings["decrypt"]) * 1000,
        "total_ms": np.mean(timings["total"]) * 1000,
        "total_std_ms": np.std(timings["total"]) * 1000,
        "upload_kb": upload_kb,
        "download_kb": download_kb,
        "n_chunks_submitted": len(enc_chunks),
    }


def benchmark_sparse_no_oblipack(he, indicator, matrix, n_items,
                                  n_trials=3, warmup=1):
    """Benchmark sparse chunked HE without ObliPack (baseline)."""
    S = he._get_slot_capacity()
    d = matrix.shape[1]
    timings = {"encrypt": [], "server": [], "decrypt": [], "total": []}

    for trial in range(warmup + n_trials):
        t0 = time.perf_counter()
        enc_chunks = he.sparse_chunked_encrypt(indicator)
        t_enc = time.perf_counter() - t0

        t0 = time.perf_counter()
        enc_user_emb = he.sparse_chunked_server_compute(enc_chunks, matrix)
        t_srv = time.perf_counter() - t0

        t0 = time.perf_counter()
        u_plain = np.array(enc_user_emb.decrypt()[:d])
        t_dec = time.perf_counter() - t0

        if trial >= warmup:
            timings["encrypt"].append(t_enc)
            timings["server"].append(t_srv)
            timings["decrypt"].append(t_dec)
            timings["total"].append(t_enc + t_srv + t_dec)

    upload_kb = sum(HERecommender.ciphertext_size_kb(enc)
                    for _, enc in enc_chunks)
    download_kb = HERecommender.ciphertext_size_kb(enc_user_emb)

    return {
        "encrypt_ms": np.mean(timings["encrypt"]) * 1000,
        "server_ms": np.mean(timings["server"]) * 1000,
        "decrypt_ms": np.mean(timings["decrypt"]) * 1000,
        "total_ms": np.mean(timings["total"]) * 1000,
        "total_std_ms": np.std(timings["total"]) * 1000,
        "upload_kb": upload_kb,
        "download_kb": download_kb,
        "n_chunks_submitted": len(enc_chunks),
    }


def main():
    parser = argparse.ArgumentParser(description="Decoy latency benchmark")
    parser.add_argument("--dataset", type=str, default="gowalla",
                        choices=list(DATASET_DEFAULTS.keys()))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_users_sample", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ds = args.dataset
    defaults = DATASET_DEFAULTS[ds]

    print(f"\n{'='*65}")
    print(f"  Decoy Latency Benchmark — {ds}")
    print(f"{'='*65}")

    # Load dataset
    dataset = RecDataset(f"data/{ds}", dataset_name=ds,
                         rating_threshold=defaults["rating_threshold"],
                         seed=args.seed)

    # Load model
    model_path = f"models/{ds}_best.pt"
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available()
                          else "cpu")
    ckpt = torch.load(model_path, weights_only=False, map_location=device)

    model = LightGCN(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        emb_dim=ckpt["args"]["emb_dim"],
        n_layers=ckpt["args"]["n_layers"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    adj = dataset.adj_matrix.to(device)

    # Get per-layer embeddings
    with torch.no_grad():
        E = torch.cat([model.user_embedding.weight,
                       model.item_embedding.weight], dim=0)
        layers = [E.cpu().numpy()]
        for _ in range(model.n_layers):
            E = torch.sparse.mm(adj, E)
            layers.append(E.cpu().numpy())

    n_users = dataset.n_users
    n_items = dataset.n_items
    item_emb_final = np.mean([l[n_users:] for l in layers], axis=0)
    item_layers = [l[n_users:] for l in layers]

    # Setup HE (depth 1, N=8192 for P2-MH)
    user_emb_final = np.mean([l[:n_users] for l in layers], axis=0)
    he = HERecommender(user_emb_final, item_emb_final, item_layers=item_layers)
    he.setup_context(poly_modulus_degree=8192, depth=1)

    S = he._get_slot_capacity()
    C = he.get_n_chunks()  # total chunks
    print(f"  Items: {n_items}, Slot capacity: {S}, Total chunks C={C}")

    if C <= 1:
        print(f"  [!] Only 1 chunk — decoy mode has no effect. Exiting.")
        return

    # Build ObliPack
    oblipack = ObliRec(dataset)

    # Build degree maps
    user_degrees = {u: len(items) for u, items in dataset.train_dict.items()}
    item_degrees = {}
    for u, items in dataset.train_dict.items():
        for i in items:
            item_degrees[i] = item_degrees.get(i, 0) + 1

    # Compute bucket chunk targets
    print("  Computing per-bucket chunk targets...")
    bucket_targets = he.compute_bucket_chunk_targets(
        oblipack, user_degrees, item_degrees)

    # Sample users across different buckets
    rng = np.random.RandomState(args.seed)
    test_users = [u for u in dataset.test_dict if dataset.test_dict[u]]
    rng.shuffle(test_users)
    sample_users = test_users[:args.n_users_sample]

    # E_I_agg for P2-MH
    E_agg = he.E_I_agg if he.E_I_agg is not None else item_emb_final

    # Decoy budgets to test
    decoy_budgets = [0, 1, 3, 5]
    # Add dense mode (all remaining chunks)
    max_target = max(bucket_targets.values())
    dense_k = C - max_target
    if dense_k > 5:
        decoy_budgets.append(dense_k)
    # Ensure unique and sorted
    decoy_budgets = sorted(set(decoy_budgets))

    print(f"  Decoy budgets to test: {decoy_budgets}")
    print(f"  Dense mode requires k={dense_k} (C={C}, max target={max_target})")
    print(f"  Sample users: {len(sample_users)}")

    results = {"dataset": ds, "C": C, "slot_capacity": S,
               "n_users_sample": len(sample_users), "modes": {}}

    # ── Sparse (no ObliPack) baseline ────────────────────────────────
    print(f"\n  [Sparse] No ObliPack (baseline)...")
    sparse_timings = []
    sparse_chunks_list = []
    for uid in sample_users:
        indicator = he.make_indicator(uid, dataset.train_dict,
                                       user_degrees, item_degrees)
        t = benchmark_sparse_no_oblipack(he, indicator, E_agg, n_items,
                                          n_trials=3, warmup=1)
        sparse_timings.append(t)
        sparse_chunks_list.append(t["n_chunks_submitted"])

    sparse_avg = {k: np.mean([t[k] for t in sparse_timings])
                  for k in sparse_timings[0]}
    results["modes"]["sparse_no_oblipack"] = {
        "total_ms": sparse_avg["total_ms"],
        "encrypt_ms": sparse_avg["encrypt_ms"],
        "server_ms": sparse_avg["server_ms"],
        "decrypt_ms": sparse_avg["decrypt_ms"],
        "upload_kb": sparse_avg["upload_kb"],
        "avg_chunks": np.mean(sparse_chunks_list),
        "label": "Sparse (no ObliPack)",
    }
    print(f"    Total: {sparse_avg['total_ms']:.1f}ms | "
          f"Server: {sparse_avg['server_ms']:.1f}ms | "
          f"Chunks: {np.mean(sparse_chunks_list):.1f}")

    # ── ObliPack with varying decoy budgets ──────────────────────────
    for k in decoy_budgets:
        mode_name = f"decoy_k{k}"
        if k == 0:
            label = "ObliPack sparse (k=0)"
        elif k == dense_k:
            label = f"ObliPack dense (k={k})"
        else:
            label = f"ObliPack decoy (k={k})"

        print(f"\n  [{label}]...")
        mode_timings = []
        mode_chunks_list = []

        for uid in sample_users:
            indicator = he.make_indicator(uid, dataset.train_dict,
                                           user_degrees, item_degrees)
            deg = user_degrees[uid]
            bid = oblipack.bucket_id(deg)
            target = bucket_targets.get(bid, C)

            t = benchmark_chunked_with_decoys(
                he, indicator, target, n_decoys=k, matrix=E_agg,
                n_items=n_items, n_trials=3, warmup=1,
                rng=np.random.RandomState(args.seed + uid))
            mode_timings.append(t)
            mode_chunks_list.append(t["n_chunks_submitted"])

        mode_avg = {mk: np.mean([t[mk] for t in mode_timings])
                    for mk in mode_timings[0]}
        overhead_vs_sparse = (mode_avg["total_ms"] / sparse_avg["total_ms"] - 1) * 100

        results["modes"][mode_name] = {
            "total_ms": mode_avg["total_ms"],
            "encrypt_ms": mode_avg["encrypt_ms"],
            "server_ms": mode_avg["server_ms"],
            "decrypt_ms": mode_avg["decrypt_ms"],
            "upload_kb": mode_avg["upload_kb"],
            "avg_chunks": np.mean(mode_chunks_list),
            "overhead_vs_sparse_pct": overhead_vs_sparse,
            "label": label,
        }
        print(f"    Total: {mode_avg['total_ms']:.1f}ms | "
              f"Server: {mode_avg['server_ms']:.1f}ms | "
              f"Chunks: {np.mean(mode_chunks_list):.1f} | "
              f"Overhead: +{overhead_vs_sparse:.1f}%")

    # ── Summary table ────────────────────────────────────────────────
    print(f"\n  {'='*75}")
    print(f"  DECOY LATENCY SUMMARY — {ds} (P2-MH, depth 1, N=8192)")
    print(f"  {'='*75}")
    print(f"  {'Mode':<30} {'Total(ms)':>10} {'Server(ms)':>12} "
          f"{'Chunks':>8} {'Overhead':>10}")
    print(f"  {'-'*75}")

    for mode_key in ["sparse_no_oblipack"] + [f"decoy_k{k}" for k in decoy_budgets]:
        m = results["modes"][mode_key]
        ovh = f"+{m.get('overhead_vs_sparse_pct', 0):.1f}%" if mode_key != "sparse_no_oblipack" else "baseline"
        print(f"  {m['label']:<30} {m['total_ms']:>10.1f} {m['server_ms']:>12.1f} "
              f"{m['avg_chunks']:>8.1f} {ovh:>10}")

    # Save
    out_path = f"results_decoy_latency_{ds}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
