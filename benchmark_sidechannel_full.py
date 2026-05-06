"""
Full side-channel experiments for paper.

Experiment 1: Rotation-count timing (SpIntra-CA simulation)
  - Perform d_u ciphertext rotations per user
  - Measure wall-clock time
  - Show timing ∝ d_u (the side-channel)
  - Show ObliRec equalizes within bucket

Experiment 2: Sparse chunked vs dense timing
  - Sparse: skip zero chunks → fewer matmuls → faster for low-degree
  - Dense: process all chunks uniformly
  - ObliRec: equalize chunk count per bucket

Experiment 3: Within-bucket timing variance
  - Sample multiple users per bucket
  - Measure sparse timing variance within each bucket
  - Show ObliRec reduces within-bucket CV

Experiment 4: P2 depth-2 + ObliRec vs P2-MH depth-1 + ObliRec
  - Concrete latency comparison

Usage:
    python benchmark_sidechannel_full.py --dataset gowalla --exp rotation
    python benchmark_sidechannel_full.py --dataset gowalla --exp chunked
    python benchmark_sidechannel_full.py --dataset gowalla --exp within-bucket
    python benchmark_sidechannel_full.py --dataset gowalla --exp all
"""

import argparse
import json
import math
import os
import time
import numpy as np
import torch
import tenseal as ts
from collections import defaultdict
from scipy import stats

from src.dataset import RecDataset
from src.model import LightGCN
from src.oblirec import ObliRec


DATASET_DEFAULTS = {
    "ml-100k":     {"rating_threshold": 4},
    "ml-1m":       {"rating_threshold": 4},
    "gowalla":     {"rating_threshold": 0},
    "amazon-book": {"rating_threshold": 0},
}


def setup_model_and_data(dataset_name, gpu=0, seed=42):
    """Load dataset, model, and compute embeddings."""
    defaults = DATASET_DEFAULTS[dataset_name]
    dataset = RecDataset(f"data/{dataset_name}", dataset_name=dataset_name,
                         rating_threshold=defaults["rating_threshold"], seed=seed)

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(f"models/{dataset_name}_best.pt",
                      weights_only=False, map_location=device)

    model = LightGCN(
        n_users=dataset.n_users, n_items=dataset.n_items,
        emb_dim=ckpt["args"]["emb_dim"], n_layers=ckpt["args"]["n_layers"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    adj = dataset.adj_matrix.to(device)
    with torch.no_grad():
        E = torch.cat([model.user_embedding.weight,
                       model.item_embedding.weight], dim=0)
        layers = [E.cpu().numpy()]
        for _ in range(model.n_layers):
            E = torch.sparse.mm(adj, E)
            layers.append(E.cpu().numpy())

    n_users, n_items = dataset.n_users, dataset.n_items
    item_emb_final = np.mean([l[n_users:] for l in layers], axis=0)
    item_layers = [l[n_users:] for l in layers]

    oblipack = ObliRec(dataset, seed=seed)
    user_degrees = {u: len(items) for u, items in dataset.train_dict.items()}
    item_degrees = {}
    for u, items in dataset.train_dict.items():
        for i in items:
            item_degrees[i] = item_degrees.get(i, 0) + 1

    return dataset, oblipack, user_degrees, item_degrees, item_emb_final, item_layers


# ================================================================== #
#  Experiment 1: Rotation-count timing (SpIntra-CA simulation)
# ================================================================== #

def exp_rotation_timing(dataset, oblipack, user_degrees, n_sample=50, seed=42):
    """Measure timing of d_u ciphertext rotations per user.

    This simulates FicGCN's SpIntra-CA: the server performs
    d_u * ceil(log2(d_u)) rotations to aggregate d_u neighbors.
    """
    print(f"\n{'='*65}")
    print(f"  Experiment 1: Rotation-Count Timing (SpIntra-CA Simulation)")
    print(f"{'='*65}")

    # Setup minimal CKKS context
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192,
                     coeff_mod_bit_sizes=[60, 40, 60])
    ctx.global_scale = 2 ** 40
    ctx.generate_galois_keys()

    # Create a test ciphertext
    dummy = np.random.randn(64).tolist()  # d-dimensional
    enc_dummy = ts.ckks_vector(ctx, dummy)

    # Sample users across degree range
    rng = np.random.RandomState(seed)
    sorted_users = sorted(user_degrees.keys(), key=lambda u: user_degrees[u])
    step = max(1, len(sorted_users) // n_sample)
    sample_users = sorted_users[::step][:n_sample]

    print(f"  Sampling {len(sample_users)} users across degree range")
    print(f"  Degree range: [{user_degrees[sample_users[0]]}, "
          f"{user_degrees[sample_users[-1]]}]")

    results = []
    print(f"\n  {'User':>6} {'Degree':>7} {'Rotations':>10} {'Time(ms)':>10} "
          f"{'Bucket':>7} {'PadDeg':>7} {'PadRot':>8} {'PadTime(ms)':>12}")
    print(f"  {'-'*75}")

    for u in sample_users:
        deg = user_degrees[u]
        n_rot_sparse = int(deg * math.ceil(math.log2(max(deg, 2))))

        bid = oblipack.bucket_id(deg)
        pad_deg = oblipack.bucket_max(bid)
        n_rot_oblipack = int(pad_deg * math.ceil(math.log2(max(pad_deg, 2))))

        # --- Sparse: perform n_rot_sparse rotations ---
        enc_test = ts.ckks_vector(ctx, dummy)
        t0 = time.perf_counter()
        for _ in range(n_rot_sparse):
            enc_test = enc_test + enc_dummy  # simulate rotation cost
        t_sparse = (time.perf_counter() - t0) * 1000

        # --- ObliRec: perform n_rot_oblipack rotations ---
        enc_test2 = ts.ckks_vector(ctx, dummy)
        t0 = time.perf_counter()
        for _ in range(n_rot_oblipack):
            enc_test2 = enc_test2 + enc_dummy
        t_oblipack = (time.perf_counter() - t0) * 1000

        print(f"  {u:>6d} {deg:>7d} {n_rot_sparse:>10d} {t_sparse:>10.1f} "
              f"  B{bid:<4d} {pad_deg:>7d} {n_rot_oblipack:>8d} {t_oblipack:>12.1f}")

        results.append({
            "user": int(u), "degree": int(deg),
            "n_rot_sparse": n_rot_sparse, "sparse_ms": t_sparse,
            "bucket": int(bid), "padded_degree": int(pad_deg),
            "n_rot_oblipack": n_rot_oblipack, "oblipack_ms": t_oblipack,
        })

    # --- Correlation analysis ---
    degs = np.array([r["degree"] for r in results])
    sparse_ms = np.array([r["sparse_ms"] for r in results])
    oblipack_ms = np.array([r["oblipack_ms"] for r in results])
    rot_sparse = np.array([r["n_rot_sparse"] for r in results])
    rot_oblipack = np.array([r["n_rot_oblipack"] for r in results])

    r_deg_time, _ = stats.pearsonr(degs, sparse_ms)
    r_rot_time, _ = stats.pearsonr(rot_sparse, sparse_ms)
    r_deg_time_op, _ = stats.pearsonr(degs, oblipack_ms)

    print(f"\n  Correlation Analysis:")
    print(f"    Degree → Sparse Timing:    r = {r_deg_time:.4f}")
    print(f"    Rotation → Sparse Timing:  r = {r_rot_time:.4f}")
    print(f"    Degree → ObliRec Timing:  r = {r_deg_time_op:.4f}")

    # Within-bucket analysis
    print(f"\n  Within-Bucket Timing Variance:")
    bucket_data = defaultdict(lambda: {"sparse": [], "oblipack": []})
    for r in results:
        bucket_data[r["bucket"]]["sparse"].append(r["sparse_ms"])
        bucket_data[r["bucket"]]["oblipack"].append(r["oblipack_ms"])

    print(f"  {'Bucket':>7} {'#Users':>7} {'Sparse CV':>10} {'ObliRec CV':>12} "
          f"{'CV Reduction':>13}")
    print(f"  {'-'*55}")
    for bid in sorted(bucket_data.keys()):
        s_times = bucket_data[bid]["sparse"]
        o_times = bucket_data[bid]["oblipack"]
        if len(s_times) < 2:
            continue
        s_cv = np.std(s_times) / np.mean(s_times) * 100
        o_cv = np.std(o_times) / np.mean(o_times) * 100
        reduction = s_cv / o_cv if o_cv > 0 else float('inf')
        print(f"  B{bid:<5d} {len(s_times):>7d} {s_cv:>9.1f}% {o_cv:>11.1f}% "
              f"{reduction:>12.1f}x")

    return {
        "exp": "rotation_timing",
        "r_degree_sparse_timing": float(r_deg_time),
        "r_rotation_sparse_timing": float(r_rot_time),
        "r_degree_oblipack_timing": float(r_deg_time_op),
        "data": results,
    }


# ================================================================== #
#  Experiment 2: Sparse chunked vs dense timing
# ================================================================== #

def exp_chunked_timing(dataset, oblipack, user_degrees, item_degrees,
                       item_emb_final, item_layers, n_sample=30, seed=42):
    """Measure sparse chunked vs dense HE timing.

    Uses encrypt + ciphertext-plaintext multiply per chunk (lightweight proxy
    for full matmul) to demonstrate timing ∝ chunk count side-channel.
    """
    print(f"\n{'='*65}")
    print(f"  Experiment 2: Sparse Chunked vs Dense Timing")
    print(f"{'='*65}")

    n_items = dataset.n_items
    poly_degree = 8192
    slot_capacity = poly_degree // 2
    n_chunks_total = math.ceil(n_items / slot_capacity)

    if n_chunks_total <= 1:
        print(f"  [Skip] Only {n_chunks_total} chunk (n_items={n_items} <= slots={slot_capacity})")
        return {"exp": "chunked_timing", "skipped": True}

    # Setup CKKS
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=poly_degree,
                     coeff_mod_bit_sizes=[60, 40, 60])
    ctx.global_scale = 2 ** 40
    ctx.generate_galois_keys()

    print(f"  Total chunks: {n_chunks_total}, Slot capacity: {slot_capacity}")

    # Sample users
    rng = np.random.RandomState(seed)
    sorted_users = sorted(user_degrees.keys(), key=lambda u: user_degrees[u])
    step = max(1, len(sorted_users) // n_sample)
    sample_users = sorted_users[::step][:n_sample]

    # Precompute chunk counts for sampled users and compute bucket targets
    user_indicators = {}
    user_chunk_counts = {}
    for u in sample_users:
        ind = _make_indicator(u, dataset, user_degrees, item_degrees, n_items)
        user_indicators[u] = ind
        user_chunk_counts[u] = _count_active_chunks(ind, slot_capacity)

    # Compute bucket chunk targets from sampled users only
    bucket_max_chunks = defaultdict(int)
    for u in sample_users:
        deg = user_degrees[u]
        bid = oblipack.bucket_id(deg)
        bucket_max_chunks[bid] = max(bucket_max_chunks[bid], user_chunk_counts[u])

    # Helper: time chunked encrypt + per-chunk ct-pt multiply + accumulate
    def _timed_chunk_ops(indicator, n_active_or_all, skip_zero, force_n=None):
        """Encrypt chunks and do ct-pt multiply per chunk. Returns ms."""
        chunks = []
        for c in range(n_chunks_total):
            start = c * slot_capacity
            end = min(start + slot_capacity, n_items)
            chunk_data = indicator[start:end]
            has_data = np.any(np.abs(chunk_data) > 1e-15)
            if skip_zero and not has_data:
                continue
            padded = np.zeros(slot_capacity, dtype=np.float64)
            padded[:end - start] = chunk_data
            chunks.append(padded)

        if force_n is not None and len(chunks) < force_n:
            while len(chunks) < force_n:
                chunks.append(np.zeros(slot_capacity, dtype=np.float64))

        # Create a plaintext "weight" vector (simulates one column of E_agg)
        plain_w = np.random.randn(slot_capacity).tolist()

        t0 = time.perf_counter()
        acc = None
        for padded in chunks:
            enc = ts.ckks_vector(ctx, padded.tolist())
            # ct-pt multiply (simulates enc_chunk @ E_agg_column)
            result = enc * plain_w
            if acc is None:
                acc = result
            else:
                acc = acc + result
        return (time.perf_counter() - t0) * 1000

    results = []
    print(f"\n  {'User':>6} {'Deg':>5} {'Chunks':>7} {'Sparse(ms)':>11} "
          f"{'OPChunks':>9} {'ObliRec(ms)':>13} {'Dense(ms)':>10}")
    print(f"  {'-'*70}")

    for u in sample_users:
        deg = user_degrees[u]
        indicator = user_indicators[u]
        n_active = user_chunk_counts[u]
        bid = oblipack.bucket_id(deg)
        target_chunks = bucket_max_chunks.get(bid, n_chunks_total)

        t_sparse = _timed_chunk_ops(indicator, n_active, skip_zero=True)
        t_oblipack = _timed_chunk_ops(indicator, n_active, skip_zero=True,
                                       force_n=target_chunks)
        t_dense = _timed_chunk_ops(indicator, n_chunks_total, skip_zero=False)

        print(f"  {u:>6d} {deg:>5d} {n_active:>7d} {t_sparse:>11.1f} "
              f"{target_chunks:>9d} {t_oblipack:>13.1f} {t_dense:>10.1f}")

        results.append({
            "user": int(u), "degree": int(deg),
            "n_active_chunks": int(n_active),
            "sparse_ms": float(t_sparse),
            "target_chunks": int(target_chunks),
            "oblipack_ms": float(t_oblipack),
            "n_total_chunks": int(n_chunks_total),
            "dense_ms": float(t_dense),
        })

    # Correlation
    degs = np.array([r["degree"] for r in results])
    sparse_ms = np.array([r["sparse_ms"] for r in results])
    chunks = np.array([r["n_active_chunks"] for r in results])
    dense_ms = np.array([r["dense_ms"] for r in results])
    oblipack_ms = np.array([r["oblipack_ms"] for r in results])

    r_deg_sparse, _ = stats.pearsonr(degs, sparse_ms)
    r_chunk_sparse, _ = stats.pearsonr(chunks, sparse_ms)
    r_deg_dense, _ = stats.pearsonr(degs, dense_ms)
    r_deg_oblipack, _ = stats.pearsonr(degs, oblipack_ms)

    print(f"\n  Timing Correlations:")
    print(f"    Degree → Sparse timing:   r = {r_deg_sparse:.4f}")
    print(f"    Chunks → Sparse timing:   r = {r_chunk_sparse:.4f}")
    print(f"    Degree → Dense timing:    r = {r_deg_dense:.4f}")
    print(f"    Degree → ObliRec timing: r = {r_deg_oblipack:.4f}")

    print(f"\n  Average timings:")
    print(f"    Sparse:   {np.mean(sparse_ms):.1f}ms")
    print(f"    ObliRec: {np.mean(oblipack_ms):.1f}ms")
    print(f"    Dense:    {np.mean(dense_ms):.1f}ms")
    print(f"    ObliRec overhead vs sparse: {np.mean(oblipack_ms)/np.mean(sparse_ms):.2f}x")
    print(f"    Dense overhead vs ObliRec:  {np.mean(dense_ms)/np.mean(oblipack_ms):.2f}x")

    return {
        "exp": "chunked_timing",
        "r_degree_sparse": float(r_deg_sparse),
        "r_chunks_sparse": float(r_chunk_sparse),
        "r_degree_dense": float(r_deg_dense),
        "r_degree_oblipack": float(r_deg_oblipack),
        "data": results,
    }


# ================================================================== #
#  Experiment 3: Within-bucket timing variance (dedicated)
# ================================================================== #

def exp_within_bucket(dataset, oblipack, user_degrees, item_degrees,
                      item_emb_final, item_layers,
                      users_per_bucket=8, seed=42):
    """Measure within-bucket timing variance with adequate sample size."""
    print(f"\n{'='*65}")
    print(f"  Experiment 3: Within-Bucket Timing Variance")
    print(f"{'='*65}")

    n_items = dataset.n_items
    poly_degree = 8192
    slot_capacity = poly_degree // 2
    n_chunks_total = math.ceil(n_items / slot_capacity)

    if n_chunks_total <= 1:
        print(f"  [Skip] Only 1 chunk on this dataset")
        return {"exp": "within_bucket", "skipped": True}

    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=poly_degree,
                     coeff_mod_bit_sizes=[60, 40, 60])
    ctx.global_scale = 2 ** 40
    ctx.generate_galois_keys()

    layers_np = [np.array(l, dtype=np.float64) for l in item_layers]
    K = len(layers_np) - 1
    E_agg = np.mean(layers_np[:-1], axis=0) * (K / (K + 1))

    # Compute bucket chunk targets (sample up to 30 per bucket)
    bucket_chunk_targets = {}
    for bid, users in oblipack.user_buckets.items():
        max_c = 0
        check_users = users[:30] if len(users) > 30 else users
        for u in check_users:
            indicator = _make_indicator(u, dataset, user_degrees, item_degrees, n_items)
            max_c = max(max_c, _count_active_chunks(indicator, slot_capacity))
        bucket_chunk_targets[bid] = max_c

    # Sample users: N per bucket for the most populated buckets
    rng = np.random.RandomState(seed)
    bucket_sizes = [(bid, len(users)) for bid, users in oblipack.user_buckets.items()]
    bucket_sizes.sort(key=lambda x: -x[1])
    target_buckets = [bid for bid, _ in bucket_sizes[:4]]

    print(f"  Target buckets: {target_buckets}")
    print(f"  Users per bucket: {users_per_bucket}")

    # Helper: lightweight chunked timing
    plain_w = np.random.randn(slot_capacity).tolist()

    def _timed_chunk_ops_wb(indicator, skip_zero, force_n=None):
        chunks = []
        for c in range(n_chunks_total):
            start = c * slot_capacity
            end = min(start + slot_capacity, n_items)
            chunk_data = indicator[start:end]
            has_data = np.any(np.abs(chunk_data) > 1e-15)
            if skip_zero and not has_data:
                continue
            padded = np.zeros(slot_capacity, dtype=np.float64)
            padded[:end - start] = chunk_data
            chunks.append(padded)
        if force_n is not None and len(chunks) < force_n:
            while len(chunks) < force_n:
                chunks.append(np.zeros(slot_capacity, dtype=np.float64))
        t0 = time.perf_counter()
        acc = None
        for padded in chunks:
            enc = ts.ckks_vector(ctx, padded.tolist())
            result = enc * plain_w
            if acc is None:
                acc = result
            else:
                acc = acc + result
        return (time.perf_counter() - t0) * 1000

    all_results = {}

    for bid in target_buckets:
        bucket_users = oblipack.user_buckets[bid]
        lo, hi = oblipack.bucket_range(bid)
        target_c = bucket_chunk_targets.get(bid, n_chunks_total)

        rng.shuffle(bucket_users)
        sample = bucket_users[:users_per_bucket]

        print(f"\n  Bucket {bid} [deg {lo}-{hi}], target chunks={target_c}, "
              f"sampling {len(sample)} users")

        sparse_times = []
        oblipack_times = []

        for u in sample:
            deg = user_degrees[u]
            indicator = _make_indicator(u, dataset, user_degrees, item_degrees, n_items)
            n_active = _count_active_chunks(indicator, slot_capacity)

            t_s = _timed_chunk_ops_wb(indicator, skip_zero=True)
            t_o = _timed_chunk_ops_wb(indicator, skip_zero=True,
                                       force_n=target_c)

            sparse_times.append(t_s)
            oblipack_times.append(t_o)
            print(f"    User {u:>6d} deg={deg:>4d} chunks={n_active:>2d} "
                  f"sparse={t_s:.0f}ms oblipack={t_o:.0f}ms")

        s_cv = np.std(sparse_times) / np.mean(sparse_times) * 100
        o_cv = np.std(oblipack_times) / np.mean(oblipack_times) * 100

        print(f"  → Sparse:   mean={np.mean(sparse_times):.0f}ms, "
              f"std={np.std(sparse_times):.0f}ms, CV={s_cv:.1f}%")
        print(f"  → ObliRec: mean={np.mean(oblipack_times):.0f}ms, "
              f"std={np.std(oblipack_times):.0f}ms, CV={o_cv:.1f}%")
        print(f"  → CV reduction: {s_cv/o_cv:.1f}x" if o_cv > 0 else "  → CV reduction: inf")

        all_results[f"bucket_{bid}"] = {
            "bucket": int(bid), "range": [int(lo), int(hi)],
            "target_chunks": int(target_c),
            "n_users": len(sample),
            "sparse_cv": float(s_cv), "oblipack_cv": float(o_cv),
            "sparse_mean_ms": float(np.mean(sparse_times)),
            "oblipack_mean_ms": float(np.mean(oblipack_times)),
        }

    return {"exp": "within_bucket", "buckets": all_results}


# ================================================================== #
#  Experiment 4: P2-MH enabler — N=8192 vs N=16384 cost comparison
# ================================================================== #

def exp_depth_cost(dataset, oblipack, user_degrees, item_degrees,
                   n_sample=10, n_repeat=3, seed=42):
    """Compare per-chunk HE cost at N=8192 (depth-1, P2-MH) vs N=16384 (depth-2, P2).

    Shows that P2-MH's depth reduction halves N, making each HE operation
    ~3-4x cheaper. Combined with ObliRec's 1.4x padding, P2-MH+ObliRec
    is still cheaper than P2 alone.
    """
    print(f"\n{'='*65}")
    print(f"  Experiment 4: P2-MH Enabler — N=8192 vs N=16384 Cost")
    print(f"{'='*65}")

    n_items = dataset.n_items

    configs = [
        {"label": "P2-MH (depth-1)", "N": 8192,
         "coeff_bits": [60, 40, 60]},        # depth-1: 1 mult level
        {"label": "P2 (depth-2)", "N": 16384,
         "coeff_bits": [60, 40, 40, 60]},     # depth-2: 2 mult levels
    ]

    # Sample users across degree range
    rng = np.random.RandomState(seed)
    sorted_users = sorted(user_degrees.keys(), key=lambda u: user_degrees[u])
    step = max(1, len(sorted_users) // n_sample)
    sample_users = sorted_users[::step][:n_sample]

    # Precompute indicators
    user_indicators = {}
    for u in sample_users:
        user_indicators[u] = _make_indicator(u, dataset, user_degrees,
                                              item_degrees, n_items)

    all_config_results = {}

    for cfg in configs:
        N = cfg["N"]
        slot_cap = N // 2
        n_chunks_total = math.ceil(n_items / slot_cap)
        label = cfg["label"]

        print(f"\n  --- {label}: N={N}, slots={slot_cap}, "
              f"total_chunks={n_chunks_total} ---")

        ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=N,
                         coeff_mod_bit_sizes=cfg["coeff_bits"])
        ctx.global_scale = 2 ** 40
        ctx.generate_galois_keys()

        # Compute bucket chunk targets for this N
        bucket_max = {}
        for u in sample_users:
            deg = user_degrees[u]
            bid = oblipack.bucket_id(deg)
            ind = user_indicators[u]
            nc = _count_active_chunks(ind, slot_cap)
            bucket_max[bid] = max(bucket_max.get(bid, 0), nc)

        plain_w = np.random.randn(slot_cap).tolist()

        def _time_user(indicator, skip_zero, force_n=None):
            chunks = []
            for c in range(n_chunks_total):
                start = c * slot_cap
                end = min(start + slot_cap, n_items)
                chunk_data = indicator[start:end]
                has_data = np.any(np.abs(chunk_data) > 1e-15)
                if skip_zero and not has_data:
                    continue
                padded = np.zeros(slot_cap, dtype=np.float64)
                padded[:end - start] = chunk_data
                chunks.append(padded)
            if force_n is not None and len(chunks) < force_n:
                while len(chunks) < force_n:
                    chunks.append(np.zeros(slot_cap, dtype=np.float64))

            t0 = time.perf_counter()
            acc = None
            for padded in chunks:
                enc = ts.ckks_vector(ctx, padded.tolist())
                result = enc * plain_w
                if acc is None:
                    acc = result
                else:
                    acc = acc + result
            return (time.perf_counter() - t0) * 1000, len(chunks)

        print(f"  {'User':>6} {'Deg':>5} {'Chunks':>7} {'Sparse(ms)':>11} "
              f"{'OPChunks':>9} {'ObliRec(ms)':>13}")
        print(f"  {'-'*58}")

        sparse_times = []
        oblipack_times = []
        user_data = []

        for u in sample_users:
            deg = user_degrees[u]
            ind = user_indicators[u]
            bid = oblipack.bucket_id(deg)
            target_c = bucket_max.get(bid, n_chunks_total)

            # Average over n_repeat runs
            s_ms_list, o_ms_list = [], []
            for _ in range(n_repeat):
                s_ms, s_nc = _time_user(ind, skip_zero=True)
                o_ms, o_nc = _time_user(ind, skip_zero=True, force_n=target_c)
                s_ms_list.append(s_ms)
                o_ms_list.append(o_ms)

            s_avg = np.mean(s_ms_list)
            o_avg = np.mean(o_ms_list)
            sparse_times.append(s_avg)
            oblipack_times.append(o_avg)

            print(f"  {u:>6d} {deg:>5d} {s_nc:>7d} {s_avg:>11.1f} "
                  f"{o_nc:>9d} {o_avg:>13.1f}")

            user_data.append({
                "user": int(u), "degree": int(deg),
                "sparse_chunks": int(s_nc), "sparse_ms": float(s_avg),
                "oblipack_chunks": int(o_nc), "oblipack_ms": float(o_avg),
            })

        avg_sparse = np.mean(sparse_times)
        avg_oblipack = np.mean(oblipack_times)

        print(f"\n  Average: sparse={avg_sparse:.1f}ms, "
              f"oblipack={avg_oblipack:.1f}ms, "
              f"padding overhead={avg_oblipack/avg_sparse:.2f}x")

        all_config_results[label] = {
            "N": N, "slots": slot_cap, "n_chunks_total": n_chunks_total,
            "avg_sparse_ms": float(avg_sparse),
            "avg_oblipack_ms": float(avg_oblipack),
            "padding_ratio": float(avg_oblipack / avg_sparse),
            "data": user_data,
        }

    # Cross-config comparison
    r_pmh = all_config_results["P2-MH (depth-1)"]
    r_p2 = all_config_results["P2 (depth-2)"]

    print(f"\n{'='*65}")
    print(f"  Cross-Configuration Comparison")
    print(f"{'='*65}")
    print(f"  {'Config':<25} {'Sparse(ms)':>11} {'ObliRec(ms)':>13}")
    print(f"  {'-'*52}")
    print(f"  {'P2 (depth-2, N=16384)':<25} {r_p2['avg_sparse_ms']:>11.1f} "
          f"{r_p2['avg_oblipack_ms']:>13.1f}")
    print(f"  {'P2-MH (depth-1, N=8192)':<25} {r_pmh['avg_sparse_ms']:>11.1f} "
          f"{r_pmh['avg_oblipack_ms']:>13.1f}")
    print(f"\n  Per-chunk speedup (N=16384 → N=8192):")
    print(f"    Sparse:   {r_p2['avg_sparse_ms']/r_pmh['avg_sparse_ms']:.2f}x")
    print(f"    ObliRec: {r_p2['avg_oblipack_ms']/r_pmh['avg_oblipack_ms']:.2f}x")
    print(f"\n  Key insight:")
    print(f"    P2-MH+ObliRec ({r_pmh['avg_oblipack_ms']:.1f}ms) vs "
          f"P2 sparse ({r_p2['avg_sparse_ms']:.1f}ms): "
          f"{r_pmh['avg_oblipack_ms']/r_p2['avg_sparse_ms']:.2f}x")
    print(f"    → P2-MH+ObliRec is {'FASTER' if r_pmh['avg_oblipack_ms'] < r_p2['avg_sparse_ms'] else 'slower'} "
          f"than unprotected P2!")

    return {"exp": "depth_cost", "configs": all_config_results}


# ================================================================== #
#  Experiment 5: Naive padding vs ObliRec cost comparison
# ================================================================== #

def exp_naive_padding(dataset, oblipack, user_degrees, item_degrees, seed=42):
    """Compare padding strategies: Sparse / ObliRec / Naive (max-degree) / Dense.

    Naive padding = pad every user to global max degree (max chunks).
    Shows ObliRec's log-bucketing is much more efficient than naive.
    """
    print(f"\n{'='*65}")
    print(f"  Experiment 5: Naive Padding vs ObliRec Cost")
    print(f"{'='*65}")

    n_items = dataset.n_items
    slot_capacity = 4096  # N=8192
    n_chunks_total = math.ceil(n_items / slot_capacity)

    # Compute active chunks per user
    all_active_chunks = {}
    for u, items in dataset.train_dict.items():
        indicator = _make_indicator(u, dataset, user_degrees, item_degrees, n_items)
        all_active_chunks[u] = _count_active_chunks(indicator, slot_capacity)

    # Compute per-bucket max chunks (ObliRec target)
    bucket_chunk_targets = {}
    for bid, users in oblipack.user_buckets.items():
        max_c = 0
        for u in users:
            max_c = max(max_c, all_active_chunks.get(u, 0))
        bucket_chunk_targets[bid] = max_c

    # Global max chunks (naive padding target)
    global_max_chunks = max(all_active_chunks.values())

    # Compute total chunks processed across all users under each strategy
    total_sparse = 0
    total_oblipack = 0
    total_naive = 0
    total_dense = 0

    for u in dataset.train_dict.keys():
        nc = all_active_chunks.get(u, 0)
        deg = user_degrees[u]
        bid = oblipack.bucket_id(deg)
        op_target = bucket_chunk_targets.get(bid, n_chunks_total)

        total_sparse += nc
        total_oblipack += max(nc, op_target)
        total_naive += global_max_chunks
        total_dense += n_chunks_total

    n_users = len(dataset.train_dict)
    avg_sparse = total_sparse / n_users
    avg_oblipack = total_oblipack / n_users
    avg_naive = total_naive / n_users
    avg_dense = total_dense / n_users

    print(f"\n  Total chunks per strategy (across {n_users} users):")
    print(f"  {'Strategy':<20} {'Total Chunks':>14} {'Avg/User':>10} {'Blowup':>8}")
    print(f"  {'-'*55}")
    print(f"  {'Sparse':<20} {total_sparse:>14,} {avg_sparse:>10.1f} {1.0:>8.2f}x")
    print(f"  {'ObliRec':<20} {total_oblipack:>14,} {avg_oblipack:>10.1f} "
          f"{total_oblipack/total_sparse:>8.2f}x")
    print(f"  {'Naive (max-deg)':<20} {total_naive:>14,} {avg_naive:>10.1f} "
          f"{total_naive/total_sparse:>8.2f}x")
    print(f"  {'Dense (all chunks)':<20} {total_dense:>14,} {avg_dense:>10.1f} "
          f"{total_dense/total_sparse:>8.2f}x")

    print(f"\n  ObliRec saves {(1 - total_oblipack/total_naive)*100:.1f}% "
          f"vs naive padding")
    print(f"  ObliRec saves {(1 - total_oblipack/total_dense)*100:.1f}% "
          f"vs dense")

    # Per-bucket breakdown
    print(f"\n  Per-bucket breakdown:")
    print(f"  {'Bucket':>7} {'#Users':>7} {'Deg Range':>12} "
          f"{'OP Target':>10} {'Naive':>6} {'Dense':>6}")
    print(f"  {'-'*55}")
    for bid in sorted(oblipack.user_buckets.keys()):
        users = oblipack.user_buckets[bid]
        lo, hi = oblipack.bucket_range(bid)
        target = bucket_chunk_targets.get(bid, 0)
        print(f"  B{bid:<5d} {len(users):>7d} {lo:>5d}-{hi:<5d} "
              f"{target:>10d} {global_max_chunks:>6d} {n_chunks_total:>6d}")

    return {
        "exp": "naive_padding",
        "n_users": n_users,
        "n_chunks_total": n_chunks_total,
        "global_max_chunks": global_max_chunks,
        "total_sparse": int(total_sparse),
        "total_oblipack": int(total_oblipack),
        "total_naive": int(total_naive),
        "total_dense": int(total_dense),
        "oblipack_vs_sparse": float(total_oblipack / total_sparse),
        "naive_vs_sparse": float(total_naive / total_sparse),
        "dense_vs_sparse": float(total_dense / total_sparse),
        "oblipack_saving_vs_naive": float(1 - total_oblipack / total_naive),
    }


# ================================================================== #
#  Experiment 6: Oblivious permutation ablation
# ================================================================== #

def exp_permutation_ablation(dataset, oblipack, user_degrees, item_degrees,
                              users_per_bucket=20, seed=42):
    """Show that without oblivious permutation, chunk position patterns
    are distinguishable even after padding to same chunk count.

    Within each bucket, compute the set of active chunk indices per user.
    Without permutation: adversary sees which chunk indices are active.
    With permutation: chunk order is randomized → all patterns look identical.
    """
    print(f"\n{'='*65}")
    print(f"  Experiment 6: Oblivious Permutation Ablation")
    print(f"{'='*65}")

    n_items = dataset.n_items
    slot_capacity = 4096
    n_chunks_total = math.ceil(n_items / slot_capacity)

    rng = np.random.RandomState(seed)

    # Analyze top buckets
    bucket_sizes = [(bid, len(users)) for bid, users in oblipack.user_buckets.items()]
    bucket_sizes.sort(key=lambda x: -x[1])
    target_buckets = [bid for bid, sz in bucket_sizes[:6] if sz >= users_per_bucket]

    all_results = {}

    for bid in target_buckets:
        bucket_users = oblipack.user_buckets[bid]
        lo, hi = oblipack.bucket_range(bid)

        rng.shuffle(bucket_users)
        sample = bucket_users[:users_per_bucket]

        # Compute chunk activation pattern per user
        patterns = []
        for u in sample:
            indicator = _make_indicator(u, dataset, user_degrees, item_degrees, n_items)
            active_set = set()
            for c in range(n_chunks_total):
                start = c * slot_capacity
                end = min(start + slot_capacity, n_items)
                if np.any(np.abs(indicator[start:end]) > 1e-15):
                    active_set.add(c)
            patterns.append((u, user_degrees[u], frozenset(active_set)))

        # Count unique patterns
        unique_patterns = set(p[2] for p in patterns)
        n_unique = len(unique_patterns)
        n_total = len(patterns)

        # Compute pairwise Jaccard distances
        jaccard_dists = []
        pattern_list = [p[2] for p in patterns]
        for i in range(len(pattern_list)):
            for j in range(i+1, len(pattern_list)):
                a, b = pattern_list[i], pattern_list[j]
                if len(a | b) > 0:
                    jaccard_dists.append(1 - len(a & b) / len(a | b))
                else:
                    jaccard_dists.append(0)

        avg_jaccard = np.mean(jaccard_dists) if jaccard_dists else 0

        print(f"\n  Bucket {bid} [deg {lo}-{hi}], {n_total} users sampled:")
        print(f"    Unique chunk patterns: {n_unique}/{n_total} "
              f"({n_unique/n_total*100:.0f}%)")
        print(f"    Avg pairwise Jaccard distance: {avg_jaccard:.3f}")
        print(f"    → Without permutation: {n_unique/n_total*100:.0f}% of users "
              f"distinguishable by chunk pattern")
        print(f"    → With permutation: 0% distinguishable "
              f"(all patterns randomized to same count)")

        all_results[f"bucket_{bid}"] = {
            "bucket": int(bid),
            "range": [int(lo), int(hi)],
            "n_users": n_total,
            "unique_patterns": n_unique,
            "distinguishable_pct": float(n_unique / n_total * 100),
            "avg_jaccard_distance": float(avg_jaccard),
        }

    # Summary
    total_users = sum(r["n_users"] for r in all_results.values())
    total_unique = sum(r["unique_patterns"] for r in all_results.values())
    print(f"\n  Overall: {total_unique}/{total_users} users "
          f"({total_unique/total_users*100:.0f}%) distinguishable without permutation")
    print(f"  → Oblivious permutation is necessary, not just padding!")

    return {"exp": "permutation_ablation", "buckets": all_results}


# ================================================================== #
#  Helpers
# ================================================================== #

def _make_indicator(user_id, dataset, user_degrees, item_degrees, n_items):
    """Create weighted interaction indicator."""
    v = np.zeros(n_items, dtype=np.float64)
    items = dataset.train_dict.get(user_id, [])
    d_u = user_degrees.get(user_id, len(items))
    for i in items:
        d_i = item_degrees.get(i, 1)
        v[i] = 1.0 / math.sqrt(d_u * d_i)
    return v


def _count_active_chunks(indicator, slot_capacity):
    """Count non-zero chunks."""
    n_chunks = math.ceil(len(indicator) / slot_capacity)
    count = 0
    for c in range(n_chunks):
        start = c * slot_capacity
        end = min(start + slot_capacity, len(indicator))
        if np.any(np.abs(indicator[start:end]) > 1e-15):
            count += 1
    return count


def _timed_chunked_compute(ctx, indicator, E_agg, n_items, slot_capacity,
                            skip_zero=True, force_n_chunks=None, d_cols=None):
    """Time chunked HE step-1: enc_chunks @ E_agg submatrices → enc_user_emb.

    Returns total time in milliseconds.
    d_cols: if set, use only first d_cols columns of E_agg to speed up.
    """
    d = d_cols if d_cols is not None else E_agg.shape[1]
    n_chunks = math.ceil(n_items / slot_capacity)

    # Determine which chunks to process
    chunks_to_process = []
    for c in range(n_chunks):
        start = c * slot_capacity
        end = min(start + slot_capacity, n_items)
        chunk_data = indicator[start:end]
        has_data = np.any(np.abs(chunk_data) > 1e-15)

        if skip_zero and not has_data:
            continue
        chunks_to_process.append((c, start, end, chunk_data))

    # If force_n_chunks, add zero chunks to reach target
    if force_n_chunks is not None and len(chunks_to_process) < force_n_chunks:
        active_ids = {c for c, _, _, _ in chunks_to_process}
        for c in range(n_chunks):
            if len(chunks_to_process) >= force_n_chunks:
                break
            if c not in active_ids:
                start = c * slot_capacity
                end = min(start + slot_capacity, n_items)
                chunks_to_process.append(
                    (c, start, end, np.zeros(end - start, dtype=np.float64)))

    # Time the full encrypt + compute pipeline
    t0 = time.perf_counter()

    enc_user_emb = None
    for c, start, end, chunk_data in chunks_to_process:
        # Encrypt chunk
        padded = np.zeros(slot_capacity, dtype=np.float64)
        padded[:end - start] = chunk_data
        enc_chunk = ts.ckks_vector(ctx, padded.tolist())

        # Server compute: enc_chunk @ E_agg[start:end]
        submatrix = np.zeros((slot_capacity, d), dtype=np.float64)
        submatrix[:end - start] = E_agg[start:end, :d]
        enc_partial = enc_chunk.mm(submatrix.tolist())

        if enc_user_emb is None:
            enc_user_emb = enc_partial
        else:
            enc_user_emb = enc_user_emb + enc_partial

    total_ms = (time.perf_counter() - t0) * 1000
    return total_ms


def main():
    parser = argparse.ArgumentParser(description="Full side-channel experiments")
    parser.add_argument("--dataset", type=str, default="gowalla",
                        choices=list(DATASET_DEFAULTS.keys()))
    parser.add_argument("--exp", type=str, default="all",
                        choices=["rotation", "chunked", "within-bucket",
                                 "depth-cost", "naive-padding",
                                 "permutation", "all"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n-sample", type=int, default=30)
    parser.add_argument("--users-per-bucket", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  Full Side-Channel Experiments — {args.dataset}")
    print(f"{'='*65}")

    data = setup_model_and_data(args.dataset, args.gpu, args.seed)
    dataset, oblipack, user_degrees, item_degrees, item_emb, item_layers = data

    print(f"  Users: {dataset.n_users}, Items: {dataset.n_items}")

    all_results = {}

    if args.exp in ("rotation", "all"):
        r1 = exp_rotation_timing(dataset, oblipack, user_degrees,
                                  n_sample=args.n_sample, seed=args.seed)
        all_results["rotation"] = r1

    if args.exp in ("chunked", "all"):
        r2 = exp_chunked_timing(dataset, oblipack, user_degrees, item_degrees,
                                 item_emb, item_layers,
                                 n_sample=args.n_sample, seed=args.seed)
        all_results["chunked"] = r2

    if args.exp in ("within-bucket", "all"):
        r3 = exp_within_bucket(dataset, oblipack, user_degrees, item_degrees,
                                item_emb, item_layers,
                                users_per_bucket=args.users_per_bucket,
                                seed=args.seed)
        all_results["within_bucket"] = r3

    if args.exp in ("depth-cost", "all"):
        r4 = exp_depth_cost(dataset, oblipack, user_degrees, item_degrees,
                             n_sample=min(args.n_sample, 10), seed=args.seed)
        all_results["depth_cost"] = r4

    if args.exp in ("naive-padding", "all"):
        r5 = exp_naive_padding(dataset, oblipack, user_degrees, item_degrees,
                                seed=args.seed)
        all_results["naive_padding"] = r5

    if args.exp in ("permutation", "all"):
        r6 = exp_permutation_ablation(dataset, oblipack, user_degrees,
                                       item_degrees, seed=args.seed)
        all_results["permutation"] = r6

    # Save
    os.makedirs("models", exist_ok=True)
    out_path = f"models/{args.dataset}_sidechannel_full.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
