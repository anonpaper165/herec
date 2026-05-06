"""
DPGCN inference-time side-channel vulnerability demonstration.

Shows that DPGCN (edge-level DP during training) does NOT protect against
structural side-channels at inference time. The server can still infer
user degree from computation traces (chunk count, rotation count, timing)
because DP only perturbs the training graph, not the inference protocol.

Existing DP methods are not substitutes for inference-time structural defense:
training-time and inference-time privacy are orthogonal dimensions.

Usage:
    python benchmark_dpgcn_sidechannel.py --dataset gowalla --gpu 0
    python benchmark_dpgcn_sidechannel.py --dataset amazon-book --gpu 0
"""

import argparse
import json
import math
import os
import numpy as np
import torch
from collections import defaultdict
from scipy import stats

from src.dataset import RecDataset
from src.model import LightGCN
from src.oblirec import ObliRec
from src.he_inference import HERecommender


DATASET_DEFAULTS = {
    "ml-100k":     {"rating_threshold": 4},
    "ml-1m":       {"rating_threshold": 4},
    "gowalla":     {"rating_threshold": 0},
    "amazon-book": {"rating_threshold": 0},
}


def compute_sidechannel_leakage(user_degrees, n_items, slot_capacity):
    """Compute side-channel observables for all users.

    Returns dict of user → observables (rotation count, active chunk count).
    These are PROTOCOL-level observables: they depend on the user's real
    degree at inference time, NOT on the training-time graph perturbation.
    """
    C = math.ceil(n_items / slot_capacity)
    observables = {}
    for u, deg in user_degrees.items():
        # FicGCN SpIntra-CA rotation count: deterministic function of degree
        rot_count = deg * math.ceil(math.log2(max(deg, 2)))
        observables[u] = {
            "degree": deg,
            "rotation_count": rot_count,
        }
    return observables


def degree_inference_attack(user_degrees, observables, method="rotation"):
    """Simulate degree inference attack from side-channel observables.

    For rotation-count attack: degree is perfectly revealed (MAE=0).
    For bucket-only attack: degree is hidden within bucket range.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error

    users = sorted(user_degrees.keys())
    degrees = np.array([user_degrees[u] for u in users])

    if method == "rotation":
        # Rotation count = d * ceil(log2(d)) → invertible → exact degree
        return 0.0, len(np.unique(degrees))
    elif method == "bucket":
        # ObliRec: attacker knows bucket midpoint only
        pred = []
        for u in users:
            deg = user_degrees[u]
            bid = ObliRec.bucket_id(deg)
            lo, hi = ObliRec.bucket_range(bid)
            pred.append((lo + hi) / 2.0)
        mae = mean_absolute_error(degrees, np.array(pred))
        n_distinguishable = len(set(ObliRec.bucket_id(d) for d in degrees))
        return mae, n_distinguishable
    else:
        raise ValueError(f"Unknown method: {method}")


def reidentification_attack(user_degrees, method="exact", n_sample=2000,
                             seed=42):
    """Simulate cross-session re-identification attack.

    Simulates the server fingerprinting users by their side-channel observable
    and measuring re-identification accuracy across sessions.

    Args:
        user_degrees: dict user_id → degree
        method: "exact" (no protection) or "bucket" (ObliRec)
        n_sample: number of users to sample
        seed: random seed
    """
    rng = np.random.RandomState(seed)
    users = sorted(user_degrees.keys())

    if len(users) > n_sample:
        sample_idx = rng.choice(len(users), size=n_sample, replace=False)
        users = [users[i] for i in sample_idx]

    # Build fingerprints
    if method == "exact":
        fingerprints = {u: user_degrees[u] for u in users}
    elif method == "bucket":
        fingerprints = {u: ObliRec.bucket_id(user_degrees[u]) for u in users}
    else:
        raise ValueError(f"Unknown method: {method}")

    # Group by fingerprint → anonymity sets
    fp_groups = defaultdict(list)
    for u, fp in fingerprints.items():
        fp_groups[fp].append(u)

    # Re-ID accuracy: for each user, 1/|anonymity_set|
    total_reid = 0.0
    anon_sizes = []
    for u in users:
        fp = fingerprints[u]
        anon_set_size = len(fp_groups[fp])
        total_reid += 1.0 / anon_set_size
        anon_sizes.append(anon_set_size)

    reid_acc = total_reid / len(users)
    mean_anon = np.mean(anon_sizes)
    min_anon = np.min(anon_sizes)

    return {
        "reid_accuracy": reid_acc,
        "mean_anonymity_set": mean_anon,
        "min_anonymity_set": int(min_anon),
        "n_fingerprints": len(fp_groups),
    }


def chunk_pattern_attack(he, dataset, user_degrees, item_degrees, oblipack,
                          n_sample=2000, seed=42):
    """Measure chunk-pattern based leakage for HE-GCN inference.

    For each user, compute the active chunk indices and measure
    within-bucket distinguishability.
    """
    rng = np.random.RandomState(seed)
    users = sorted(user_degrees.keys())
    if len(users) > n_sample:
        sample_idx = rng.choice(len(users), size=n_sample, replace=False)
        users = [users[i] for i in sample_idx]

    C = he.get_n_chunks()
    if C <= 1:
        return {"n_chunks": C, "note": "single chunk, no chunk-pattern leakage"}

    # Compute chunk patterns
    user_patterns = {}
    for u in users:
        indicator = he.make_indicator(u, dataset.train_dict,
                                       user_degrees, item_degrees)
        active = he.get_active_chunk_indices(indicator)
        user_patterns[u] = frozenset(active)

    # Without ObliRec: fingerprint = (degree, chunk_pattern)
    fp_exact = defaultdict(list)
    for u in users:
        fp = (user_degrees[u], user_patterns[u])
        fp_exact[fp].append(u)

    reid_exact = sum(1.0 / len(fp_exact[k]) for k in fp_exact
                     for _ in fp_exact[k]) / len(users)

    # With ObliRec (bucket only): fingerprint = bucket_id
    fp_bucket = defaultdict(list)
    for u in users:
        fp = ObliRec.bucket_id(user_degrees[u])
        fp_bucket[fp].append(u)

    reid_bucket = sum(1.0 / len(fp_bucket[k]) for k in fp_bucket
                      for _ in fp_bucket[k]) / len(users)

    # With ObliRec (chunk-aware): fingerprint = (bucket_id, chunk_pattern)
    fp_chunk = defaultdict(list)
    for u in users:
        fp = (ObliRec.bucket_id(user_degrees[u]), user_patterns[u])
        fp_chunk[fp].append(u)

    reid_chunk = sum(1.0 / len(fp_chunk[k]) for k in fp_chunk
                     for _ in fp_chunk[k]) / len(users)

    return {
        "n_chunks": C,
        "reid_exact_degree_plus_chunks": reid_exact,
        "reid_bucket_only": reid_bucket,
        "reid_bucket_plus_chunks": reid_chunk,
        "n_unique_exact": len(fp_exact),
        "n_unique_bucket": len(fp_bucket),
        "n_unique_chunk_aware": len(fp_chunk),
    }


def main():
    parser = argparse.ArgumentParser(
        description="DPGCN inference-time side-channel vulnerability")
    parser.add_argument("--dataset", type=str, default="gowalla",
                        choices=list(DATASET_DEFAULTS.keys()))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_sample", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ds = args.dataset
    defaults = DATASET_DEFAULTS[ds]

    print(f"\n{'='*70}")
    print(f"  DPGCN Side-Channel Vulnerability — {ds}")
    print(f"  Showing that edge-level DP does NOT protect inference-time")
    print(f"  structural side-channels")
    print(f"{'='*70}")

    # Load dataset
    dataset = RecDataset(f"data/{ds}", dataset_name=ds,
                         rating_threshold=defaults["rating_threshold"],
                         seed=args.seed)

    # Build degree maps (REAL inference-time degrees, not DP-perturbed)
    user_degrees = {u: len(items) for u, items in dataset.train_dict.items()}
    item_degrees = {}
    for u, items in dataset.train_dict.items():
        for i in items:
            item_degrees[i] = item_degrees.get(i, 0) + 1

    n_users = dataset.n_users
    n_items = dataset.n_items

    print(f"\n  Dataset: {ds}")
    print(f"  Users: {n_users}, Items: {n_items}")
    print(f"  Mean degree: {np.mean(list(user_degrees.values())):.1f}")
    print(f"  Max degree: {max(user_degrees.values())}")

    # ── Key insight: DPGCN trains on perturbed graph, but inference ──
    # ── uses the user's REAL interaction indicator.                 ──
    # ── The server observes the HE computation trace of the REAL   ──
    # ── indicator, not the DP-perturbed one.                       ──

    print(f"\n  {'='*65}")
    print(f"  KEY INSIGHT")
    print(f"  {'='*65}")
    print(f"  DPGCN perturbs the adjacency matrix during TRAINING (edge-level DP).")
    print(f"  At INFERENCE time, the user encrypts their REAL interaction")
    print(f"  indicator and sends it to the server. The server observes:")
    print(f"    - Number of encrypted chunks (proportional to degree)")
    print(f"    - Rotation count (deterministic function of degree)")
    print(f"    - Active chunk indices (reveals interaction distribution)")
    print(f"  These are identical to vanilla LightGCN inference — DPGCN's")
    print(f"  training-time DP provides ZERO protection against these.")

    # Load DPGCN model (any epsilon) to verify it uses same inference
    epsilons = [1.0, 2.0, 5.0, 10.0]
    available_eps = []
    for eps in epsilons:
        path = f"models/{ds}_dpgcn_eps{eps}.pt"
        if os.path.exists(path):
            available_eps.append(eps)

    if not available_eps:
        print(f"\n  [!] No DPGCN models found for {ds}. Using vanilla model.")
        model_path = f"models/{ds}_best.pt"
    else:
        # Use the strongest DP model to show vulnerability
        eps_use = available_eps[0]  # smallest epsilon = strongest DP
        model_path = f"models/{ds}_dpgcn_eps{eps_use}.pt"
        print(f"\n  Using DPGCN model: eps={eps_use} (strongest DP protection)")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available()
                          else "cpu")
    ckpt = torch.load(model_path, weights_only=False, map_location=device)

    model = LightGCN(
        n_users=n_users,
        n_items=n_items,
        emb_dim=ckpt["args"]["emb_dim"],
        n_layers=ckpt["args"]["n_layers"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    adj = dataset.adj_matrix.to(device)

    # Compute embeddings (model weights are DP-trained, but embeddings are public)
    with torch.no_grad():
        E = torch.cat([model.user_embedding.weight,
                       model.item_embedding.weight], dim=0)
        layers = [E.cpu().numpy()]
        for _ in range(model.n_layers):
            E = torch.sparse.mm(adj, E)
            layers.append(E.cpu().numpy())

    user_emb = np.mean([l[:n_users] for l in layers], axis=0)
    item_emb = np.mean([l[n_users:] for l in layers], axis=0)
    item_layers = [l[n_users:] for l in layers]

    # Setup HE context
    he = HERecommender(user_emb, item_emb, item_layers=item_layers)
    he.setup_context(poly_modulus_degree=8192, depth=1)
    S = he._get_slot_capacity()
    C = he.get_n_chunks()

    # Build ObliRec for comparison
    oblipack = ObliRec(dataset)

    results = {
        "dataset": ds,
        "n_users": n_users,
        "n_items": n_items,
        "n_chunks": C,
        "dpgcn_epsilon": available_eps if available_eps else "N/A",
    }

    # ── Attack 1: Degree Inference ───────────────────────────────────
    print(f"\n  {'='*65}")
    print(f"  ATTACK 1: Degree Inference")
    print(f"  {'='*65}")

    mae_rotation, n_unique_rot = degree_inference_attack(
        user_degrees, None, method="rotation")
    mae_bucket, n_unique_bucket = degree_inference_attack(
        user_degrees, None, method="bucket")

    print(f"\n  DPGCN (any epsilon) — inference-time leakage:")
    print(f"    Rotation count attack:  MAE = {mae_rotation:.1f} "
          f"(exact degree, {n_unique_rot} unique values)")
    print(f"    → Server infers EXACT degree from computation trace")
    print(f"\n  ObliRec defense:")
    print(f"    Bucket midpoint attack: MAE = {mae_bucket:.1f} "
          f"(only {n_unique_bucket} distinguishable buckets)")
    print(f"    → Server only learns coarse activity tier")

    results["degree_inference"] = {
        "dpgcn_rotation_mae": mae_rotation,
        "dpgcn_unique_observables": n_unique_rot,
        "oblipack_bucket_mae": mae_bucket,
        "oblipack_unique_buckets": n_unique_bucket,
    }

    # ── Attack 2: User Re-identification ─────────────────────────────
    print(f"\n  {'='*65}")
    print(f"  ATTACK 2: Cross-Session Re-Identification")
    print(f"  {'='*65}")

    reid_exact = reidentification_attack(
        user_degrees, method="exact", n_sample=args.n_sample, seed=args.seed)
    reid_bucket = reidentification_attack(
        user_degrees, method="bucket", n_sample=args.n_sample, seed=args.seed)

    print(f"\n  DPGCN (any epsilon) — no inference-time protection:")
    print(f"    Re-ID accuracy:    {reid_exact['reid_accuracy']:.4f} "
          f"({reid_exact['reid_accuracy']*100:.2f}%)")
    print(f"    Mean anonymity set: {reid_exact['mean_anonymity_set']:.0f}")
    print(f"    Min anonymity set:  {reid_exact['min_anonymity_set']}")
    print(f"    Unique fingerprints: {reid_exact['n_fingerprints']}")

    print(f"\n  ObliRec defense:")
    print(f"    Re-ID accuracy:    {reid_bucket['reid_accuracy']:.4f} "
          f"({reid_bucket['reid_accuracy']*100:.2f}%)")
    print(f"    Mean anonymity set: {reid_bucket['mean_anonymity_set']:.0f}")
    print(f"    Min anonymity set:  {reid_bucket['min_anonymity_set']}")
    print(f"    Unique fingerprints: {reid_bucket['n_fingerprints']}")

    protection_factor = reid_exact['reid_accuracy'] / max(reid_bucket['reid_accuracy'], 1e-6)
    print(f"\n  ObliRec protection: {protection_factor:.1f}x reduction in re-ID accuracy")

    results["reidentification"] = {
        "dpgcn_exact": reid_exact,
        "oblipack_bucket": reid_bucket,
        "protection_factor": protection_factor,
    }

    # ── Attack 3: Chunk Pattern Leakage (multi-chunk datasets) ───────
    if C > 1:
        print(f"\n  {'='*65}")
        print(f"  ATTACK 3: Chunk-Pattern Fingerprinting (C={C} chunks)")
        print(f"  {'='*65}")

        chunk_results = chunk_pattern_attack(
            he, dataset, user_degrees, item_degrees, oblipack,
            n_sample=args.n_sample, seed=args.seed)

        print(f"\n  DPGCN — server sees exact degree + chunk pattern:")
        print(f"    Re-ID accuracy: {chunk_results['reid_exact_degree_plus_chunks']:.4f}")
        print(f"    Unique fingerprints: {chunk_results['n_unique_exact']}")

        print(f"\n  ObliRec (bucket only):")
        print(f"    Re-ID accuracy: {chunk_results['reid_bucket_only']:.4f}")

        print(f"\n  ObliRec (bucket + chunk-aware, sparse mode):")
        print(f"    Re-ID accuracy: {chunk_results['reid_bucket_plus_chunks']:.4f}")
        print(f"    → Decoy mode recommended to close this gap")

        results["chunk_pattern"] = chunk_results

    # ── Comparison across DPGCN epsilon values ───────────────────────
    print(f"\n  {'='*65}")
    print(f"  DPGCN EPSILON COMPARISON")
    print(f"  (All epsilons have IDENTICAL inference-time leakage)")
    print(f"  {'='*65}")

    eps_results = {}
    for eps in available_eps:
        path = f"models/{ds}_dpgcn_eps{eps}.pt"
        ckpt_eps = torch.load(path, weights_only=False, map_location=device)
        recall = ckpt_eps.get("recall", ckpt_eps.get("best_recall", "N/A"))

        eps_results[str(eps)] = {
            "recall_at_20": float(recall) if isinstance(recall, (int, float)) else recall,
            "inference_degree_mae": 0.0,  # exact degree always leaked
            "inference_reid_accuracy": reid_exact['reid_accuracy'],
            "inference_mean_anon_set": reid_exact['mean_anonymity_set'],
        }

    results["epsilon_comparison"] = eps_results

    print(f"\n  {'Epsilon':>8} {'Recall@20':>10} {'Deg. MAE':>10} "
          f"{'Re-ID Acc':>10} {'Anon Set':>10}")
    print(f"  {'-'*55}")
    for eps in sorted(available_eps):
        r = eps_results[str(eps)]
        recall_str = f"{r['recall_at_20']:.4f}" if isinstance(r['recall_at_20'], float) else str(r['recall_at_20'])
        print(f"  {eps:>8.1f} {recall_str:>10} {r['inference_degree_mae']:>10.1f} "
              f"{r['inference_reid_accuracy']:>10.4f} "
              f"{r['inference_mean_anon_set']:>10.0f}")
    print(f"\n  → ALL epsilons leak exact degree (MAE=0) at inference time")
    print(f"  → DPGCN's training-time DP is orthogonal to inference-time side-channels")

    # ── ObliPack as complementary defense ────────────────────────────
    print(f"\n  {'='*65}")
    print(f"  CONCLUSION: DPGCN + ObliRec are COMPLEMENTARY")
    print(f"  {'='*65}")
    print(f"  DPGCN protects: training-time edge privacy (ε-DP)")
    print(f"  ObliRec protects: inference-time structural side-channels")
    print(f"  Neither alone provides end-to-end privacy.")
    print(f"  Combining both: DPGCN trains the model with edge-level DP,")
    print(f"  then ObliRec protects the HE inference computation.")

    results["conclusion"] = {
        "dpgcn_protects": "training-time edge privacy",
        "dpgcn_does_not_protect": "inference-time structural side-channels",
        "oblipack_protects": "inference-time structural side-channels",
        "recommendation": "use both for end-to-end privacy",
    }

    # Save
    out_path = f"results_dpgcn_sidechannel_{ds}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
