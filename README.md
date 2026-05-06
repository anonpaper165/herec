# Encrypted but Observable: Auditing Structural Leakage in Graph Recommendation

Code for reproducing experiments in the anonymous submission
**"Encrypted but Observable: Auditing Structural Leakage in Graph Recommendation."**

---

## Overview

Homomorphic encryption (HE) protects recommendation content but leaves structural metadata
(operation counts, ciphertext chunk patterns, timing) visible to an honest-but-curious server.
This repository audits two structural leakage channels and demonstrates ObliRec's defenses:

- **Channel 1** — support-size leakage via operation-volume signals (e.g., rotation count)
- **Channel 2** — support-location leakage via the active chunk-index set

Five audit metrics: Degree MAE, Re-ID accuracy, Anonymity set size (mean/min),
Chunk unique %, Timing CV.

---

## Setup

**Python 3.8+** and a CUDA-capable GPU (optional; CPU works for all non-HE experiments).

```bash
pip install -r requirements.txt
```

Key dependencies: `torch`, `tenseal`, `numpy`, `scipy`, `matplotlib`.

### Data

Download the four datasets and place them under `data/`:

| Dataset | Source | Path |
|---------|--------|------|
| ML-100K | [grouplens.org](https://grouplens.org/datasets/movielens/100k/) | `data/ml-100k/` |
| ML-1M | [grouplens.org](https://grouplens.org/datasets/movielens/1m/) | `data/ml-1m/` |
| Gowalla | [LightGCN repo](https://github.com/kuandeng/LightGCN) | `data/gowalla/` |
| Amazon-Book | [LightGCN repo](https://github.com/kuandeng/LightGCN) | `data/amazon-book/` |

Gowalla and Amazon-Book use pre-split `train.txt` / `test.txt` from the LightGCN repository.
ML-100K and ML-1M use ratings filtered at threshold 4.

### Train base models

```bash
python train.py --dataset ml-100k
python train.py --dataset ml-1m
python train.py --dataset gowalla
python train.py --dataset amazon-book
```

Checkpoints are saved to `models/DATASET_best.pt` (e.g., `models/gowalla_best.pt`).

---

## Reproducing Paper Results

### HE inference sanity-check (Table 2 reference)

```bash
python benchmark_e2e.py --dataset ml-100k
python benchmark_e2e.py --dataset ml-1m
python benchmark_e2e.py --dataset gowalla
python benchmark_e2e.py --dataset amazon-book
```

Requires trained models. Reports Recall@20 for Vanilla LightGCN and Protocol 1/2 HE
variants under ObliPack buckets. This is a sanity-check that HE inference preserves
recommendation quality; exact Table 2 numbers depend on hardware-specific HE timing
and may differ slightly from the paper.

### Table 3 — Structural leakage under four deployment regimes

```bash
python audit/run_structural_audit.py --dataset ml-100k
python audit/run_structural_audit.py --dataset ml-1m
python audit/run_structural_audit.py --dataset gowalla
python audit/run_structural_audit.py --dataset amazon-book
# or all at once:
python audit/run_structural_audit.py --dataset all
```

Uses the default log-scale bucketing configuration used in the main structural audit.
Adaptive bucketing variants (including λ=10) are reported in the appendix bucketing table.

### Table 4 — Inference-time structural leakage unchanged by training-time DP

```bash
python audit/run_dp_mismatch.py --dataset gowalla
python audit/run_dp_mismatch.py --dataset amazon-book
```

Requires DPGCN checkpoints (e.g., `models/gowalla_dpgcn_eps1.0.pt`).
If absent, the script reports vanilla sparse HE results only.

### Figure 1 — Degree vs. observable side-channel (correlation paradox)

```bash
python generate_correlation_figure.py --dataset gowalla
```

Outputs `figures/fig_correlation.pdf`.

### Appendix — Per-user chunked HE cost across deployment modes

```bash
python benchmark_he_timing.py --dataset gowalla
python benchmark_he_timing.py --dataset amazon-book
```

Requires TenSEAL. Reports per-user HE inference latency under sparse, bucketed,
decoy (k=1), and dense modes.

### Appendix — Bucketing strategies (adaptive vs. log-scale)

```bash
python benchmark_adaptive.py --dataset ml-100k
python benchmark_adaptive.py --dataset gowalla
python benchmark_adaptive.py --dataset amazon-book
```

### Appendix — Degree-trajectory re-ID across sessions

```bash
python run_multisession.py
```

### Appendix — Item-layout sensitivity

```bash
python run_item_reorder_audit.py
```

Runs Gowalla and Amazon-Book automatically.

### Appendix — Fixed-fanout trace simulation (GraphSAGE-style)

```bash
python run_graphsage_trace_audit.py
```

Runs Gowalla and Amazon-Book automatically.

---

## Repository Structure

```
herec/
├── requirements.txt
├── train.py                        # LightGCN training
├── src/
│   ├── dataset.py                  # Dataset loading (ML-100K/1M, Gowalla, Amazon-Book)
│   ├── model.py                    # LightGCN model
│   ├── evaluate.py                 # Recall@K, NDCG@K
│   ├── he_inference.py             # HE inference (TenSEAL/CKKS, P2 and P2-MH protocols)
│   └── oblirec.py                  # ObliRec: bucketing, decoy, dense modes
├── structural_audit/
│   ├── core.py                     # ExecutionTrace, LeakageMetrics data structures
│   └── metrics.py                  # Five leakage metric computations
├── audit/
│   ├── run_structural_audit.py     # Table 3: four-regime structural leakage audit
│   └── run_dp_mismatch.py          # Table 4: DPGCN inference-time leakage
├── benchmark_e2e.py                # Table 2: Recall@20 comparison
├── benchmark_he_timing.py          # Appendix: per-mode HE latency
├── benchmark_sidechannel_full.py   # Full side-channel experiment suite
├── benchmark_dpgcn_sidechannel.py  # DPGCN side-channel analysis
├── benchmark_adaptive.py           # Appendix: adaptive vs. log bucketing
├── benchmark_decoy_latency.py      # Decoy mode latency overhead
├── generate_correlation_figure.py  # Figure 1: correlation paradox
├── run_multisession.py             # Appendix: multi-session re-ID
├── run_item_reorder_audit.py       # Appendix: item-layout sensitivity
├── run_graphsage_trace_audit.py    # Appendix: GraphSAGE trace audit
├── run_bucketing_analysis.py       # Bucketing strategy comparison
└── run_decoy_analysis.py           # Decoy chunk analysis
```

---

## Notes

- All random seeds default to `--seed 42` for reproducibility.
- HE timing benchmarks (`benchmark_he_timing.py`, `benchmark_decoy_latency.py`,
  `benchmark_e2e.py`) require TenSEAL (`pip install tenseal==0.3.16`) and
  report wall-clock times on the machine where they are run; absolute latency values
  will differ from the paper's hardware but relative comparisons (sparse vs. ObliRec)
  hold across platforms.
- The structural leakage audit (`audit/run_structural_audit.py`) does not
  require trained models and can be run immediately after data download.
- Decoy mode uses a per-user deterministic seed (static assignment), which
  models the fixed-RNG fingerprinting vulnerability described in §6.5 of the paper.
  Per-session randomized decoy assignment reduces this static-decoy fingerprint;
  finite-k decoys still leave residual chunk-index leakage, while dense submission
  removes the modeled chunk-index channel.
