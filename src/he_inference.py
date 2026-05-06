"""
HE-based LightGCN inference using TenSEAL CKKS.

Two protocols:

Protocol 1 (User Selector):
  - Client encrypts one-hot user selector (n_users,)
  - Server: enc_selector @ ScoreMatrix → enc_scores (n_items,)
  - Depth 1, fast (~2s), exact match with vanilla LightGCN

Protocol 2 (Interaction-Based, ObliRec-compatible):
  - Client encrypts weighted interaction indicator (n_items,)
  - Server: enc_indicator @ E_I → enc_user_emb → @ E_I^T → enc_scores
  - Depth 2, slower (~8s), approximates vanilla via item embedding aggregation
  - Compatible with ObliRec padding (dummy entries = 0, contribute nothing)
"""

import time
import math
import numpy as np
import tenseal as ts


class HERecommender:
    # Threshold (bytes) above which we skip dense matrix pre-computation.
    # 2 GB default — avoids OOM on large datasets like Gowalla/Amazon-Book.
    _MAX_DENSE_BYTES = 2 * 1024**3

    def __init__(self, user_emb_final, item_emb_final, item_layers=None):
        """
        Args:
            user_emb_final: (n_users, d) numpy array from trained LightGCN
            item_emb_final: (n_items, d) numpy array from trained LightGCN
            item_layers: optional list of per-layer item embeddings
                         [E_I^(0), E_I^(1), ..., E_I^(K)], each (n_items, d)
        """
        self.user_emb = np.array(user_emb_final, dtype=np.float64)
        self.item_emb = np.array(item_emb_final, dtype=np.float64)
        self.n_users, self.emb_dim = self.user_emb.shape
        self.n_items = self.item_emb.shape[0]

        # Pre-compute score matrix for Protocol 1 (only if small enough)
        score_bytes = self.n_users * self.n_items * 8
        if score_bytes <= self._MAX_DENSE_BYTES:
            self.score_matrix = self.user_emb @ self.item_emb.T
        else:
            self.score_matrix = None  # compute on-the-fly
            print(f"[HERecommender] Skipping dense score_matrix "
                  f"({self.n_users}x{self.n_items} = {score_bytes/1e9:.1f}GB)")

        # multi-hop matrices (if per-layer embeddings provided)
        self.E_I_agg = None
        self.score_matrix_multihop = None
        if item_layers is not None:
            self._setup_multihop(item_layers)

        self.ctx = None

    def _setup_multihop(self, item_layers):
        """Pre-compute multi-hop aggregation matrices.

        Vanilla LightGCN user embedding:
            e_u_final = (1/(K+1)) * (e_u^(0) + e_u^(1) + ... + e_u^(K))

        Where e_u^(k) = v @ E_I^(k-1) for k >= 1.

        Standard P2 (over-propagation):
            v @ E_I_final = (1/(K+1)) * (v@E_I^0 + v@E_I^1 + ... + v@E_I^K)
            Includes an extra (K+1)-th hop term v @ E_I^(K) beyond
            what vanilla LightGCN computes.

        Multi-hop (aligned with LightGCN):
            e_u_approx = (1/(K+1)) * (v@E_I^0 + v@E_I^1 + ... + v@E_I^(K-1))
                       = v @ E_I_agg
            Drops the extra hop. Only missing e_u^(0).

        E_I_agg uses layers 0..K-1 (not K), matching LightGCN's actual propagation.
        """
        layers = [np.array(l, dtype=np.float64) for l in item_layers]
        K = len(layers) - 1  # number of propagation layers

        # E_I_agg = (1/(K+1)) * sum(E_I^(0), ..., E_I^(K-1))
        self.E_I_agg = np.mean(layers[:-1], axis=0) * (K / (K + 1))

        # one-step scoring: S_multihop = E_I_agg @ E_I_final^T  (n_items × n_items)
        # Only pre-compute if fits in memory
        mh_bytes = self.n_items * self.n_items * 8
        if mh_bytes <= self._MAX_DENSE_BYTES:
            self.score_matrix_multihop = self.E_I_agg @ self.item_emb.T
            print(f"Multi-hop setup: K={K}, E_I_agg from layers 0..{K-1}, "
                  f"S_multihop shape={self.score_matrix_multihop.shape}")
        else:
            self.score_matrix_multihop = None
            print(f"Multi-hop setup: K={K}, E_I_agg from layers 0..{K-1}, "
                  f"S_multihop skipped ({self.n_items}x{self.n_items} = {mh_bytes/1e9:.1f}GB)")

    def _encrypt(self, vector):
        return ts.ckks_vector(self.ctx, vector.tolist())

    # ------------------------------------------------------------------ #
    #  Protocol 1: User Selector × Score Matrix (depth 1)
    # ------------------------------------------------------------------ #

    def p1_encrypt(self, user_id):
        """Client: create encrypted one-hot user selector."""
        selector = np.zeros(self.n_users, dtype=np.float64)
        selector[user_id] = 1.0
        return self._encrypt(selector)

    def p1_server_compute(self, enc_selector):
        """Server: enc_selector @ score_matrix → enc_scores."""
        if self.score_matrix is not None:
            return enc_selector.mm(self.score_matrix.tolist())
        # On-the-fly: enc_selector @ user_emb → enc_u_emb, then @ item_emb.T
        enc_u = enc_selector.mm(self.user_emb.tolist())
        return enc_u.mm(self.item_emb.T.tolist())

    def p1_decrypt(self, enc_scores):
        """Client: decrypt and return scores."""
        return np.array(enc_scores.decrypt()[:self.n_items])

    def p1_plaintext_ref(self, user_id):
        """Plaintext reference scores for Protocol 1."""
        if self.score_matrix is not None:
            return self.score_matrix[user_id]
        return self.user_emb[user_id] @ self.item_emb.T

    # ------------------------------------------------------------------ #
    #  Protocol 2: Interaction Indicator (ObliRec-compatible, depth 2)
    # ------------------------------------------------------------------ #

    def make_indicator(self, user_id, train_dict, user_degrees, item_degrees):
        """Create weighted interaction indicator for a user.

        v[i] = 1/sqrt(d_u * d_i) if user interacted with item i, else 0.
        This matches LightGCN's symmetric normalization.
        """
        v = np.zeros(self.n_items, dtype=np.float64)
        items = train_dict.get(user_id, [])
        d_u = user_degrees.get(user_id, len(items))
        for i in items:
            d_i = item_degrees.get(i, 1)
            v[i] = 1.0 / math.sqrt(d_u * d_i)
        return v

    def make_oblirec_indicator(self, user_id, oblipack):
        """Create ObliRec-padded indicator using pre-built packing structures.

        The padded indicator has the same non-zero pattern as ObliRec's
        neighbor list (including dummy entries at 0 weight).
        Since dummy weights are 0, they don't affect the result.
        The key privacy property: the NUMBER of non-zeros is hidden
        (all users in the same bucket have the same padded size).
        """
        v = np.zeros(self.n_items, dtype=np.float64)
        if user_id in oblipack.user_padded_neighbors:
            neighbors = oblipack.user_padded_neighbors[user_id]
            weights = oblipack.user_padded_weights[user_id]
            for idx, w in zip(neighbors, weights):
                if idx < self.n_items:  # skip dummy index
                    v[idx] = w
        return v

    def p2_encrypt(self, indicator):
        """Client: encrypt interaction indicator vector."""
        return self._encrypt(indicator)

    def p2_server_compute(self, enc_indicator):
        """Server: two-step matmul.

        Step 1: enc_indicator @ E_I → enc_user_emb (d,)
        Step 2: enc_user_emb @ E_I^T → enc_scores (n_items,)
        """
        enc_user_emb = enc_indicator.mm(self.item_emb.tolist())
        enc_scores = enc_user_emb.mm(self.item_emb.T.tolist())
        return enc_scores

    def p2_decrypt(self, enc_scores):
        """Client: decrypt scores."""
        return np.array(enc_scores.decrypt()[:self.n_items])

    def p2_plaintext_ref(self, indicator):
        """Plaintext reference for Protocol 2."""
        user_emb = indicator @ self.item_emb
        return user_emb @ self.item_emb.T

    # ------------------------------------------------------------------ #
    #  Protocol 2-MH: Multi-Hop (ObliRec-compatible, depth 1)
    # ------------------------------------------------------------------ #

    def p2mh_server_compute_onestep(self, enc_indicator):
        """Server: one-step multi-hop scoring (depth 1).

        enc_indicator @ S_multihop → enc_scores
        where S_multihop = E_I_agg @ E_I_final^T is pre-computed.

        Same depth and speed as Protocol 1, but privacy of Protocol 2.
        """
        if self.score_matrix_multihop is not None:
            return enc_indicator.mm(self.score_matrix_multihop.tolist())
        # Fallback: two-step (depth 2)
        return self.p2mh_server_compute_twostep(enc_indicator)

    def p2mh_server_compute_twostep(self, enc_indicator):
        """Server: two-step multi-hop scoring (depth 2).

        Step 1: enc_indicator @ E_I_agg → enc_user_emb
        Step 2: enc_user_emb @ E_I_final^T → enc_scores
        """
        enc_user_emb = enc_indicator.mm(self.E_I_agg.tolist())
        enc_scores = enc_user_emb.mm(self.item_emb.T.tolist())
        return enc_scores

    def p2mh_plaintext_ref(self, indicator):
        """Plaintext reference for multi-hop P2."""
        if self.score_matrix_multihop is not None:
            return indicator @ self.score_matrix_multihop
        # On-the-fly: indicator @ E_I_agg → user_emb, then @ item_emb.T
        user_emb = indicator @ self.E_I_agg
        return user_emb @ self.item_emb.T

    # ------------------------------------------------------------------ #
    #  Sparse Chunked HE (side-channel-aware)
    # ------------------------------------------------------------------ #

    def _get_slot_capacity(self):
        """Return current CKKS slot capacity (poly_degree // 2)."""
        if self.ctx is None:
            raise RuntimeError("Call setup_context() first")
        # TenSEAL doesn't expose poly_degree directly; we store it.
        return self._slot_capacity

    def setup_context(self, poly_modulus_degree=8192, depth=2):
        """Setup CKKS context with given depth."""
        coeff_mod = [60] + [40] * depth + [60]

        self.ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_mod,
        )
        self.ctx.global_scale = 2 ** 40
        self.ctx.generate_galois_keys()

        self._slot_capacity = poly_modulus_degree // 2
        self._poly_degree = poly_modulus_degree

        n_slots = poly_modulus_degree // 2
        total_bits = sum(coeff_mod)
        print(f"CKKS context: N={poly_modulus_degree}, slots={n_slots}, "
              f"depth={depth}, Q={total_bits} bits")
        return self

    def get_n_chunks(self):
        """Number of chunks needed to cover n_items."""
        S = self._get_slot_capacity()
        return math.ceil(self.n_items / S)

    def get_active_chunk_count(self, indicator):
        """Count non-zero chunks in an indicator vector.

        This is the side-channel: the server observes how many chunks
        the client sends, which correlates with user degree.
        """
        S = self._get_slot_capacity()
        count = 0
        for c in range(self.get_n_chunks()):
            start = c * S
            end = min(start + S, self.n_items)
            if np.any(np.abs(indicator[start:end]) > 1e-15):
                count += 1
        return count

    def get_active_chunk_indices(self, indicator):
        """Return set of chunk indices containing non-zero entries."""
        S = self._get_slot_capacity()
        indices = set()
        for c in range(self.get_n_chunks()):
            start = c * S
            end = min(start + S, self.n_items)
            if np.any(np.abs(indicator[start:end]) > 1e-15):
                indices.add(c)
        return indices

    def sparse_chunked_encrypt(self, indicator):
        """Sparse chunked encryption: skip zero-only chunks.

        Side-channel: number of returned ciphertexts reveals degree info.

        Returns:
            list of (chunk_id, enc_chunk) tuples (only non-zero chunks)
        """
        S = self._get_slot_capacity()
        enc_chunks = []
        for c in range(self.get_n_chunks()):
            start = c * S
            end = min(start + S, self.n_items)
            chunk_data = indicator[start:end]
            if np.any(np.abs(chunk_data) > 1e-15):
                padded = np.zeros(S, dtype=np.float64)
                padded[:end - start] = chunk_data
                enc_chunks.append((c, self._encrypt(padded)))
        return enc_chunks

    def sparse_chunked_encrypt_oblirec(self, indicator, target_n_chunks,
                                        n_decoys=0, rng=None):
        """ObliRec-protected chunked encryption with optional decoy chunks.

        NOTE — latency proxy only: this method pads to target_n_chunks zero
        chunks so that all users in a bucket send exactly target_n_chunks +
        n_decoys ciphertexts, producing uniform per-bucket HE cost.  This is
        used ONLY for the HE timing benchmarks (benchmark_he_timing.py,
        benchmark_e2e.py).

        The paper's formal bucketed mode (§4.2, Definition B.1) submits
        active chunks only (D_u = ∅); chunk-count padding is NOT part of the
        structural leakage model and is NOT used in the audit scripts.

        Args:
            indicator: (n_items,) interaction indicator
            target_n_chunks: fixed number of active chunks (determined per bucket)
            n_decoys: number of additional decoy chunks (default 0 = bucketed mode)
            rng: numpy RandomState for reproducible decoy selection

        Returns:
            list of (chunk_id, enc_chunk) tuples,
            length == target_n_chunks + n_decoys
        """
        S = self._get_slot_capacity()
        n_total = self.get_n_chunks()

        active = []
        inactive_ids = []
        for c in range(n_total):
            start = c * S
            end = min(start + S, self.n_items)
            chunk_data = indicator[start:end]
            if np.any(np.abs(chunk_data) > 1e-15):
                padded = np.zeros(S, dtype=np.float64)
                padded[:end - start] = chunk_data
                active.append((c, padded))
            else:
                inactive_ids.append(c)

        # Pad with zero chunks to reach target_n_chunks
        n_pad = max(0, target_n_chunks - len(active))
        used_inactive = set()
        for c in inactive_ids[:n_pad]:
            active.append((c, np.zeros(S, dtype=np.float64)))
            used_inactive.add(c)

        # Add decoy chunks from remaining inactive indices
        if n_decoys > 0:
            remaining = [c for c in inactive_ids if c not in used_inactive]
            n_actual_decoys = min(n_decoys, len(remaining))
            if n_actual_decoys > 0:
                if rng is None:
                    rng = np.random.RandomState()
                decoy_ids = rng.choice(remaining, size=n_actual_decoys,
                                       replace=False)
                for c in decoy_ids:
                    active.append((c, np.zeros(S, dtype=np.float64)))

        # Sort by chunk id and encrypt
        active.sort(key=lambda x: x[0])
        enc_chunks = [(c, self._encrypt(data)) for c, data in active]
        return enc_chunks

    def sparse_chunked_server_compute(self, enc_chunks, matrix=None):
        """Server-side computation on chunked ciphertexts.

        Computes: sum_c enc_v_c @ M[c*S:(c+1)*S, :] → enc_user_emb

        This is Step 1 of the two-step protocol.  The number of operations
        is proportional to len(enc_chunks), creating the side-channel.

        Args:
            enc_chunks: list of (chunk_id, enc_chunk)
            matrix: (n_items, d) matrix for aggregation.
                    Defaults to self.item_emb (P2) or self.E_I_agg (P2-MH).

        Returns:
            enc_user_emb: encrypted user embedding (d-dimensional)
        """
        if matrix is None:
            matrix = self.E_I_agg if self.E_I_agg is not None else self.item_emb
        S = self._get_slot_capacity()
        d = matrix.shape[1]

        enc_user_emb = None
        for chunk_id, enc_chunk in enc_chunks:
            start = chunk_id * S
            end = min(start + S, self.n_items)
            # Build S × d submatrix (zero-padded)
            submatrix = np.zeros((S, d), dtype=np.float64)
            submatrix[:end - start] = matrix[start:end]
            enc_partial = enc_chunk.mm(submatrix.tolist())
            if enc_user_emb is None:
                enc_user_emb = enc_partial
            else:
                enc_user_emb = enc_user_emb + enc_partial

        return enc_user_emb

    def compute_bucket_chunk_targets(self, oblipack, user_degrees, item_degrees):
        """Pre-compute the target chunk count per bucket.

        For each bucket, the target = max active chunks across all users
        in that bucket.  This ensures within-bucket uniformity.

        Returns:
            dict: bucket_id → target_n_chunks
        """
        S = self._get_slot_capacity()
        bucket_targets = {}

        for bid, users in oblipack.user_buckets.items():
            max_chunks = 0
            for u in users:
                indicator = self.make_indicator(
                    u, oblipack.dataset.train_dict,
                    user_degrees, item_degrees)
                n_active = self.get_active_chunk_count(indicator)
                max_chunks = max(max_chunks, n_active)
            bucket_targets[bid] = max_chunks

        return bucket_targets

    # ------------------------------------------------------------------ #
    #  Utilities
    # ------------------------------------------------------------------ #

    @staticmethod
    def topk(scores, K, mask_items=None):
        """Return top-K item indices from scores, optionally masking items."""
        s = scores.copy()
        if mask_items:
            for i in mask_items:
                s[i] = -np.inf
        return np.argsort(s)[::-1][:K]

    @staticmethod
    def ciphertext_size_kb(enc_vec):
        return len(enc_vec.serialize()) / 1024
