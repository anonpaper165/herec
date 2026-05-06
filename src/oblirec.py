"""
ObliRec: Oblivious Recommendation for privacy-preserving LightGCN inference.

Implements three core mechanisms:
1. Degree-based bucketing - user-side supports privacy-aware adaptive (DP)
   or log-scale; item-side always uses log-scale
2. Dummy padding - pads neighbor lists to uniform bucket size with zero-weight entries
3. Intra-bucket shuffling - random shuffle within buckets and neighbor lists

The bucketed dummy-padding operation preserves exact LightGCN aggregation
semantics for the padded propagation path because dummy entries have zero
weight. This applies to the packing component; the P2-MH serving formulation
evaluated in the paper is a separate protocol with its own fidelity analysis.
"""

import math
import numpy as np
import torch
from collections import defaultdict


class ObliRec:
    def __init__(self, dataset, seed=42, bucketing="log",
                 n_buckets=None, min_anon_set=10, privacy_weight=0.0):
        """
        Args:
            dataset: RecDataset instance
            seed: random seed
            bucketing: "log" (default power-of-2) or "adaptive" (DP-optimal)
            n_buckets: number of buckets for adaptive bucketing (auto if None)
            min_anon_set: minimum users per bucket for adaptive bucketing
            privacy_weight: λ for privacy-aware adaptive bucketing on the
                user side only (0 = padding-only DP, >0 adds N/|B_k|
                anonymity penalty to the DP objective). Item-side bucketing
                always uses log-scale regardless of this parameter.
        """
        self.dataset = dataset
        self.rng = np.random.RandomState(seed)
        self.bucketing = bucketing
        self.min_anon_set = min_anon_set
        self.privacy_weight = privacy_weight

        # dummy indices (appended after real nodes)
        self.dummy_item_idx = dataset.n_items
        self.dummy_user_idx = dataset.n_users

        # computed in _build()
        self.user_buckets = {}   # bucket_id → [user_ids]
        self.item_buckets = {}   # bucket_id → [item_ids]

        # per-node padded structures
        self.user_padded_neighbors = {}  # user → [item indices, padded]
        self.user_padded_weights = {}    # user → [float weights, padded]
        self.item_padded_neighbors = {}  # item → [user indices, padded]
        self.item_padded_weights = {}    # item → [float weights, padded]

        # degree info
        self.user_degrees = {}
        self.item_degrees = {}

        # adaptive bucketing boundaries (set by _build if bucketing="adaptive")
        self._adaptive_boundaries = None  # list of (lo, hi) tuples
        self._adaptive_pad_targets = None  # list of pad targets per bucket

        self._build(n_buckets=n_buckets)

    # ------------------------------------------------------------------ #
    #  Bucketing helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def bucket_id(degree: int) -> int:
        """Map degree → log-scale bucket ID.

        Bucket 0: degree 1       → pad to 1
        Bucket 1: degree 2       → pad to 2
        Bucket k: degree (2^(k-1)+1 .. 2^k) → pad to 2^k
        """
        if degree <= 0:
            return -1
        if degree == 1:
            return 0
        return int(math.ceil(math.log2(degree)))

    @staticmethod
    def bucket_max(bucket_id: int) -> int:
        """Padding target size for a bucket."""
        if bucket_id <= 0:
            return 1
        return 2 ** bucket_id

    @staticmethod
    def bucket_range(bucket_id: int):
        """Human-readable degree range for a bucket."""
        if bucket_id == 0:
            return (1, 1)
        lo = 2 ** (bucket_id - 1) + 1
        hi = 2 ** bucket_id
        return (lo, hi)

    # ------------------------------------------------------------------ #
    #  Adaptive Bucketing via Dynamic Programming
    # ------------------------------------------------------------------ #

    @staticmethod
    def optimal_buckets(degrees, n_buckets, min_anon_set=10):
        """Find optimal bucket boundaries minimizing total padding cost.

        Works on unique degree values with counts for efficiency.
        DP on unique degrees: O(D^2 * K) where D = #unique degrees << N.

        Returns:
            boundaries: list of (start_idx, end_idx) pairs into sorted unique degrees
            pad_targets: list of padding target per bucket
            total_cost: total padding cost
        """
        from collections import Counter

        deg_counts = Counter(degrees)
        unique_degs = sorted(deg_counts.keys())
        D = len(unique_degs)
        counts = [deg_counts[d] for d in unique_degs]

        K = min(n_buckets, D)

        if D <= K:
            boundaries = [(i, i) for i in range(D)]
            pad_targets = unique_degs[:]
            total_cost = sum(
                counts[i] * (unique_degs[i] - unique_degs[i])
                for i in range(D)
            )
            return boundaries, pad_targets, total_cost

        # Prefix sums for O(1) cost computation
        # cost of grouping unique_degs[i..j] = sum over k in [i,j]:
        #   counts[k] * (unique_degs[j] - unique_degs[k])
        # = unique_degs[j] * sum(counts[i..j]) - sum(counts[k]*unique_degs[k] for k in [i,j])
        prefix_count = [0] * (D + 1)
        prefix_weighted = [0] * (D + 1)
        for i in range(D):
            prefix_count[i + 1] = prefix_count[i] + counts[i]
            prefix_weighted[i + 1] = prefix_weighted[i] + counts[i] * unique_degs[i]

        def bucket_cost(i, j):
            total_count = prefix_count[j + 1] - prefix_count[i]
            weighted_sum = prefix_weighted[j + 1] - prefix_weighted[i]
            return unique_degs[j] * total_count - weighted_sum

        def bucket_user_count(i, j):
            return prefix_count[j + 1] - prefix_count[i]

        INF = float('inf')

        # dp[j] = min cost for unique_degs[0..j] using current k buckets
        dp = [INF] * D
        parent = [[0] * D for _ in range(K)]

        # Base case: k=1
        for j in range(D):
            if bucket_user_count(0, j) >= min_anon_set:
                dp[j] = bucket_cost(0, j)

        # Fill k=2..K
        for k in range(2, K + 1):
            new_dp = [INF] * D
            for j in range(k - 1, D):
                for i in range(k - 2, j):
                    if dp[i] == INF:
                        continue
                    uc = bucket_user_count(i + 1, j)
                    if uc < min_anon_set:
                        continue
                    c = dp[i] + bucket_cost(i + 1, j)
                    if c < new_dp[j]:
                        new_dp[j] = c
                        parent[k - 1][j] = i
            dp = new_dp

        if dp[D - 1] == INF:
            return ObliRec.optimal_buckets(degrees, n_buckets - 1,
                                            min_anon_set)

        # Backtrack
        boundaries = []
        j = D - 1
        for k in range(K, 1, -1):
            i = parent[k - 1][j]
            boundaries.append((i + 1, j))
            j = i
        boundaries.append((0, j))
        boundaries.reverse()

        pad_targets = [unique_degs[end] for start, end in boundaries]
        total_cost = dp[D - 1]

        return boundaries, pad_targets, total_cost

    @staticmethod
    def optimal_buckets_privacy(degrees, n_buckets, min_anon_set=10,
                                privacy_weight=1.0):
        """Find optimal bucket boundaries balancing padding cost and privacy.

        Objective per bucket:
            padding_cost(bucket) + privacy_weight * N / |bucket|

        The privacy term penalizes small buckets (weak anonymity sets).
        When privacy_weight=0, reduces to padding-only optimization.
        When privacy_weight→∞, prefers equal-sized buckets.

        Same DP structure as optimal_buckets, O(D^2 * K).
        """
        from collections import Counter

        deg_counts = Counter(degrees)
        unique_degs = sorted(deg_counts.keys())
        D = len(unique_degs)
        counts = [deg_counts[d] for d in unique_degs]
        N = sum(counts)

        K = min(n_buckets, D)

        if D <= K:
            boundaries = [(i, i) for i in range(D)]
            pad_targets = unique_degs[:]
            total_cost = sum(privacy_weight * N / max(counts[i], 1)
                             for i in range(D))
            return boundaries, pad_targets, total_cost

        # Prefix sums
        prefix_count = [0] * (D + 1)
        prefix_weighted = [0] * (D + 1)
        for i in range(D):
            prefix_count[i + 1] = prefix_count[i] + counts[i]
            prefix_weighted[i + 1] = (prefix_weighted[i]
                                      + counts[i] * unique_degs[i])

        def padding_cost(i, j):
            total_count = prefix_count[j + 1] - prefix_count[i]
            weighted_sum = prefix_weighted[j + 1] - prefix_weighted[i]
            return unique_degs[j] * total_count - weighted_sum

        def bucket_user_count(i, j):
            return prefix_count[j + 1] - prefix_count[i]

        def combined_cost(i, j):
            uc = bucket_user_count(i, j)
            pc = padding_cost(i, j)
            privacy_penalty = privacy_weight * N / max(uc, 1)
            return pc + privacy_penalty

        INF = float('inf')

        dp = [INF] * D
        parent = [[0] * D for _ in range(K)]

        # Base case: k=1
        for j in range(D):
            if bucket_user_count(0, j) >= min_anon_set:
                dp[j] = combined_cost(0, j)

        # Fill k=2..K
        for k in range(2, K + 1):
            new_dp = [INF] * D
            for j in range(k - 1, D):
                for i in range(k - 2, j):
                    if dp[i] == INF:
                        continue
                    uc = bucket_user_count(i + 1, j)
                    if uc < min_anon_set:
                        continue
                    c = dp[i] + combined_cost(i + 1, j)
                    if c < new_dp[j]:
                        new_dp[j] = c
                        parent[k - 1][j] = i
            dp = new_dp

        if dp[D - 1] == INF:
            return ObliRec.optimal_buckets_privacy(
                degrees, n_buckets - 1, min_anon_set, privacy_weight)

        # Backtrack
        boundaries = []
        j = D - 1
        for k in range(K, 1, -1):
            i = parent[k - 1][j]
            boundaries.append((i + 1, j))
            j = i
        boundaries.append((0, j))
        boundaries.reverse()

        pad_targets = [unique_degs[end] for start, end in boundaries]
        total_cost = dp[D - 1]

        return boundaries, pad_targets, total_cost

    # ------------------------------------------------------------------ #
    #  Build packing structures
    # ------------------------------------------------------------------ #

    def _build(self, n_buckets=None):
        ds = self.dataset

        # --- compute degrees and neighbor lists ---
        user_neighbors = {}  # user → [items]
        item_neighbors = defaultdict(list)  # item → [users]

        for u, items in ds.train_dict.items():
            user_neighbors[u] = list(items)
            self.user_degrees[u] = len(items)
            for i in items:
                item_neighbors[i].append(u)

        for i in item_neighbors:
            self.item_degrees[i] = len(item_neighbors[i])

        if self.bucketing == "adaptive":
            self._build_adaptive_buckets(n_buckets)
        else:
            # --- log bucketing (default) ---
            for u, deg in self.user_degrees.items():
                bid = self.bucket_id(deg)
                if bid < 0:
                    continue
                self.user_buckets.setdefault(bid, []).append(u)

        # --- bucket items (always log-scale for items) ---
        for i, deg in self.item_degrees.items():
            bid = self.bucket_id(deg)
            if bid < 0:
                continue
            self.item_buckets.setdefault(bid, []).append(i)

        # --- pad & permute user neighbor lists ---
        for bid, users in self.user_buckets.items():
            if self.bucketing == "adaptive" and self._adaptive_pad_targets:
                pad_size = self.adaptive_bucket_max(bid)
            else:
                pad_size = self.bucket_max(bid)
            self.rng.shuffle(users)  # intra-bucket permutation

            for u in users:
                neighbors = user_neighbors.get(u, [])
                d_u = self.user_degrees[u]

                # normalization weights: 1/sqrt(d_u * d_i)
                weights = []
                for i in neighbors:
                    d_i = self.item_degrees.get(i, 1)
                    weights.append(1.0 / math.sqrt(d_u * d_i))

                # dummy padding
                n_pad = pad_size - len(neighbors)
                neighbors_pad = list(neighbors) + [self.dummy_item_idx] * n_pad
                weights_pad = weights + [0.0] * n_pad

                # neighbor-list permutation
                perm = self.rng.permutation(pad_size)
                self.user_padded_neighbors[u] = [neighbors_pad[p] for p in perm]
                self.user_padded_weights[u] = [weights_pad[p] for p in perm]

        # --- pad & permute item neighbor lists ---
        for bid, items in self.item_buckets.items():
            pad_size = self.bucket_max(bid)
            self.rng.shuffle(items)

            for i in items:
                neighbors = item_neighbors.get(i, [])
                d_i = self.item_degrees[i]

                weights = []
                for u in neighbors:
                    d_u = self.user_degrees.get(u, 1)
                    weights.append(1.0 / math.sqrt(d_i * d_u))

                n_pad = pad_size - len(neighbors)
                neighbors_pad = list(neighbors) + [self.dummy_user_idx] * n_pad
                weights_pad = weights + [0.0] * n_pad

                perm = self.rng.permutation(pad_size)
                self.item_padded_neighbors[i] = [neighbors_pad[p] for p in perm]
                self.item_padded_weights[i] = [weights_pad[p] for p in perm]

    def _build_adaptive_buckets(self, n_buckets=None):
        """Build user buckets using DP-optimal boundaries."""
        from collections import Counter

        all_degrees = [self.user_degrees[u] for u in self.user_degrees
                       if self.user_degrees[u] > 0]

        if n_buckets is None:
            max_deg = max(all_degrees) if all_degrees else 1
            n_buckets = max(1, int(math.ceil(math.log2(max_deg))) + 1)

        if self.privacy_weight > 0:
            boundaries, pad_targets, total_cost = self.optimal_buckets_privacy(
                all_degrees, n_buckets, self.min_anon_set, self.privacy_weight)
        else:
            boundaries, pad_targets, total_cost = self.optimal_buckets(
                all_degrees, n_buckets, self.min_anon_set)

        unique_degs = sorted(Counter(all_degrees).keys())

        # Build degree → bucket mapping
        self._adaptive_boundaries = []
        self._adaptive_pad_targets = []
        deg_to_bucket = {}

        for bid, ((start, end), pad_target) in enumerate(
                zip(boundaries, pad_targets)):
            lo_deg = unique_degs[start]
            hi_deg = unique_degs[end]
            self._adaptive_boundaries.append((lo_deg, hi_deg))
            self._adaptive_pad_targets.append(pad_target)
            for idx in range(start, end + 1):
                deg_to_bucket[unique_degs[idx]] = bid

        # Assign users to buckets
        for u, deg in self.user_degrees.items():
            if deg <= 0:
                continue
            bid = deg_to_bucket.get(deg)
            if bid is None:
                for b, (lo, hi) in enumerate(self._adaptive_boundaries):
                    if lo <= deg <= hi:
                        bid = b
                        break
                if bid is None:
                    bid = len(self._adaptive_boundaries) - 1
            self.user_buckets.setdefault(bid, []).append(u)

    def adaptive_bucket_id(self, degree):
        """Map degree → bucket ID using adaptive boundaries."""
        if self._adaptive_boundaries is None:
            return self.bucket_id(degree)
        for bid, (lo, hi) in enumerate(self._adaptive_boundaries):
            if degree <= hi:
                return bid
        return len(self._adaptive_boundaries) - 1

    def adaptive_bucket_max(self, bucket_id):
        """Padding target for adaptive bucket."""
        if self._adaptive_pad_targets is None:
            return self.bucket_max(bucket_id)
        if 0 <= bucket_id < len(self._adaptive_pad_targets):
            return self._adaptive_pad_targets[bucket_id]
        return self._adaptive_pad_targets[-1]

    def adaptive_bucket_range(self, bucket_id):
        """Degree range for adaptive bucket."""
        if self._adaptive_boundaries is None:
            return self.bucket_range(bucket_id)
        if 0 <= bucket_id < len(self._adaptive_boundaries):
            return self._adaptive_boundaries[bucket_id]
        return self._adaptive_boundaries[-1]

    # ------------------------------------------------------------------ #
    #  Batched propagation (mimics HE computation path)
    # ------------------------------------------------------------------ #

    def forward(self, user_emb_init, item_emb_init, n_layers=3):
        """Run LightGCN propagation using ObliRec padded structures.

        Each bucket is processed as a dense batch:
          - All nodes in the bucket have the same neighbor list size
          - Dummy entries have weight=0, contributing nothing to aggregation
          - This mirrors how HE would process fixed-size ciphertext blocks

        Returns (user_emb_final, item_emb_final) identical to vanilla LightGCN.
        """
        device = user_emb_init.device
        n_users = self.dataset.n_users
        n_items = self.dataset.n_items
        emb_dim = user_emb_init.shape[1]

        user_emb = user_emb_init
        item_emb = item_emb_init

        all_user = [user_emb]
        all_item = [item_emb]

        for layer in range(n_layers):
            # --- users aggregate from items ---
            # extend item embeddings with dummy (zero) at index n_items
            item_ext = torch.cat([item_emb, torch.zeros(1, emb_dim, device=device)])
            new_user = torch.zeros(n_users, emb_dim, device=device)

            for bid, users in self.user_buckets.items():
                B = len(users)
                if self.bucketing == "adaptive" and self._adaptive_pad_targets:
                    pad_size = self.adaptive_bucket_max(bid)
                else:
                    pad_size = self.bucket_max(bid)

                # (B, pad_size) index and weight tensors
                idx = torch.LongTensor(
                    [self.user_padded_neighbors[u] for u in users]
                ).to(device)
                w = torch.FloatTensor(
                    [self.user_padded_weights[u] for u in users]
                ).to(device)

                # gather: (B, pad_size, d)
                nbr_emb = item_ext[idx]
                # weighted sum: (B, d)
                agg = (w.unsqueeze(2) * nbr_emb).sum(dim=1)

                uid = torch.LongTensor(users).to(device)
                new_user.index_copy_(0, uid, agg)

            # --- items aggregate from users ---
            user_ext = torch.cat([user_emb, torch.zeros(1, emb_dim, device=device)])
            new_item = torch.zeros(n_items, emb_dim, device=device)

            for bid, items in self.item_buckets.items():
                B = len(items)
                pad_size = self.bucket_max(bid)

                idx = torch.LongTensor(
                    [self.item_padded_neighbors[i] for i in items]
                ).to(device)
                w = torch.FloatTensor(
                    [self.item_padded_weights[i] for i in items]
                ).to(device)

                nbr_emb = user_ext[idx]
                agg = (w.unsqueeze(2) * nbr_emb).sum(dim=1)

                iid = torch.LongTensor(items).to(device)
                new_item.index_copy_(0, iid, agg)

            user_emb = new_user
            item_emb = new_item
            all_user.append(user_emb)
            all_item.append(item_emb)

        # mean pooling across layers
        final_user = torch.stack(all_user, dim=0).mean(dim=0)
        final_item = torch.stack(all_item, dim=0).mean(dim=0)

        return final_user, final_item

    # ------------------------------------------------------------------ #
    #  Statistics & privacy analysis
    # ------------------------------------------------------------------ #

    def print_stats(self):
        """Print bucketing statistics and storage overhead analysis."""
        ds = self.dataset

        print("=" * 65)
        print("  ObliRec Packing Statistics")
        print("=" * 65)

        # --- user buckets ---
        print("\n[User Buckets]")
        print(f"  {'Bucket':>6} {'Range':>10} {'Pad→':>6} {'#Users':>7} "
              f"{'Real':>7} {'Padded':>8} {'Overhead':>9}")
        print("  " + "-" * 57)

        total_real_u, total_pad_u = 0, 0
        for bid in sorted(self.user_buckets.keys()):
            users = self.user_buckets[bid]
            if self.bucketing == "adaptive" and self._adaptive_pad_targets:
                pad_size = self.adaptive_bucket_max(bid)
                lo, hi = self.adaptive_bucket_range(bid)
            else:
                pad_size = self.bucket_max(bid)
                lo, hi = self.bucket_range(bid)
            real = sum(self.user_degrees[u] for u in users)
            padded = len(users) * pad_size
            total_real_u += real
            total_pad_u += padded
            overhead = padded / max(real, 1)
            print(f"  B{bid:>4d}  [{lo:>4d}-{hi:>4d}] {pad_size:>5d}  {len(users):>6d} "
                  f"{real:>7d} {padded:>8d}  {overhead:>7.2f}x")

        blowup_u = total_pad_u / max(total_real_u, 1)
        print(f"\n  User total: real={total_real_u}, padded={total_pad_u}, "
              f"blowup={blowup_u:.2f}x")

        # --- item buckets ---
        print("\n[Item Buckets]")
        print(f"  {'Bucket':>6} {'Range':>10} {'Pad→':>6} {'#Items':>7} "
              f"{'Real':>7} {'Padded':>8} {'Overhead':>9}")
        print("  " + "-" * 57)

        total_real_i, total_pad_i = 0, 0
        for bid in sorted(self.item_buckets.keys()):
            items = self.item_buckets[bid]
            pad_size = self.bucket_max(bid)
            lo, hi = self.bucket_range(bid)
            real = sum(self.item_degrees[i] for i in items)
            padded = len(items) * pad_size
            total_real_i += real
            total_pad_i += padded
            overhead = padded / max(real, 1)
            print(f"  B{bid:>4d}  [{lo:>4d}-{hi:>4d}] {pad_size:>5d}  {len(items):>6d} "
                  f"{real:>7d} {padded:>8d}  {overhead:>7.2f}x")

        blowup_i = total_pad_i / max(total_real_i, 1)
        print(f"\n  Item total: real={total_real_i}, padded={total_pad_i}, "
              f"blowup={blowup_i:.2f}x")

        # --- degree hiding analysis ---
        print("\n[Privacy: Degree Hiding]")
        unique_user_degs = len(set(self.user_degrees.values()))
        n_user_buckets = len(self.user_buckets)
        unique_item_degs = len(set(self.item_degrees.values()))
        n_item_buckets = len(self.item_buckets)

        print(f"  User: {unique_user_degs} unique degrees → {n_user_buckets} buckets "
              f"(anonymity set ≥ {min(len(v) for v in self.user_buckets.values())})")
        print(f"  Item: {unique_item_degs} unique degrees → {n_item_buckets} buckets "
              f"(anonymity set ≥ {min(len(v) for v in self.item_buckets.values())})")

        # per-bucket anonymity set sizes
        print("\n  User bucket anonymity sets:")
        for bid in sorted(self.user_buckets.keys()):
            if self.bucketing == "adaptive" and self._adaptive_boundaries:
                lo, hi = self.adaptive_bucket_range(bid)
            else:
                lo, hi = self.bucket_range(bid)
            k = len(self.user_buckets[bid])
            print(f"    B{bid} [{lo}-{hi}]: {k} users indistinguishable")

        print("=" * 65)
