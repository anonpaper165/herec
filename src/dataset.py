"""
Unified dataset loader for LightGCN.
Supports: ML-100K, ML-1M (rating-based), Gowalla, Amazon-Book (pre-split).
"""

import os
import numpy as np
import scipy.sparse as sp
import torch


class RecDataset:
    """Unified recommendation dataset for implicit feedback."""

    def __init__(self, data_dir: str, dataset_name: str = "auto",
                 rating_threshold: int = 4, seed: int = 42):
        self.data_dir = data_dir
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        if dataset_name == "auto":
            dataset_name = os.path.basename(data_dir.rstrip("/"))
        self.name = dataset_name

        if dataset_name in ("ml-100k", "ml100k"):
            self._load_ml100k(rating_threshold)
        elif dataset_name in ("ml-1m", "ml1m"):
            self._load_ml1m(rating_threshold)
        elif dataset_name in ("gowalla", "amazon-book", "yelp2018"):
            self._load_lightgcn_format()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        self._build_adjacency()

    # ------------------------------------------------------------------ #
    #  Format-specific loaders
    # ------------------------------------------------------------------ #

    def _load_ml100k(self, threshold):
        path = os.path.join(self.data_dir, "u.data")
        raw = np.loadtxt(path, dtype=np.int32)
        users, items, ratings = raw[:, 0] - 1, raw[:, 1] - 1, raw[:, 2]
        self._from_ratings(users, items, ratings, threshold)

    def _load_ml1m(self, threshold):
        path = os.path.join(self.data_dir, "ratings.dat")
        users, items, ratings = [], [], []
        with open(path) as f:
            for line in f:
                parts = line.strip().split("::")
                users.append(int(parts[0]) - 1)
                items.append(int(parts[1]) - 1)
                ratings.append(int(parts[2]))
        self._from_ratings(np.array(users), np.array(items),
                           np.array(ratings), threshold)

    def _from_ratings(self, users, items, ratings, threshold):
        """Convert explicit ratings to implicit feedback with leave-one-out split."""
        pos_mask = ratings >= threshold
        pos_users, pos_items = users[pos_mask], items[pos_mask]

        # reindex to make IDs contiguous
        unique_u = np.unique(pos_users)
        unique_i = np.unique(pos_items)
        u_map = {old: new for new, old in enumerate(unique_u)}
        i_map = {old: new for new, old in enumerate(unique_i)}
        pos_users = np.array([u_map[u] for u in pos_users])
        pos_items = np.array([i_map[i] for i in pos_items])

        self.n_users = len(unique_u)
        self.n_items = len(unique_i)
        self.n_nodes = self.n_users + self.n_items

        user_items = {}
        for u, i in zip(pos_users, pos_items):
            user_items.setdefault(int(u), []).append(int(i))

        self.train_dict, self.test_dict = {}, {}
        for u, items_list in user_items.items():
            if len(items_list) < 2:
                self.train_dict[u] = items_list
                self.test_dict[u] = []
            else:
                self.train_dict[u] = items_list[:-1]
                self.test_dict[u] = [items_list[-1]]

        self._finalize()

    def _load_lightgcn_format(self):
        """Load pre-split train.txt / test.txt (LightGCN-PyTorch format).

        Each line: user_id item_id1 item_id2 ...
        """
        self.train_dict, self.test_dict = {}, {}
        max_u, max_i = 0, 0

        for split, target in [("train.txt", self.train_dict),
                               ("test.txt", self.test_dict)]:
            path = os.path.join(self.data_dir, split)
            with open(path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue
                    uid = int(parts[0])
                    items = [int(x) for x in parts[1:]]
                    target[uid] = items
                    max_u = max(max_u, uid)
                    if items:
                        max_i = max(max_i, max(items))

        self.n_users = max_u + 1
        self.n_items = max_i + 1
        self.n_nodes = self.n_users + self.n_items
        self._finalize()

    # ------------------------------------------------------------------ #
    #  Common setup
    # ------------------------------------------------------------------ #

    def _finalize(self):
        train_u, train_i = [], []
        for u, items in self.train_dict.items():
            for i in items:
                train_u.append(u)
                train_i.append(i)

        self.train_users = np.array(train_u, dtype=np.int64)
        self.train_items = np.array(train_i, dtype=np.int64)
        self.n_train = len(self.train_users)

        self.user_pos_items = {u: set(its) for u, its in self.train_dict.items()}

        n_test = sum(len(v) for v in self.test_dict.values())
        print(f"[{self.name}] {self.n_users} users, {self.n_items} items, "
              f"{self.n_train} train, {n_test} test")

    def _build_adjacency(self):
        rows, cols = self.train_users, self.train_items
        vals = np.ones(self.n_train, dtype=np.float32)
        R = sp.coo_matrix((vals, (rows, cols)), shape=(self.n_users, self.n_items))

        zero_uu = sp.csr_matrix((self.n_users, self.n_users), dtype=np.float32)
        zero_ii = sp.csr_matrix((self.n_items, self.n_items), dtype=np.float32)
        A = sp.bmat([[zero_uu, R], [R.T, zero_ii]], format="coo")

        degrees = np.array(A.sum(axis=1)).flatten()
        d_inv_sqrt = np.where(degrees > 0, np.power(degrees, -0.5), 0.0)
        D_inv_sqrt = sp.diags(d_inv_sqrt)
        A_hat = (D_inv_sqrt @ A @ D_inv_sqrt).tocoo()

        indices = torch.LongTensor(np.stack([A_hat.row, A_hat.col]))
        values = torch.FloatTensor(A_hat.data)
        self.adj_matrix = torch.sparse_coo_tensor(indices, values, A_hat.shape).coalesce()

    def sample_negative(self, user: int) -> int:
        pos = self.user_pos_items.get(user, set())
        while True:
            neg = self.rng.randint(0, self.n_items)
            if neg not in pos:
                return neg

    def get_train_batch(self, batch_size: int):
        idx = self.rng.randint(0, self.n_train, size=batch_size)
        users = self.train_users[idx]
        pos_items = self.train_items[idx]
        neg_items = np.array([self.sample_negative(int(u)) for u in users], dtype=np.int64)
        return (torch.LongTensor(users), torch.LongTensor(pos_items),
                torch.LongTensor(neg_items))


# Backward compatibility alias
def ML100KDataset(data_dir, rating_threshold=4):
    return RecDataset(data_dir, dataset_name="ml-100k",
                      rating_threshold=rating_threshold)
