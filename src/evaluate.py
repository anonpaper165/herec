"""Evaluation metrics for top-K recommendation: Recall@K, NDCG@K."""

import torch
import numpy as np


@torch.no_grad()
def evaluate(model, dataset, adj_matrix, K=20, device="cpu"):
    """Evaluate Recall@K and NDCG@K over all test users.

    Args:
        model: LightGCN model
        dataset: ML100KDataset instance
        adj_matrix: normalized adjacency matrix on device
        K: top-K cutoff
        device: torch device

    Returns:
        dict with recall and ndcg
    """
    model.eval()
    user_emb, item_emb = model(adj_matrix)

    recalls, ndcgs = [], []

    for user_id, test_items in dataset.test_dict.items():
        if not test_items:
            continue

        # score all items for this user
        u_emb = user_emb[user_id]                    # (d,)
        scores = item_emb @ u_emb                     # (n_items,)

        # mask out training items (don't recommend already-seen items)
        train_items = list(dataset.user_pos_items.get(user_id, []))
        if train_items:
            scores[train_items] = -float("inf")

        # top-K items
        _, topk_idx = torch.topk(scores, K)
        topk_idx = topk_idx.cpu().numpy()

        test_set = set(test_items)

        # Recall@K
        hits = len(set(topk_idx) & test_set)
        recalls.append(hits / min(len(test_set), K))

        # NDCG@K
        dcg = 0.0
        for rank, item in enumerate(topk_idx):
            if int(item) in test_set:
                dcg += 1.0 / np.log2(rank + 2)  # rank is 0-indexed
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(test_set), K)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return {
        "recall": np.mean(recalls),
        "ndcg": np.mean(ndcgs),
        "n_eval_users": len(recalls),
    }
