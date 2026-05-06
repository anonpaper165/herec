"""
LightGCN model implementation.

Reference: He et al., "LightGCN: Simplifying and Powering Graph Convolution
Network for Recommendation", SIGIR 2020.

Key design:
- No feature transformation, no activation function
- Linear propagation: E^(k+1) = A_hat @ E^(k)
- Final embedding: mean of all layers E = (1/(K+1)) * sum(E^(0)..E^(K))
- BPR loss for training
"""

import torch
import torch.nn as nn


class LightGCN(nn.Module):
    def __init__(self, n_users: int, n_items: int, emb_dim: int = 64, n_layers: int = 3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers

        # initial embeddings (only trainable parameters)
        self.user_embedding = nn.Embedding(n_users, emb_dim)
        self.item_embedding = nn.Embedding(n_items, emb_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def forward(self, adj_matrix: torch.Tensor):
        """Perform K-layer light graph convolution.

        Args:
            adj_matrix: normalized sparse adjacency matrix (n_nodes x n_nodes)

        Returns:
            user_embeddings: (n_users, emb_dim) final user embeddings
            item_embeddings: (n_items, emb_dim) final item embeddings
        """
        # E^(0): concatenate user and item embeddings
        E = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)

        all_layers = [E]  # collect E^(0), E^(1), ..., E^(K)

        for _ in range(self.n_layers):
            E = torch.sparse.mm(adj_matrix, E)  # E^(k+1) = A_hat @ E^(k)
            all_layers.append(E)

        # final embedding: mean pooling across all layers
        E_final = torch.stack(all_layers, dim=0).mean(dim=0)

        user_emb = E_final[:self.n_users]
        item_emb = E_final[self.n_users:]
        return user_emb, item_emb

    def bpr_loss(self, user_emb, item_emb, users, pos_items, neg_items):
        """Bayesian Personalized Ranking loss.

        loss = -log(sigmoid(score_pos - score_neg)) + reg * ||embeddings||^2
        """
        u_emb = user_emb[users]          # (B, d)
        pos_emb = item_emb[pos_items]    # (B, d)
        neg_emb = item_emb[neg_items]    # (B, d)

        pos_scores = (u_emb * pos_emb).sum(dim=1)  # (B,)
        neg_scores = (u_emb * neg_emb).sum(dim=1)  # (B,)

        bpr = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10).mean()

        # L2 regularization on initial embeddings (not propagated ones)
        reg = (
            self.user_embedding(users).norm(2).pow(2)
            + self.item_embedding(pos_items).norm(2).pow(2)
            + self.item_embedding(neg_items).norm(2).pow(2)
        ) / (2 * len(users))

        return bpr, reg
