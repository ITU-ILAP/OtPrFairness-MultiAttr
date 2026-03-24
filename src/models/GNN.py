# coding=utf-8
"""
LightGCN Multi-Graph Embedding backbone, following:
  "Fair Re-Ranking Recommendation Based on Debiased Multi-graph Representations"
  Han et al., ADMA 2023

Three graphs are built from training interactions:
  1. User-item bipartite (heterogeneous)
  2. User-user co-interaction (homogeneous)
  3. Item-item co-user     (homogeneous)

Propagation (L layers):
  Hetero:  h_u^l, h_i^l  ← LightGCN aggregate over user-item graph
  Homo:    S_u^l, S_i^l  ← aggregate over user-user / item-item graphs
  Fusion:  e^l = S^l + h^l
  Final:   e* = mean across all layers (including l=0 initial)

Classes:
  LightGCN_MultiGraph  – base model (dot-product prediction)
  LightGCN_PCFR        – + adversarial filter on user embeddings
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from models.BaseRecModel import BaseRecModel
from utils.constants import LABEL


class LightGCN_MultiGraph(BaseRecModel):

    @staticmethod
    def parse_model_args(parser, model_name='LightGCN_MultiGraph'):
        parser.add_argument('--n_gnn_layers', type=int, default=2,
                            help='Number of GNN propagation layers.')
        return BaseRecModel.parse_model_args(parser, model_name)

    def __init__(self, data_processor_dict, user_num, item_num,
                 u_vector_size, i_vector_size,
                 n_gnn_layers=2, min_common=3, random_seed=2020, dropout=0.2,
                 model_path='../model/LightGCN/LightGCN.pt'):
        self.n_gnn_layers = n_gnn_layers
        self.min_common   = min_common   # min shared items for user-user edge
        self._cache_u = None             # propagation cache for eval mode
        self._cache_i = None
        super().__init__(data_processor_dict, user_num, item_num,
                         u_vector_size, i_vector_size,
                         random_seed=random_seed, dropout=dropout,
                         model_path=model_path)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_nn(self):
        self.uid_embeddings = nn.Embedding(self.user_num, self.u_vector_size)
        self.iid_embeddings = nn.Embedding(self.item_num, self.u_vector_size)
        self._build_graphs()

    def _build_graphs(self):
        """Build and store normalised sparse adjacency matrices."""
        dr       = self.data_processor_dict['train'].data_reader
        train_df = dr.train_df

        # 0-indexed uid / iid
        users = train_df['uid'].values - 1
        items = train_df['iid'].values - 1
        n_u, n_i = self.user_num, self.item_num

        # ── 1. User-item bipartite (LightGCN normalisation) ────────────────
        R = sp.coo_matrix(
            (np.ones(len(users)), (users, items)), shape=(n_u, n_i)
        ).tocsr()

        upper = sp.hstack([sp.csr_matrix((n_u, n_u)), R])
        lower = sp.hstack([R.T,                       sp.csr_matrix((n_i, n_i))])
        A     = sp.vstack([upper, lower]).tocsr()

        d        = np.array(A.sum(1)).flatten()
        d_inv_sq = np.zeros_like(d); d_inv_sq[d > 0] = np.power(d[d > 0], -0.5)
        D        = sp.diags(d_inv_sq)
        A_ui_norm = (D @ A @ D).tocoo()
        self.A_ui = self._sp_to_tensor(A_ui_norm)

        # ── 2. User-user co-interaction (threshold: >= min_common shared items) ─
        A_uu = (R @ R.T).tocsr()
        A_uu.setdiag(0)
        # Keep only edges where users share >= min_common items
        A_uu.data[A_uu.data < self.min_common] = 0
        A_uu.eliminate_zeros()
        d_uu     = np.array(A_uu.sum(1)).flatten()
        d_uu_inv = np.zeros_like(d_uu); d_uu_inv[d_uu > 0] = np.power(d_uu[d_uu > 0], -0.5)
        D_uu     = sp.diags(d_uu_inv)
        A_uu_norm = (D_uu @ A_uu @ D_uu).tocoo()
        self.A_uu = self._sp_to_tensor(A_uu_norm)

        # ── 3. Item-item co-user ────────────────────────────────────────────
        A_ii = (R.T @ R).tocsr()
        A_ii.setdiag(0);  A_ii.eliminate_zeros()
        d_ii     = np.array(A_ii.sum(1)).flatten()
        d_ii_inv = np.zeros_like(d_ii); d_ii_inv[d_ii > 0] = np.power(d_ii[d_ii > 0], -0.5)
        D_ii     = sp.diags(d_ii_inv)
        A_ii_norm = (D_ii @ A_ii @ D_ii).tocoo()
        self.A_ii = self._sp_to_tensor(A_ii_norm)

    @staticmethod
    def _sp_to_tensor(coo):
        idx = torch.LongTensor(np.vstack([coo.row, coo.col]))
        val = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(idx, val, torch.Size(coo.shape)).coalesce()

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------

    def train(self, mode=True):
        """Clear propagation cache when switching to train mode."""
        if mode:
            self._cache_u = None
            self._cache_i = None
        return super().train(mode)

    def _propagate(self):
        """
        Run L-layer multi-graph propagation.
        Returns (all_user_emb, all_item_emb) — averaged across all layers.
        In eval mode the result is cached so batches don't recompute it.
        """
        if not self.training and self._cache_u is not None:
            return self._cache_u, self._cache_i

        device = self.uid_embeddings.weight.device
        A_ui = self.A_ui.to(device)
        A_uu = self.A_uu.to(device)
        A_ii = self.A_ii.to(device)
        n_u  = self.user_num

        S_u = self.uid_embeddings.weight   # [n_u, d]
        S_i = self.iid_embeddings.weight   # [n_i, d]

        agg_u = [S_u]
        agg_i = [S_i]

        for _ in range(self.n_gnn_layers):
            # Heterogeneous propagation (user-item bipartite)
            E      = torch.cat([S_u, S_i], dim=0)        # [n_u+n_i, d]
            E_prop = torch.sparse.mm(A_ui, E)
            h_u    = E_prop[:n_u]                         # [n_u, d]
            h_i    = E_prop[n_u:]                         # [n_i, d]

            # Homogeneous propagation
            S_u = torch.sparse.mm(A_uu, h_u)             # [n_u, d]
            S_i = torch.sparse.mm(A_ii, h_i)             # [n_i, d]

            # Fusion: e^l = S^l + h^l
            agg_u.append(S_u + h_u)
            agg_i.append(S_i + h_i)

        final_u = torch.stack(agg_u, dim=0).mean(dim=0)  # [n_u, d]
        final_i = torch.stack(agg_i, dim=0).mean(dim=0)  # [n_i, d]

        if not self.training:
            self._cache_u = final_u
            self._cache_i = final_i

        return final_u, final_i

    # ------------------------------------------------------------------
    # Public embedding accessor (used by eval / attacker scripts)
    # ------------------------------------------------------------------

    def get_user_vectors(self, uids):
        """
        Return the model's user representation for 0-indexed uid tensor.
        Base model: propagated (GNN) embedding.
        PCFR subclass: propagated + filtered embedding.
        """
        all_u, _ = self._propagate()
        return all_u[uids]

    # ------------------------------------------------------------------
    # Predict / forward
    # ------------------------------------------------------------------

    def predict(self, feed_dict):
        u_ids = feed_dict['X'][:, 0] - 1
        i_ids = feed_dict['X'][:, 1] - 1

        all_u, all_i = self._propagate()
        u_emb = all_u[u_ids]
        i_emb = all_i[i_ids]

        prediction = (u_emb * i_emb).sum(dim=1).view([-1])
        return {'prediction': prediction, 'u_vectors': u_emb, 'check': []}

    def forward(self, feed_dict):
        out_dict   = self.predict(feed_dict)
        batch_size = int(feed_dict[LABEL].shape[0] / 2)
        pos, neg   = out_dict['prediction'][:batch_size], out_dict['prediction'][batch_size:]
        out_dict['loss'] = -(pos - neg).sigmoid().log().sum()
        return out_dict


# ──────────────────────────────────────────────────────────────────────────────
# PCFR variant: adversarial filter on top of propagated user embeddings
# ──────────────────────────────────────────────────────────────────────────────

class LightGCN_PCFR(LightGCN_MultiGraph):
    """
    LightGCN multi-graph embedding + PCFR adversarial filter.
    The filter is identical in architecture to BiasedMF_PCFR.
    """

    def _init_nn(self):
        LightGCN_MultiGraph._init_nn(self)
        d = self.u_vector_size
        self.filter = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.LeakyReLU(),
            nn.Linear(d * 2, d),
            nn.LeakyReLU(),
            nn.BatchNorm1d(d),
        )

    def get_user_vectors(self, uids):
        """Return FILTERED propagated user embeddings for 0-indexed uid tensor."""
        all_u, _ = self._propagate()
        return self.filter(all_u[uids])

    def predict(self, feed_dict):
        u_ids = feed_dict['X'][:, 0] - 1
        i_ids = feed_dict['X'][:, 1] - 1

        all_u, all_i = self._propagate()
        u_emb = self.filter(all_u[u_ids])   # filtered GNN user embedding
        i_emb = all_i[i_ids]

        prediction = (u_emb * i_emb).sum(dim=1).view([-1])
        return {'prediction': prediction, 'u_vectors': u_emb, 'check': []}
