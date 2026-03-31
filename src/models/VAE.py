# coding=utf-8
"""
Variational Autoencoder (VAE) User Embedding Backbone
=======================================================
Replaces the lookup-table user embedding with a VAE encoder.

Architecture:
  Encoder : interaction_profile → μ (d-dim)  +  log_σ² (d-dim)
  z       : μ + ε·σ   during training  (reparameterization trick)
  z       : μ          during eval      (deterministic)

User embedding = μ  →  used for dot-product prediction and PCFR filter.

Training loss = BPR ranking loss  +  β · KL(N(μ,σ²) ‖ N(0,I))
  KL = −½ Σ (1 + log_σ² − μ² − σ²)

No reconstruction loss — avoids the objective conflict between
reconstruction and BPR ranking seen in plain autoencoders.

Classes:
  VAE            – base model
  VAE_PCFR       – + adversarial filter on μ (same arch as BiasedMF_PCFR)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BaseRecModel import BaseRecModel
from utils.constants import LABEL


class VAE(BaseRecModel):

    @staticmethod
    def parse_model_args(parser, model_name='VAE'):
        parser.add_argument('--vae_hidden', type=int, default=256,
                            help='Hidden layer size in VAE encoder.')
        parser.add_argument('--vae_beta', type=float, default=0.1,
                            help='Weight on KL divergence term (β-VAE).')
        return BaseRecModel.parse_model_args(parser, model_name)

    def __init__(self, data_processor_dict, user_num, item_num,
                 u_vector_size, i_vector_size,
                 vae_hidden=256, vae_beta=0.1,
                 random_seed=2020, dropout=0.2,
                 model_path='../model/VAE/VAE.pt'):
        self.vae_hidden = vae_hidden
        self.vae_beta   = vae_beta
        super().__init__(data_processor_dict, user_num, item_num,
                         u_vector_size, i_vector_size,
                         random_seed=random_seed, dropout=dropout,
                         model_path=model_path)
        self._build_user_profiles()

    # ------------------------------------------------------------------
    # Network initialisation
    # ------------------------------------------------------------------

    def _init_nn(self):
        d   = self.u_vector_size
        h   = self.vae_hidden
        n_i = self.item_num

        # Encoder: profile → shared hidden → μ and log_σ²
        self.encoder_hidden = nn.Sequential(
            nn.Linear(n_i, h),
            nn.Tanh(),
        )
        self.fc_mu      = nn.Linear(h, d)
        self.fc_log_var = nn.Linear(h, d)

        # Item lookup embedding (standard)
        self.iid_embeddings = nn.Embedding(self.item_num, d)

    # ------------------------------------------------------------------
    # Build user profile matrix from training interactions
    # ------------------------------------------------------------------

    def _build_user_profiles(self):
        dr       = self.data_processor_dict['train'].data_reader
        train_df = dr.train_df
        profiles = torch.zeros(self.user_num, self.item_num)
        users    = torch.tensor(train_df['uid'].values - 1, dtype=torch.long)
        items    = torch.tensor(train_df['iid'].values - 1, dtype=torch.long)
        profiles[users, items] = 1.0
        # L2-normalise each row so dense and sparse users are on the same scale
        norms = profiles.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self._user_profiles = profiles / norms

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, uids):
        """
        Encode user profiles → (μ, log_σ²).
        uids : 0-indexed LongTensor [batch]
        Returns μ [batch, d], log_var [batch, d]
        """
        device   = self.iid_embeddings.weight.device
        profiles = self._user_profiles[uids].to(device)    # [batch, item_num]
        h        = self.encoder_hidden(profiles)            # [batch, hidden]
        mu       = self.fc_mu(h)                            # [batch, d]
        log_var  = self.fc_log_var(h)                       # [batch, d]
        return mu, log_var

    def reparameterise(self, mu, log_var):
        """Sample z = μ + ε·σ during training; return μ during eval."""
        if self.training:
            std = (0.5 * log_var).exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    # ------------------------------------------------------------------
    # Public accessor for attacker / eval scripts
    # ------------------------------------------------------------------

    def get_user_vectors(self, uids):
        """Return μ (deterministic user embedding) for 0-indexed uid tensor."""
        mu, _ = self.encode(uids)
        return mu

    # ------------------------------------------------------------------
    # Predict / forward
    # ------------------------------------------------------------------

    def predict(self, feed_dict):
        u_ids = feed_dict['X'][:, 0] - 1
        i_ids = feed_dict['X'][:, 1] - 1

        mu, log_var = self.encode(u_ids)
        z           = self.reparameterise(mu, log_var)      # z = μ at eval
        i_emb       = self.iid_embeddings(i_ids)

        prediction = (z * i_emb).sum(dim=1).view([-1])
        return {'prediction': prediction, 'u_vectors': mu, 'check': [],
                '_log_var': log_var}

    def forward(self, feed_dict):
        batch_size = feed_dict[LABEL].shape[0] // 2
        out_dict   = self.predict(feed_dict)

        # BPR ranking loss
        pos = out_dict['prediction'][:batch_size]
        neg = out_dict['prediction'][batch_size:]
        bpr = -(pos - neg).sigmoid().log().sum()

        # KL divergence: −½ Σ (1 + log_σ² − μ² − σ²)
        mu      = out_dict['u_vectors'][:batch_size]
        log_var = out_dict['_log_var'][:batch_size]
        kl      = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=1).mean()

        out_dict['loss'] = bpr + self.vae_beta * kl
        return out_dict


# ──────────────────────────────────────────────────────────────────────────────
# PCFR variant: adversarial filter on top of μ
# ──────────────────────────────────────────────────────────────────────────────

class VAE_PCFR(VAE):
    """
    VAE encoder + PCFR adversarial filter applied on μ.
    Filter architecture identical to BiasedMF_PCFR / GNN_PCFR.
    """

    def _init_nn(self):
        VAE._init_nn(self)
        d = self.u_vector_size
        self.filter = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.LeakyReLU(),
            nn.Linear(d * 2, d),
            nn.LeakyReLU(),
            nn.BatchNorm1d(d),
        )

    def get_user_vectors(self, uids):
        """Return filtered μ for 0-indexed uid tensor."""
        mu, _ = self.encode(uids)
        return self.filter(mu)

    def predict(self, feed_dict):
        u_ids = feed_dict['X'][:, 0] - 1
        i_ids = feed_dict['X'][:, 1] - 1

        mu, log_var  = self.encode(u_ids)
        z            = self.reparameterise(mu, log_var)
        z_filtered   = self.filter(z)                      # filter on sampled z
        i_emb        = self.iid_embeddings(i_ids)

        prediction = (z_filtered * i_emb).sum(dim=1).view([-1])
        return {'prediction': prediction, 'u_vectors': z_filtered, 'check': [],
                '_mu': mu, '_log_var': log_var}

    def forward(self, feed_dict):
        batch_size = feed_dict[LABEL].shape[0] // 2
        out_dict   = self.predict(feed_dict)

        pos = out_dict['prediction'][:batch_size]
        neg = out_dict['prediction'][batch_size:]
        bpr = -(pos - neg).sigmoid().log().sum()

        mu      = out_dict['_mu'][:batch_size]
        log_var = out_dict['_log_var'][:batch_size]
        kl      = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=1).mean()

        out_dict['loss'] = bpr + self.vae_beta * kl
        return out_dict
