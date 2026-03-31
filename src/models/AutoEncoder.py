# coding=utf-8
"""
Autoencoder User Embedding Backbone
=====================================
Replaces the lookup-table user embedding with a denoising autoencoder.

Architecture:
  - Encoder : item_num → hidden → u_vector_size   (ReLU activations)
  - Decoder : u_vector_size → hidden → item_num   (Sigmoid output)
  - Input   : user interaction profile (binary vector of length item_num)
  - User embedding = encoder bottleneck output

The encoder bottleneck is used as the user representation for dot-product
prediction against a standard item lookup embedding.

Training loss = BPR ranking loss + recon_weight × BCE reconstruction loss

Classes:
  AutoEncoder       – base model
  AutoEncoder_PCFR  – + adversarial filter on the bottleneck (same arch as BiasedMF_PCFR)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BaseRecModel import BaseRecModel
from utils.constants import LABEL


class AutoEncoder(BaseRecModel):

    @staticmethod
    def parse_model_args(parser, model_name='AutoEncoder'):
        parser.add_argument('--ae_hidden', type=int, default=256,
                            help='Hidden layer size in autoencoder encoder/decoder.')
        parser.add_argument('--recon_weight', type=float, default=0.1,
                            help='Weight on reconstruction loss relative to BPR loss.')
        parser.add_argument('--ae_dropout', type=float, default=0.5,
                            help='Input dropout rate for denoising (corruption noise).')
        return BaseRecModel.parse_model_args(parser, model_name)

    def __init__(self, data_processor_dict, user_num, item_num,
                 u_vector_size, i_vector_size,
                 ae_hidden=256, recon_weight=0.1, ae_dropout=0.5,
                 random_seed=2020, dropout=0.2,
                 model_path='../model/AutoEncoder/AutoEncoder.pt'):
        self.ae_hidden     = ae_hidden
        self.recon_weight  = recon_weight
        self.ae_dropout    = ae_dropout
        self._user_profiles = None   # built lazily after _init_nn
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
        h   = self.ae_hidden
        n_i = self.item_num

        # Encoder: item_num → hidden → bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(n_i, h),
            nn.ReLU(),
            nn.Linear(h, d),
            nn.ReLU(),
        )

        # Decoder: bottleneck → hidden → item_num
        self.decoder = nn.Sequential(
            nn.Linear(d, h),
            nn.ReLU(),
            nn.Linear(h, n_i),
        )

        # Item lookup embedding (standard)
        self.iid_embeddings = nn.Embedding(self.item_num, d)

        # Input dropout (denoising)
        self.input_dropout = nn.Dropout(p=self.ae_dropout)

    # ------------------------------------------------------------------
    # Build user profile matrix from training interactions
    # ------------------------------------------------------------------

    def _build_user_profiles(self):
        """
        Build a dense [user_num, item_num] binary matrix from training data.
        Rows are 0-indexed users; columns are 0-indexed items.
        """
        dr       = self.data_processor_dict['train'].data_reader
        train_df = dr.train_df

        profiles = torch.zeros(self.user_num, self.item_num)
        users = torch.tensor(train_df['uid'].values - 1, dtype=torch.long)
        items = torch.tensor(train_df['iid'].values - 1, dtype=torch.long)
        profiles[users, items] = 1.0
        self._user_profiles = profiles   # stays on CPU; moved to device in encode()

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, uids):
        """
        Encode user interaction profiles to latent vectors.
        uids: 0-indexed LongTensor of shape [batch]
        Returns: [batch, u_vector_size]
        """
        device   = self.iid_embeddings.weight.device
        profiles = self._user_profiles[uids].to(device)   # [batch, item_num]
        if self.training:
            profiles = self.input_dropout(profiles)        # denoising
        return self.encoder(profiles)                      # [batch, d]

    # ------------------------------------------------------------------
    # Public accessor for attacker / eval scripts
    # ------------------------------------------------------------------

    def get_user_vectors(self, uids):
        """Return raw encoder output for 0-indexed uid tensor."""
        return self.encode(uids)

    # ------------------------------------------------------------------
    # Predict / forward
    # ------------------------------------------------------------------

    def predict(self, feed_dict):
        u_ids = feed_dict['X'][:, 0] - 1
        i_ids = feed_dict['X'][:, 1] - 1

        u_emb = self.encode(u_ids)                        # [batch, d]
        i_emb = self.iid_embeddings(i_ids)                # [batch, d]

        prediction = (u_emb * i_emb).sum(dim=1).view([-1])
        return {'prediction': prediction, 'u_vectors': u_emb, 'check': []}

    def forward(self, feed_dict):
        u_ids      = feed_dict['X'][:, 0] - 1
        batch_size = feed_dict[LABEL].shape[0] // 2

        out_dict = self.predict(feed_dict)

        # BPR ranking loss
        pos   = out_dict['prediction'][:batch_size]
        neg   = out_dict['prediction'][batch_size:]
        bpr   = -(pos - neg).sigmoid().log().sum()

        # Reconstruction loss (positive half only — use clean profile as target)
        device   = self.iid_embeddings.weight.device
        profiles = self._user_profiles[u_ids[:batch_size]].to(device)
        u_emb    = out_dict['u_vectors'][:batch_size]
        recon    = self.decoder(u_emb)
        recon_loss = F.binary_cross_entropy_with_logits(recon, profiles)

        out_dict['loss'] = bpr + self.recon_weight * recon_loss
        return out_dict


# ──────────────────────────────────────────────────────────────────────────────
# PCFR variant: adversarial filter on top of the encoder bottleneck
# ──────────────────────────────────────────────────────────────────────────────

class AutoEncoder_PCFR(AutoEncoder):
    """
    AutoEncoder + PCFR adversarial filter.
    The filter architecture is identical to BiasedMF_PCFR / GNN_PCFR.
    """

    def _init_nn(self):
        AutoEncoder._init_nn(self)
        d = self.u_vector_size
        self.filter = nn.Sequential(
            nn.Linear(d, d * 2),
            nn.LeakyReLU(),
            nn.Linear(d * 2, d),
            nn.LeakyReLU(),
            nn.BatchNorm1d(d),
        )

    def get_user_vectors(self, uids):
        """Return filtered encoder output for 0-indexed uid tensor."""
        return self.filter(self.encode(uids))

    def predict(self, feed_dict):
        u_ids = feed_dict['X'][:, 0] - 1
        i_ids = feed_dict['X'][:, 1] - 1

        u_emb = self.filter(self.encode(u_ids))    # filtered bottleneck
        i_emb = self.iid_embeddings(i_ids)

        prediction = (u_emb * i_emb).sum(dim=1).view([-1])
        return {'prediction': prediction, 'u_vectors': u_emb, 'check': []}
