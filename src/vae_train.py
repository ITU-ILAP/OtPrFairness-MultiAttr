# coding=utf-8
"""
VAE-PCFR Training Script
==================================
Trains VVAE_PCFR on both insurance and ml100k:
  • None baseline        (no adversarial training)
  • Single-attribute PCFR × 4 attributes per dataset  (Q1 checkpoints)
  • Multi-attribute PCFR on all 4 attributes at once   (Q2 checkpoint)

Checkpoints saved to:
  ../model/VAE_None_insurance/model.pt
  ../model/VAE_PCFR_insurance_{attr}/model.pt
  ../model/VAE_PCFR_multiattr_insurance/model.pt
  ../model/VAE_None_ml100k/model.pt
  ../model/VAE_PCFR_ml100k_{attr}/model.pt
  ../model/VAE_PCFR_multiattr_ml100k/model.pt

Usage (from src/):
    python autoencoder_train.py
    python autoencoder_train.py --dataset insurance   # single dataset
    python autoencoder_train.py --dataset ml100k
"""

import os
import sys
import gc
import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.ERROR)

from data_reader import RecDataReader, DiscriminatorDataReader
from datasets import RecDataset, DiscriminatorDataset
from models.VAE import VAE, VVAE_PCFR
from models.Discriminators import BinaryDiscriminator
from utils.generic import batch_to_gpu
from utils.constants import LABEL

# ── Config ─────────────────────────────────────────────────────────────────────

DATASET_ATTRS = {
    'insurance': ['u_gender', 'u_occupation', 'u_activity', 'u_marital_status'],
    'ml100k':    ['u_gender', 'u_age', 'u_occupation', 'u_activity'],
}

DATA_PATH   = '../dataset/'
EPOCHS      = 100
LR          = 1e-3
L2          = 1e-4
BATCH_SIZE  = 1024
VT_NUM_NEG  = 10
U_VEC_SIZE  = 64
VAE_HIDDEN  = 256
VAE_BETA    = 0.1
RANDOM_SEED = 2020
NUM_WORKER  = 0
DISC_LR     = 1e-3
DISC_L2     = 1e-4
ADV_WEIGHT  = 1.0
MODEL_DIR   = '../model/'


# ── Helpers ────────────────────────────────────────────────────────────────────

def evaluate_rec(model, loader):
    """NDCG@3 proxy for checkpoint selection."""
    model.eval()
    hit, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            batch  = batch_to_gpu(batch)
            preds  = model.predict(batch)['prediction']
            n      = 1 + VT_NUM_NEG
            for i in range(len(preds) // n):
                scores = preds[i * n: (i + 1) * n]
                rank   = int((scores > scores[0]).sum().item()) + 1
                if rank <= 3:
                    hit += 1.0 / np.log2(rank + 1)
                total += 1
    model.train()
    return hit / max(total, 1)


def build_vae(dp_dict, user_num, item_num, save_path, dataset, with_filter=True):
    torch.manual_seed(RANDOM_SEED)
    cls = VAE_PCFR if with_filter else VAE
    return cls(
        data_processor_dict=dp_dict,
        user_num=user_num,
        item_num=item_num,
        u_vector_size=U_VEC_SIZE,
        i_vector_size=U_VEC_SIZE,
        vae_hidden=VAE_HIDDEN,
        vae_beta=VAE_BETA,
        random_seed=RANDOM_SEED,
        dropout=0.2,
        model_path=save_path,
    )


def build_discs(feature_info, save_dir):
    discs, opts = {}, {}
    for feat_idx, feat in feature_info.items():
        d = BinaryDiscriminator(U_VEC_SIZE, feat, random_seed=RANDOM_SEED,
                                dropout=0.2, model_dir_path=save_dir, layers=3)
        d.apply(d.init_weights)
        discs[feat_idx] = d
        opts[feat_idx]  = torch.optim.Adam(d.parameters(), lr=DISC_LR,
                                            weight_decay=DISC_L2)
    return discs, opts


# ── None baseline (no adversarial training) ───────────────────────────────────

def train_none(dataset, save_path):
    print(f"\n{'─'*70}")
    print(f"  Training: VAE_None / {dataset}")
    print(f"{'─'*70}")

    ALL_ATTRS = DATASET_ATTRS[dataset]
    dr = RecDataReader(path=DATA_PATH, dataset_name=dataset,
                       feature_columns=[ALL_ATTRS[0]], sep='\t')
    user_num, item_num = len(dr.user_ids_set), len(dr.item_ids_set)

    train_dp = RecDataset(data_reader=dr, stage='train',
                          batch_size=BATCH_SIZE, num_neg=1)
    valid_dp = RecDataset(data_reader=dr, stage='valid',
                          batch_size=BATCH_SIZE, num_neg=VT_NUM_NEG)
    dp_dict  = {'train': train_dp, 'valid': valid_dp}

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model     = build_vae(dp_dict, user_num, item_num, save_path, dataset, with_filter=False)
    model_opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=L2)

    best_ndcg, best_epoch = -1.0, 0
    train_loader = DataLoader(train_dp, batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKER, shuffle=True,
                              collate_fn=train_dp.collate_fn)
    valid_loader = DataLoader(valid_dp, batch_size=None,
                              num_workers=NUM_WORKER, pin_memory=False,
                              collate_fn=valid_dp.collate_fn)

    for epoch in range(EPOCHS):
        model.train()
        loss_acc, n_batches = 0.0, 0
        for batch in tqdm(train_loader, leave=False,
                          desc=f"Ep {epoch+1:3d}/{EPOCHS}", ncols=80):
            batch = batch_to_gpu(batch)
            model_opt.zero_grad()
            result = model(batch)
            result['loss'].backward()
            model_opt.step()
            loss_acc  += result['loss'].item()
            n_batches += 1

        ndcg = evaluate_rec(model, valid_loader)
        if ndcg > best_ndcg:
            best_ndcg, best_epoch = ndcg, epoch + 1
            torch.save(model.state_dict(), save_path)

        print(f"  Ep {epoch+1:3d}/{EPOCHS}  loss={loss_acc/n_batches:.4f}  "
              f"val_ndcg={ndcg:.4f}  best={best_ndcg:.4f} (ep {best_epoch})")
        gc.collect()

    print(f"\n  ✓ Saved best (ep {best_epoch}, NDCG={best_ndcg:.4f}) → {save_path}")


# ── PCFR training (single-attr or multi-attr) ─────────────────────────────────

def train_pcfr(dataset, attrs, save_path, label, active_mask):
    print(f"\n{'─'*70}")
    print(f"  Training: {label} / {dataset}")
    print(f"{'─'*70}")

    dr = RecDataReader(path=DATA_PATH, dataset_name=dataset,
                       feature_columns=attrs, sep='\t')
    user_num, item_num = len(dr.user_ids_set), len(dr.item_ids_set)

    train_dp = RecDataset(data_reader=dr, stage='train',
                          batch_size=BATCH_SIZE, num_neg=1)
    valid_dp = RecDataset(data_reader=dr, stage='valid',
                          batch_size=BATCH_SIZE, num_neg=VT_NUM_NEG)
    dp_dict  = {'train': train_dp, 'valid': valid_dp}

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model     = build_vae(dp_dict, user_num, item_num, save_path, dataset, with_filter=True)
    model_opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=L2)
    discs, disc_opts = build_discs(dr.feature_info, os.path.dirname(save_path))

    train_loader = DataLoader(train_dp, batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKER, shuffle=True,
                              collate_fn=train_dp.collate_fn)
    valid_loader = DataLoader(valid_dp, batch_size=None,
                              num_workers=NUM_WORKER, pin_memory=False,
                              collate_fn=valid_dp.collate_fn)

    best_ndcg, best_epoch = -1.0, 0

    for epoch in range(EPOCHS):
        model.train()
        for d in discs.values():
            d.train()

        loss_acc, n_batches = 0.0, 0

        for batch in tqdm(train_loader, leave=False,
                          desc=f"Ep {epoch+1:3d}/{EPOCHS}", ncols=80):
            batch = batch_to_gpu(batch)
            model_opt.zero_grad()

            labels     = batch['features'][:len(batch['features']) // 2, :]
            disc_pairs = [
                (discs[i + 1], labels[:, i])
                for i, v in enumerate(active_mask) if v != 0
            ]

            result   = model(batch)
            rec_loss = result['loss']
            vectors  = result['u_vectors'][:len(result['u_vectors']) // 2, :]

            adv_loss = sum(d(vectors, lbl) for d, lbl in disc_pairs)
            total    = rec_loss + ADV_WEIGHT * (-adv_loss)
            total.backward()
            model_opt.step()

            for d, lbl in disc_pairs:
                idx = list(discs.values()).index(d)
                disc_opts[list(dr.feature_info.keys())[idx]].zero_grad()
                d(vectors.detach(), lbl).backward()
                disc_opts[list(dr.feature_info.keys())[idx]].step()

            loss_acc  += rec_loss.item()
            n_batches += 1

        ndcg = evaluate_rec(model, valid_loader)
        if ndcg > best_ndcg:
            best_ndcg, best_epoch = ndcg, epoch + 1
            torch.save(model.state_dict(), save_path)

        print(f"  Ep {epoch+1:3d}/{EPOCHS}  loss={loss_acc/n_batches:.4f}  "
              f"val_ndcg={ndcg:.4f}  best={best_ndcg:.4f} (ep {best_epoch})")
        gc.collect()

    print(f"\n  ✓ Saved best (ep {best_epoch}, NDCG={best_ndcg:.4f}) → {save_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='both',
                        choices=['insurance', 'ml100k', 'both'])
    args = parser.parse_args()

    datasets = ['insurance', 'ml100k'] if args.dataset == 'both' else [args.dataset]

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    for dataset in datasets:
        ALL_ATTRS = DATASET_ATTRS[dataset]

        print("\n" + "=" * 70)
        print(f"  VAE-PCFR Training  |  {dataset}")
        print("=" * 70)

        # ── Phase 0: None baseline ─────────────────────────────────────────
        none_path = os.path.join(MODEL_DIR, f'VAE_None_{dataset}', 'model.pt')
        if os.path.exists(none_path):
            print(f"\n  ✓ VAE_None/{dataset} checkpoint exists, skipping.")
        else:
            train_none(dataset, none_path)

        # ── Phase 1: Single-attribute PCFR × 4 attrs ──────────────────────
        print(f"\n[Phase 1]  Single-attribute PCFR × {len(ALL_ATTRS)} attributes")
        for attr in ALL_ATTRS:
            save_path = os.path.join(MODEL_DIR, f'VAE_PCFR_{dataset}_{attr}', 'model.pt')
            if os.path.exists(save_path):
                print(f"\n  ✓ VAE_PCFR/{dataset}/{attr} checkpoint exists, skipping.")
                continue
            train_pcfr(dataset, [attr], save_path,
                       f"VAE_PCFR / {attr}", active_mask=[1])

        # ── Phase 2: Multi-attribute PCFR ─────────────────────────────────
        print(f"\n[Phase 2]  Multi-attribute PCFR (all {len(ALL_ATTRS)} attributes)")
        multi_path = os.path.join(MODEL_DIR, f'VAE_PCFR_multiattr_{dataset}', 'model.pt')
        if os.path.exists(multi_path):
            print(f"\n  ✓ VAE_PCFR_multiattr/{dataset} checkpoint exists, skipping.")
        else:
            active_mask = [1] * len(ALL_ATTRS)
            train_pcfr(dataset, ALL_ATTRS, multi_path,
                       f"VAE_PCFR_MultiAttr / all_attrs", active_mask=active_mask)

    print("\n" + "=" * 70)
    print("  All VAE training complete.")
    print("  Next: run autoencoder_eval.py to get Q1 + Q2 + quality results.")
    print("=" * 70)


if __name__ == '__main__':
    main()
