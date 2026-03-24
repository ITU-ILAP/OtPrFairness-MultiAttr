# coding=utf-8
"""
GNN-PCFR Training Script
=========================
Trains LightGCN_PCFR on the insurance dataset:
  • Single-attribute mode × 4 attributes  (Q1 checkpoints)
  • Multi-attribute mode on all 4 at once (Q2 checkpoint)

Checkpoints saved to:
  ../model/GNN_PCFR_insurance_{attr}/model.pt   (single-attr)
  ../model/GNN_PCFR_multiattr_insurance/model.pt (multi-attr)

Usage (from src/):
    python gnn_pcfr_train.py

After training, run gnn_eval.py to get Q1 + Q2 results.
"""

import os
import sys
import gc
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.ERROR)

from data_reader import RecDataReader, DiscriminatorDataReader
from datasets import RecDataset, DiscriminatorDataset
from models.GNN import LightGCN_PCFR
from models.Discriminators import BinaryDiscriminator
from utils.generic import batch_to_gpu
from utils.constants import LABEL

# ── Config ─────────────────────────────────────────────────────────────────────

DATASET      = 'insurance'
DATA_PATH    = '../dataset/'
ALL_ATTRS    = ['u_gender', 'u_occupation', 'u_activity', 'u_marital_status']

EPOCHS       = 100
LR           = 1e-3
L2           = 1e-4
BATCH_SIZE   = 1024
VT_NUM_NEG   = 10
U_VEC_SIZE   = 64
N_GNN_LAYERS = 2        # LightGCN propagation depth
MIN_COMMON   = 3        # min shared items for a user-user edge; keeps graph sparse
                        # insurance has 21 items — min_common<3 creates a nearly
                        # complete user-user graph (>24% dense) which is too large
RANDOM_SEED  = 2020
NUM_WORKER   = 0

DISC_LR      = 1e-3
DISC_L2      = 1e-4
ADV_WEIGHT   = 1.0

MODEL_DIR    = '../model/'


# ── Helpers ────────────────────────────────────────────────────────────────────

def evaluate_rec(model, loader):
    """Simple NDCG@3 proxy for checkpoint selection."""
    model.eval()
    hit, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_gpu(batch)
            preds = model.predict(batch)['prediction']
            n = 1 + VT_NUM_NEG
            for i in range(len(preds) // n):
                scores   = preds[i * n: (i + 1) * n]
                rank     = int((scores > scores[0]).sum().item()) + 1
                if rank <= 3:
                    hit += 1.0 / np.log2(rank + 1)
                total += 1
    model.train()
    return hit / max(total, 1)


def build_model(dp_dict, user_num, item_num, save_path):
    torch.manual_seed(RANDOM_SEED)
    model = LightGCN_PCFR(
        data_processor_dict=dp_dict,
        user_num=user_num,
        item_num=item_num,
        u_vector_size=U_VEC_SIZE,
        i_vector_size=U_VEC_SIZE,
        n_gnn_layers=N_GNN_LAYERS,
        min_common=MIN_COMMON,
        random_seed=RANDOM_SEED,
        dropout=0.2,
        model_path=save_path,
    )
    return model


def build_discriminators(feature_info, save_dir):
    discs, opts = {}, {}
    for feat_idx, feat in feature_info.items():
        d = BinaryDiscriminator(U_VEC_SIZE, feat, random_seed=RANDOM_SEED,
                                dropout=0.2, model_dir_path=save_dir, layers=3)
        d.apply(d.init_weights)
        discs[feat_idx] = d
        opts[feat_idx]  = torch.optim.Adam(d.parameters(), lr=DISC_LR,
                                            weight_decay=DISC_L2)
    return discs, opts


def train_one_run(dp_dict, user_num, item_num, feature_info,
                  save_path, label, active_mask):
    """
    Core training loop.
    active_mask: list of 0/1 flags (len = num features) — which attrs to protect.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    train_dp = dp_dict['train']
    valid_dp = dp_dict['valid']

    train_loader = DataLoader(train_dp, batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKER, shuffle=True,
                              collate_fn=train_dp.collate_fn)
    valid_loader = DataLoader(valid_dp, batch_size=None,
                              num_workers=NUM_WORKER, pin_memory=False,
                              collate_fn=valid_dp.collate_fn)

    model     = build_model(dp_dict, user_num, item_num, save_path)
    model_opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=L2)
    discs, disc_opts = build_discriminators(feature_info, os.path.dirname(save_path))

    best_ndcg, best_epoch = -1.0, 0
    print(f"\n{'─'*70}")
    print(f"  Training: {label}")
    print(f"{'─'*70}")

    for epoch in range(EPOCHS):
        model.train()
        for d in discs.values():
            d.train()

        loss_acc, n_batches = 0.0, 0

        for batch in tqdm(train_loader, leave=False,
                          desc=f"Ep {epoch+1:3d}/{EPOCHS}", ncols=80):
            batch = batch_to_gpu(batch)
            model_opt.zero_grad()

            # Labels for positive half of batch only
            labels = batch['features'][:len(batch['features']) // 2, :]
            disc_pairs = [
                (discs[i + 1], labels[:, i])
                for i, v in enumerate(active_mask) if v != 0
            ]

            result    = model(batch)
            rec_loss  = result['loss']
            vectors   = result['u_vectors'][:len(result['u_vectors']) // 2, :]

            adv_loss  = sum(d(vectors, lbl) for d, lbl in disc_pairs)
            total     = rec_loss + ADV_WEIGHT * (-adv_loss)
            total.backward()
            model_opt.step()

            # Discriminator update
            for d, lbl in disc_pairs:
                idx = list(discs.values()).index(d)
                disc_opts[list(feature_info.keys())[idx]].zero_grad()
                d(vectors.detach(), lbl).backward()
                disc_opts[list(feature_info.keys())[idx]].step()

            loss_acc  += rec_loss.item()
            n_batches += 1

        ndcg = evaluate_rec(model, valid_loader)
        if ndcg > best_ndcg:
            best_ndcg, best_epoch = ndcg, epoch + 1
            torch.save(model.state_dict(), save_path)

        print(f"  Ep {epoch+1:3d}/{EPOCHS}  "
              f"loss={loss_acc/n_batches:.4f}  "
              f"val_ndcg={ndcg:.4f}  "
              f"best={best_ndcg:.4f} (ep {best_epoch})")
        gc.collect()

    print(f"\n  ✓ Saved best (ep {best_epoch}, NDCG={best_ndcg:.4f}) → {save_path}")
    return save_path


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 70)
    print("  GNN-PCFR Training  |  insurance  |  LightGCN multi-graph")
    print("=" * 70)

    # ── Phase 1: single-attribute runs (Q1 checkpoints) ────────────────────
    print("\n[Phase 1]  Single-attribute PCFR × 4 attributes")

    for attr in ALL_ATTRS:
        save_path = os.path.join(MODEL_DIR, f'GNN_PCFR_insurance_{attr}', 'model.pt')
        if os.path.exists(save_path):
            print(f"\n  ✓ {attr} checkpoint exists, skipping.")
            continue

        dr = RecDataReader(path=DATA_PATH, dataset_name=DATASET,
                           feature_columns=[attr], sep='\t')
        user_num, item_num = len(dr.user_ids_set), len(dr.item_ids_set)

        train_dp = RecDataset(data_reader=dr, stage='train',
                              batch_size=BATCH_SIZE, num_neg=1)
        valid_dp = RecDataset(data_reader=dr, stage='valid',
                              batch_size=BATCH_SIZE, num_neg=VT_NUM_NEG)
        dp_dict  = {'train': train_dp, 'valid': valid_dp}

        active_mask = [1]   # only one feature
        train_one_run(dp_dict, user_num, item_num, dr.feature_info,
                      save_path, f"GNN_PCFR / {attr}", active_mask)

    # ── Phase 2: multi-attribute run (Q2 checkpoint) ───────────────────────
    print("\n[Phase 2]  Multi-attribute PCFR (all 4 attributes simultaneously)")

    save_path_multi = os.path.join(MODEL_DIR, 'GNN_PCFR_multiattr_insurance', 'model.pt')
    if os.path.exists(save_path_multi):
        print(f"\n  ✓ MultiAttr checkpoint exists, skipping.")
    else:
        dr = RecDataReader(path=DATA_PATH, dataset_name=DATASET,
                           feature_columns=ALL_ATTRS, sep='\t')
        user_num, item_num = len(dr.user_ids_set), len(dr.item_ids_set)

        train_dp = RecDataset(data_reader=dr, stage='train',
                              batch_size=BATCH_SIZE, num_neg=1)
        valid_dp = RecDataset(data_reader=dr, stage='valid',
                              batch_size=BATCH_SIZE, num_neg=VT_NUM_NEG)
        dp_dict  = {'train': train_dp, 'valid': valid_dp}

        active_mask = [1, 1, 1, 1]   # all four attributes
        train_one_run(dp_dict, user_num, item_num, dr.feature_info,
                      save_path_multi, "GNN_PCFR_MultiAttr / all_attrs", active_mask)

    print("\n" + "=" * 70)
    print("  All training complete.")
    print("  Run  python gnn_eval.py  to get Q1 + Q2 results.")
    print("=" * 70)


if __name__ == '__main__':
    main()
