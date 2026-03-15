# coding=utf-8
"""
Q2: Multi-Attribute PCFR Training
===================================
Trains BiasedMF_PCFR against ALL 4 sensitive attributes simultaneously
using one discriminator per attribute.  Mirrors RecRunner_PCFR.fit() logic
exactly — no changes to any existing file.

Usage (from src/):
    python multi_attr_pcfr_train.py

Checkpoint saved to:
    ../model/PCFR_multiattr_insurance/model.pt

After training, add this model to cross_leakage_eval.py to compare against
the single-attribute PCFR results (Q1).
"""

import os
import sys
import gc
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.ERROR)

from data_reader import RecDataReader
from datasets import RecDataset, DiscriminatorDataset
from data_reader import DiscriminatorDataReader
from models.BiasedMF import BiasedMF_PCFR
from models.Discriminators import BinaryDiscriminator
from utils.generic import batch_to_gpu
from utils.constants import LABEL

# ── Configuration ──────────────────────────────────────────────────────────────

DATASET     = 'insurance'
DATA_PATH   = '../dataset/'
ALL_ATTRS   = ['u_gender', 'u_occupation', 'u_activity', 'u_marital_status']

EPOCHS      = 100       # match the comparison runs
LR          = 1e-3
L2          = 1e-4
BATCH_SIZE  = 1024
VT_NUM_NEG  = 10
U_VEC_SIZE  = 64
RANDOM_SEED = 2020
NUM_WORKER  = 0

DISC_LR     = 1e-3
DISC_L2     = 1e-4
ADV_WEIGHT  = 1.0       # weight on total adversarial penalty

SAVE_DIR    = '../model/PCFR_multiattr_insurance/'
SAVE_PATH   = os.path.join(SAVE_DIR, 'model.pt')

# ── Helpers ────────────────────────────────────────────────────────────────────

def best_result(metric, results):
    """Return best value; higher is better for ndcg/f1, lower for RMSE."""
    if 'ndcg' in metric.lower() or 'f1' in metric.lower():
        return max(results)
    return min(results)


def evaluate_rec(model, loader):
    """Simple NDCG@3 proxy: mean prediction rank among negatives per user."""
    model.eval()
    hit, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_gpu(batch)
            out = model.predict(batch)
            preds = out['prediction']
            # batch has (1+VT_NUM_NEG) rows per user; first row is positive
            n = 1 + VT_NUM_NEG
            num_users = len(preds) // n
            for i in range(num_users):
                scores = preds[i * n: (i + 1) * n]
                pos_score = scores[0]
                rank = (scores > pos_score).sum().item() + 1   # 1-based rank
                if rank <= 3:
                    hit += 1.0 / np.log2(rank + 1)
                total += 1
    model.train()
    return hit / max(total, 1)


def attacker_auc(model, eval_attr):
    """Quick attacker AUC using 50-epoch discriminator (for progress logging)."""
    disc_dr = DiscriminatorDataReader(
        path=DATA_PATH, dataset_name=DATASET,
        feature_columns=[eval_attr], sep='\t', test_ratio=0.2,
    )
    train_dp = DiscriminatorDataset(data_reader=disc_dr, stage='train', batch_size=512)
    test_dp  = DiscriminatorDataset(data_reader=disc_dr, stage='test',  batch_size=512)
    tr_ld = DataLoader(train_dp, batch_size=512, shuffle=True,
                       collate_fn=DiscriminatorDataset.collate_fn)
    te_ld = DataLoader(test_dp,  batch_size=512,
                       collate_fn=DiscriminatorDataset.collate_fn)

    torch.manual_seed(RANDOM_SEED)
    d = BinaryDiscriminator(U_VEC_SIZE, disc_dr.feature_info[1],
                            random_seed=RANDOM_SEED, dropout=0.2,
                            model_dir_path='/tmp', layers=3)
    d.apply(d.init_weights)
    opt = torch.optim.Adam(d.parameters(), lr=0.001, weight_decay=1e-4)

    best = 0.5
    for _ in range(50):
        d.train()
        for b in tr_ld:
            b = batch_to_gpu(b)
            with torch.no_grad():
                v = model.uid_embeddings(b['X'] - 1)
            lbl = b['features'][:, 0]
            opt.zero_grad(); d(v, lbl).backward(); opt.step()
        d.eval(); ps, ls = [], []
        for b in te_ld:
            b = batch_to_gpu(b)
            with torch.no_grad():
                v = model.uid_embeddings(b['X'] - 1)
                lbl = b['features'][:, 0]
            ps.append(d.predict(v)['output'].squeeze().detach().cpu().numpy())
            ls.append(lbl.cpu().numpy())
        if ps:
            try:
                auc = roc_auc_score(np.concatenate(ls), np.concatenate(ps))
                best = max(best, auc)
            except Exception:
                pass
    return best


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    os.makedirs(SAVE_DIR, exist_ok=True)

    print("\n" + "=" * 80)
    print("  Q2: Multi-Attribute PCFR  |  insurance / BiasedMF")
    print("  Protecting all 4 attributes simultaneously")
    print("=" * 80)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n[1/4] Loading data with all 4 feature columns ...")
    dr = RecDataReader(
        path=DATA_PATH,
        dataset_name=DATASET,
        feature_columns=ALL_ATTRS,
        sep='\t',
    )
    train_dp = RecDataset(data_reader=dr, stage='train',
                          batch_size=BATCH_SIZE, num_neg=1)        # 1 neg per BPR pair
    valid_dp = RecDataset(data_reader=dr, stage='valid',
                          batch_size=BATCH_SIZE, num_neg=VT_NUM_NEG)  # 10 negs for ranking

    train_loader = DataLoader(train_dp, batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKER, shuffle=True,
                              collate_fn=train_dp.collate_fn)
    valid_loader = DataLoader(valid_dp, batch_size=None,
                              num_workers=NUM_WORKER, pin_memory=False,
                              collate_fn=valid_dp.collate_fn)

    user_num = len(dr.user_ids_set)
    item_num = len(dr.item_ids_set)
    num_features = dr.num_features
    print(f"    users={user_num}, items={item_num}, "
          f"features={num_features} {list(dr.feature_info[i].name for i in dr.feature_info)}")

    dp_dict = {'train': train_dp, 'valid': valid_dp}

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n[2/4] Building BiasedMF_PCFR ...")
    torch.manual_seed(RANDOM_SEED)
    model = BiasedMF_PCFR(
        data_processor_dict=dp_dict,
        user_num=user_num,
        item_num=item_num,
        u_vector_size=U_VEC_SIZE,
        i_vector_size=U_VEC_SIZE,
        random_seed=RANDOM_SEED,
        dropout=0.2,
        model_path=SAVE_PATH,
    )
    model_opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=L2)

    # ── Discriminators — one per sensitive attribute ───────────────────────────
    print("\n[3/4] Building discriminators ...")
    fair_disc_dict = {}
    disc_opts = {}
    for feat_idx, feat_info in dr.feature_info.items():
        disc = BinaryDiscriminator(
            U_VEC_SIZE,
            feat_info,
            random_seed=RANDOM_SEED,
            dropout=0.2,
            model_dir_path=SAVE_DIR,
            layers=3,
        )
        disc.apply(disc.init_weights)
        fair_disc_dict[feat_idx] = disc
        disc_opts[feat_idx] = torch.optim.Adam(
            disc.parameters(), lr=DISC_LR, weight_decay=DISC_L2)
        print(f"    disc[{feat_idx}] → {feat_info.name}  (num_class={feat_info.num_class})")

    # ── Training loop (mirrors RecRunner_PCFR.fit) ────────────────────────────
    print("\n[4/4] Training ...")
    mask = np.ones(num_features, dtype=int)    # all attributes active every batch

    best_ndcg   = -1.0
    best_epoch  = 0

    for epoch in range(EPOCHS):
        model.train()
        for disc in fair_disc_dict.values():
            disc.train()

        loss_acc = 0.0
        n_batches = 0

        for batch in tqdm(train_loader, leave=False,
                          desc=f"Epoch {epoch+1:3d}/{EPOCHS}", ncols=90):
            batch = batch_to_gpu(batch)
            model_opt.zero_grad()

            # ── labels for positive half of batch only ──
            labels = batch['features'][:len(batch['features']) // 2, :]
            # ── pair each active disc with its label column ──
            disc_label_pairs = [
                (fair_disc_dict[i + 1], labels[:, i])
                for i, v in enumerate(mask) if v != 0
            ]

            # ── forward through model ──
            result = model(batch)
            rec_loss = result['loss']
            vectors  = result['u_vectors'][:len(result['u_vectors']) // 2, :]

            # ── model adversarial update: minimise -sum(disc losses) ──
            adv_penalty = sum(disc(vectors, lbl) for disc, lbl in disc_label_pairs)
            total_loss = rec_loss + ADV_WEIGHT * (-adv_penalty)
            total_loss.backward()
            model_opt.step()

            # ── discriminator update (detached vectors) ──
            for disc, lbl in disc_label_pairs:
                opt = disc_opts[list(dr.feature_info.keys())[
                    list(fair_disc_dict.values()).index(disc)]]
                opt.zero_grad()
                disc(vectors.detach(), lbl).backward()
                opt.step()

            loss_acc  += rec_loss.item()
            n_batches += 1

        # ── validation ──
        ndcg = evaluate_rec(model, valid_loader)

        # save best
        if ndcg > best_ndcg:
            best_ndcg  = ndcg
            best_epoch = epoch + 1
            torch.save(model.state_dict(), SAVE_PATH)

        print(f"  Epoch {epoch+1:3d}/{EPOCHS}  "
              f"loss={loss_acc/n_batches:.4f}  "
              f"val_ndcg={ndcg:.4f}  "
              f"best={best_ndcg:.4f} (ep {best_epoch})")

        gc.collect()

    print(f"\n✓ Best model at epoch {best_epoch}  NDCG@3={best_ndcg:.4f}")
    print(f"✓ Saved to {SAVE_PATH}")
    print("\nNow run cross_leakage_eval.py to compare Q1 vs Q2 results.")


if __name__ == '__main__':
    main()
