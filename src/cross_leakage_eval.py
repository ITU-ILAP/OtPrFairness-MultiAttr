# coding=utf-8
"""
Cross-Attribute Leakage Evaluation
===================================
Q1: When we mitigate leakage for one sensitive attribute, what happens to
    the leakage of OTHER sensitive attributes?

For each (framework, trained_attribute) combination that has a saved model,
this script loads the frozen model and trains fresh discriminators for ALL 4
attributes, producing a cross-leakage AUC matrix.

Usage (from src/):
    python cross_leakage_eval.py

No changes to existing code. Standalone script only.
"""

import os
import sys
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_reader import RecDataReader, DiscriminatorDataReader
from datasets import RecDataset, DiscriminatorDataset
from models.BiasedMF import (BiasedMF, BiasedMF_FOCF_ValUnf, BiasedMF_FOCF_AbsUnf,
                              BiasedMF_PCFR, BiasedMF_FairRec, BiasedMF_FairPO)
from models.Discriminators import BinaryDiscriminator
from utils.generic import batch_to_gpu

# Map framework name → model class
MODEL_CLASS = {
    'None':           BiasedMF,
    'FOCF_ValUnf':    BiasedMF_FOCF_ValUnf,
    'FOCF_AbsUnf':    BiasedMF_FOCF_AbsUnf,
    'PCFR':           BiasedMF_PCFR,
    'FairRec':        BiasedMF_FairRec,
    'FairPO':         BiasedMF_FairPO,
    'PCFR_MultiAttr': BiasedMF_PCFR,   # same architecture, trained on all attrs
}

# suppress all INFO/DEBUG noise from the project
logging.basicConfig(level=logging.ERROR)

# ── Configuration ──────────────────────────────────────────────────────────────

DATASET     = 'insurance'
DATA_PATH   = '../dataset/'
ALL_ATTRS   = ['u_gender', 'u_occupation', 'u_activity', 'u_marital_status']
DISC_EPOCHS = 300   # fewer than default 500 for speed; enough to converge
DISC_LR     = 0.001
DISC_L2     = 1e-4
U_VEC_SIZE  = 64
RANDOM_SEED = 2020
NUM_WORKER  = 0
BATCH_SIZE  = 256
VT_NUM_NEG  = 10

# Saved checkpoints: (framework, trained_attr) -> model_path
MODELS = {}
model_base = '../model/'
for fw in ['PCFR']:
    # u_gender models saved by run_comparison
    cmp_path = os.path.join(model_base, f'cmp_BiasedMF_{fw}_insurance_u_gender', 'model.pt')
    if os.path.exists(cmp_path):
        MODELS[(fw, 'u_gender')] = cmp_path
    # other attribute models saved by our recent full runs
    for attr in ['u_occupation', 'u_activity', 'u_marital_status']:
        folder = f'BiasedMF_{fw}_insurance_{attr}_neg_sample={VT_NUM_NEG}'
        fname  = f'BiasedMF_{fw}_insurance_{attr}_l2=1e-4_dim=64_neg_sample={VT_NUM_NEG}.pt'
        pt = os.path.join(model_base, folder, fname)
        if os.path.exists(pt):
            MODELS[(fw, attr)] = pt

# Q2: multi-attribute PCFR trained on all 4 attrs simultaneously
_q2_path = os.path.join(model_base, 'PCFR_multiattr_insurance', 'model.pt')
if os.path.exists(_q2_path):
    MODELS[('PCFR_MultiAttr', 'all_attrs')] = _q2_path

# ── Model loading ──────────────────────────────────────────────────────────────

def build_rec_dp_dict(feature_col):
    """Build RecDataReader + RecDataset dict for one feature column."""
    dr = RecDataReader(
        path=DATA_PATH,
        dataset_name=DATASET,
        feature_columns=[feature_col],
        sep='\t',
    )
    train_dp = RecDataset(data_reader=dr, stage='train',
                          batch_size=BATCH_SIZE, num_neg=VT_NUM_NEG)
    test_dp  = RecDataset(data_reader=dr, stage='test',
                          batch_size=BATCH_SIZE, num_neg=VT_NUM_NEG)
    return dr, {'train': train_dp, 'test': test_dp}


def load_frozen_model(model_path, dp_dict, framework):
    """Load the correct BiasedMF variant checkpoint and freeze weights."""
    torch.manual_seed(RANDOM_SEED)
    dr = dp_dict['train'].data_reader
    # match how main.py computes user_num/item_num
    user_num = len(dr.user_ids_set)
    item_num = len(dr.item_ids_set)
    model_cls = MODEL_CLASS[framework]
    model = model_cls(
        data_processor_dict=dp_dict,
        user_num=user_num,
        item_num=item_num,
        u_vector_size=U_VEC_SIZE,
        i_vector_size=U_VEC_SIZE,
        random_seed=RANDOM_SEED,
        dropout=0.2,
        model_path=model_path,
    )
    model.load_model()
    model.freeze_model()
    model.eval()
    return model

# ── Discriminator evaluation ───────────────────────────────────────────────────

def eval_leakage(model, eval_attr):
    """
    Train a fresh discriminator for eval_attr on the frozen model's raw user
    embeddings (uid_embeddings), mirroring the evaluation_disc approach used in
    runner.py.  Uses DiscriminatorDataset so batch['X'] is a 1-D user-ID tensor.
    Returns best attacker AUC over DISC_EPOCHS epochs.
    """
    # Discriminator data reader + dataset for this attribute
    disc_dr = DiscriminatorDataReader(
        path=DATA_PATH,
        dataset_name=DATASET,
        feature_columns=[eval_attr],
        sep='\t',
        test_ratio=0.2,
    )
    train_disc_dp = DiscriminatorDataset(data_reader=disc_dr, stage='train',
                                         batch_size=BATCH_SIZE)
    test_disc_dp  = DiscriminatorDataset(data_reader=disc_dr, stage='test',
                                         batch_size=BATCH_SIZE)

    train_loader = DataLoader(train_disc_dp, batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKER, shuffle=True,
                              collate_fn=DiscriminatorDataset.collate_fn)
    test_loader  = DataLoader(test_disc_dp,  batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKER, pin_memory=False,
                              collate_fn=DiscriminatorDataset.collate_fn)

    # Fresh discriminator
    torch.manual_seed(RANDOM_SEED)
    disc = BinaryDiscriminator(
        U_VEC_SIZE,
        disc_dr.feature_info[1],
        random_seed=RANDOM_SEED,
        dropout=0.2,
        model_dir_path='/tmp',   # needed by constructor but we won't save
        layers=3,
    )
    disc.apply(disc.init_weights)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=DISC_LR, weight_decay=DISC_L2)

    best_auc = 0.5

    for epoch in range(DISC_EPOCHS):
        # ── train discriminator on raw user embeddings ──
        disc.train()
        for batch in train_loader:
            batch = batch_to_gpu(batch)
            with torch.no_grad():
                # batch['X'] is 1-D: user IDs (as stored in DiscriminatorDataset)
                uids    = batch['X'] - 1                      # 0-indexed
                vectors = model.uid_embeddings(uids)          # [B, embed_dim]
            labels = batch['features']
            if labels.shape[1] == 0:
                continue
            label = labels[:, 0]
            disc_opt.zero_grad()
            loss = disc(vectors, label)
            loss.backward()
            disc_opt.step()

        # ── evaluate ──
        disc.eval()
        all_preds, all_labels = [], []
        for batch in test_loader:
            batch = batch_to_gpu(batch)
            with torch.no_grad():
                uids    = batch['X'] - 1
                vectors = model.uid_embeddings(uids)
                labels  = batch['features']
            if labels.shape[1] == 0:
                continue
            label = labels[:, 0]
            # use continuous score ('output') for AUC, not thresholded 'prediction'
            pred  = disc.predict(vectors)['output'].squeeze().detach()
            all_preds.append(pred.cpu().numpy())
            all_labels.append(label.cpu().numpy())

        if not all_preds:
            break

        preds = np.concatenate(all_preds)
        lbls  = np.concatenate(all_labels)
        try:
            auc = roc_auc_score(lbls, preds)
        except Exception:
            auc = 0.5

        if auc > best_auc:
            best_auc = auc

    return best_auc

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("\n" + "="*92)
    print("  CROSS-ATTRIBUTE LEAKAGE EVALUATION  |  insurance / BiasedMF")
    print("  Row  = framework + attribute the model was trained to protect")
    print("  Col  = attribute being probed by the attacker")
    print("  0.50 = random (no leakage) | ← = trained attribute")
    print("="*92)

    frameworks = ['PCFR', 'PCFR_MultiAttr']
    col_w = 14
    results = {}

    # Build trained_attr list: single attrs for PCFR, 'all_attrs' for PCFR_MultiAttr
    trained_attrs_for = {fw: ALL_ATTRS for fw in frameworks}
    trained_attrs_for['PCFR_MultiAttr'] = ['all_attrs']

    for fw in frameworks:
        for trained_attr in trained_attrs_for[fw]:
            key = (fw, trained_attr)
            if key not in MODELS:
                continue

            model_path = MODELS[key]
            print(f"\n▶ {fw} trained on {trained_attr}  [{os.path.basename(model_path)}]")

            # Build dp_dict: multi-attr model needs all 4 cols; others need 1
            feature_cols = ALL_ATTRS if trained_attr == 'all_attrs' else [trained_attr]
            try:
                dr = RecDataReader(path=DATA_PATH, dataset_name=DATASET,
                                   feature_columns=feature_cols, sep='\t')
                train_dp = RecDataset(data_reader=dr, stage='train',
                                      batch_size=BATCH_SIZE, num_neg=VT_NUM_NEG)
                test_dp  = RecDataset(data_reader=dr, stage='test',
                                      batch_size=BATCH_SIZE, num_neg=VT_NUM_NEG)
                dp_dict = {'train': train_dp, 'test': test_dp}
                model = load_frozen_model(model_path, dp_dict, fw)
            except Exception as e:
                print(f"  ✗ Load failed: {e}")
                continue

            aucs = {}
            for eval_attr in ALL_ATTRS:
                marker = " ←" if eval_attr == trained_attr else "  "
                sys.stdout.write(f"  {eval_attr:28s} ... ")
                sys.stdout.flush()
                try:
                    auc = eval_leakage(model, eval_attr)
                    aucs[eval_attr] = auc
                    print(f"AUC = {auc:.4f}{marker}")
                except Exception as e:
                    aucs[eval_attr] = None
                    print(f"ERROR: {e}")

            results[key] = aucs

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n\n" + "="*92)
    print("  SUMMARY TABLE")
    print("="*92)
    print(f"{'Framework':<16} {'Trained on':>16}  " +
          "  ".join(f"{a:>{col_w}}" for a in ALL_ATTRS))
    print("-" * 92)

    for fw in frameworks:
        all_trained = trained_attrs_for.get(fw, ALL_ATTRS)
        for trained_attr in all_trained:
            key = (fw, trained_attr)
            if key not in results:
                continue
            aucs = results[key]
            row = f"{fw:<16} {trained_attr:>16}  " + \
                  "  ".join(
                      f"{aucs[a]:>{col_w}.4f}" if aucs.get(a) is not None
                      else f"{'—':>{col_w}}"
                      for a in ALL_ATTRS
                  )
            print(row)

    print("\n  ← trained attribute  |  values near 0.50 = low leakage")
    print("="*92 + "\n")


if __name__ == '__main__':
    main()
