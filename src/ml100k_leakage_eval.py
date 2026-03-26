# coding=utf-8
"""
Cross-Attribute Leakage Evaluation — ml100k
Same protocol as cross_leakage_eval.py but for the ml100k dataset.

Usage (from src/):
    python ml100k_leakage_eval.py
"""

import os, sys, logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.ERROR)

from data_reader import RecDataReader, DiscriminatorDataReader
from datasets import RecDataset, DiscriminatorDataset
from models.BiasedMF import BiasedMF, BiasedMF_PCFR
from models.GNN import LightGCN_MultiGraph, LightGCN_PCFR
from models.Discriminators import BinaryDiscriminator
from utils.generic import batch_to_gpu

DATASET     = 'ml100k'
DATA_PATH   = '../dataset/'
ALL_ATTRS   = ['u_gender', 'u_age', 'u_occupation', 'u_activity']
DISC_EPOCHS = 300
DISC_LR     = 0.001
DISC_L2     = 1e-4
U_VEC_SIZE  = 64
RANDOM_SEED = 2020
NUM_WORKER  = 0
BATCH_SIZE  = 256
VT_NUM_NEG  = 10

MODEL_DIR = '../model/'


class GNN_NoFilter(LightGCN_PCFR):
    """
    Loads a LightGCN_PCFR checkpoint but returns raw propagated embeddings
    (bypasses the adversarial filter). Used as a 'GNN None' baseline to show
    what GNN embeddings look like before any fairness intervention.
    """
    def get_user_vectors(self, uids):
        # Call grandparent (LightGCN_MultiGraph) — skips the filter layer
        return LightGCN_MultiGraph.get_user_vectors(self, uids)


MODEL_CLASS = {
    'None':         BiasedMF,
    'PCFR':         BiasedMF_PCFR,
    'PCFR_Multi':   BiasedMF_PCFR,
    'GNN_None':     GNN_NoFilter,
    'GNN_PCFR':     LightGCN_PCFR,
    'GNN_Multi':    LightGCN_PCFR,
}

# Build checkpoint paths
MODELS = {}

# BiasedMF None — single baseline row (no fairness training)
p = os.path.join(MODEL_DIR, 'cmp_BiasedMF_None_ml100k_u_gender', 'model.pt')
if os.path.exists(p):
    MODELS[('None', 'no_fairness')] = p

for attr in ALL_ATTRS:
    p = os.path.join(MODEL_DIR, f'BiasedMF_PCFR_ml100k_{attr}', 'model.pt')
    if os.path.exists(p):
        MODELS[('PCFR', attr)] = p
    p = os.path.join(MODEL_DIR, f'GNN_PCFR_ml100k_{attr}', 'model.pt')
    if os.path.exists(p):
        MODELS[('GNN_PCFR', attr)] = p

# GNN None — reuse u_gender PCFR checkpoint, bypass filter
p = os.path.join(MODEL_DIR, 'GNN_PCFR_ml100k_u_gender', 'model.pt')
if os.path.exists(p):
    MODELS[('GNN_None', 'no_fairness')] = p

p = os.path.join(MODEL_DIR, 'BiasedMF_PCFR_multiattr_ml100k', 'model.pt')
if os.path.exists(p): MODELS[('PCFR_Multi', 'all_attrs')] = p

p = os.path.join(MODEL_DIR, 'GNN_PCFR_multiattr_ml100k', 'model.pt')
if os.path.exists(p): MODELS[('GNN_Multi', 'all_attrs')] = p


def load_model(fw, model_path, feature_cols):
    dr = RecDataReader(path=DATA_PATH, dataset_name=DATASET,
                       feature_columns=feature_cols, sep='\t')
    train_dp = RecDataset(data_reader=dr, stage='train',
                          batch_size=BATCH_SIZE, num_neg=VT_NUM_NEG)
    test_dp  = RecDataset(data_reader=dr, stage='test',
                          batch_size=BATCH_SIZE, num_neg=VT_NUM_NEG)
    dp_dict  = {'train': train_dp, 'test': test_dp}

    torch.manual_seed(RANDOM_SEED)
    cls = MODEL_CLASS[fw]
    model = cls(data_processor_dict=dp_dict,
                user_num=len(dr.user_ids_set),
                item_num=len(dr.item_ids_set),
                u_vector_size=U_VEC_SIZE, i_vector_size=U_VEC_SIZE,
                random_seed=RANDOM_SEED, dropout=0.2, model_path=model_path)
    state = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def get_vectors(model, uids):
    if hasattr(model, 'get_user_vectors'):
        return model.get_user_vectors(uids)
    return model.uid_embeddings(uids)


def eval_leakage(model, eval_attr):
    disc_dr = DiscriminatorDataReader(path=DATA_PATH, dataset_name=DATASET,
                                      feature_columns=[eval_attr], sep='\t', test_ratio=0.2)
    train_dl = DataLoader(
        DiscriminatorDataset(data_reader=disc_dr, stage='train', batch_size=BATCH_SIZE),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER,
        collate_fn=DiscriminatorDataset.collate_fn)
    test_dl  = DataLoader(
        DiscriminatorDataset(data_reader=disc_dr, stage='test', batch_size=BATCH_SIZE),
        batch_size=BATCH_SIZE, num_workers=NUM_WORKER,
        collate_fn=DiscriminatorDataset.collate_fn)

    torch.manual_seed(RANDOM_SEED)
    disc = BinaryDiscriminator(U_VEC_SIZE, disc_dr.feature_info[1],
                               random_seed=RANDOM_SEED, dropout=0.2,
                               model_dir_path='/tmp', layers=3)
    disc.apply(disc.init_weights)
    opt = torch.optim.Adam(disc.parameters(), lr=DISC_LR, weight_decay=DISC_L2)

    best_auc = 0.5
    for _ in range(DISC_EPOCHS):
        disc.train()
        for batch in train_dl:
            batch = batch_to_gpu(batch)
            with torch.no_grad():
                vecs = get_vectors(model, batch['X'] - 1)
            lbl = batch['features'][:, 0]
            opt.zero_grad(); disc(vecs, lbl).backward(); opt.step()

        disc.eval()
        preds, lbls = [], []
        for batch in test_dl:
            batch = batch_to_gpu(batch)
            with torch.no_grad():
                vecs = get_vectors(model, batch['X'] - 1)
                lbl  = batch['features'][:, 0]
                pred = disc.predict(vecs)['output'].squeeze().detach()
            preds.append(pred.cpu().numpy())
            lbls.append(lbl.cpu().numpy())
        try:
            auc = roc_auc_score(np.concatenate(lbls), np.concatenate(preds))
            if auc > best_auc: best_auc = auc
        except Exception:
            pass
    return best_auc


def main():
    torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)

    frameworks   = ['None', 'PCFR', 'PCFR_Multi', 'GNN_None', 'GNN_PCFR', 'GNN_Multi']
    single_attrs = {
        'None':       ['no_fairness'],
        'PCFR':       ALL_ATTRS,
        'PCFR_Multi': ['all_attrs'],
        'GNN_None':   ['no_fairness'],
        'GNN_PCFR':   ALL_ATTRS,
        'GNN_Multi':  ['all_attrs'],
    }

    print("\n" + "="*92)
    print("  CROSS-ATTRIBUTE LEAKAGE — ml100k  |  BiasedMF PCFR vs GNN PCFR")
    print("  0.50 = no leakage  |  ← = trained attribute")
    print("="*92)

    results = {}
    for fw in frameworks:
        for trained_attr in single_attrs[fw]:
            key = (fw, trained_attr)
            if key not in MODELS: continue
            feat_cols = ALL_ATTRS if trained_attr == 'all_attrs' else [trained_attr]
            print(f"\n▶ {fw} / {trained_attr}  [{os.path.basename(MODELS[key])}]")
            try:
                model = load_model(fw, MODELS[key], feat_cols)
            except Exception as e:
                print(f"  ✗ Load failed: {e}"); continue
            aucs = {}
            for ea in ALL_ATTRS:
                sys.stdout.write(f"  {ea:28s} ... "); sys.stdout.flush()
                auc = eval_leakage(model, ea)
                aucs[ea] = auc
                marker = " ←" if ea == trained_attr else "  "
                print(f"AUC = {auc:.4f}{marker}")
            results[key] = aucs
            del model

    # Summary
    print("\n\n" + "="*92)
    print("  SUMMARY")
    print("="*92)
    cw = 14
    print(f"{'Framework':<14} {'Trained on':>16}  " + "  ".join(f"{a:>{cw}}" for a in ALL_ATTRS))
    print("-"*92)
    for fw in frameworks:
        for ta in single_attrs[fw]:
            key = (fw, ta)
            if key not in results: continue
            row = f"{fw:<14} {ta:>16}  " + "  ".join(f"{results[key][a]:>{cw}.4f}" for a in ALL_ATTRS)
            print(row)
    print("\n  ← trained attribute  |  0.50 = no leakage")
    print("="*92 + "\n")


if __name__ == '__main__':
    main()
