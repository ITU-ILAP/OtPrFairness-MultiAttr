# coding=utf-8
"""
GNN Evaluation Script  (Q1 + Q2)
==================================
Loads all GNN_PCFR checkpoints and produces:

  Q1 – Cross-attribute leakage AUC matrix (same protocol as cross_leakage_eval.py)
       Attacker probes the model's filtered GNN user embeddings.

  Q2 – Quality + outcome fairness comparison across all frameworks
       (GNN single-attr baseline, GNN_MultiAttr) evaluated consistently.

Usage (from src/):
    python gnn_eval.py

Requires:  gnn_pcfr_train.py to have been run first.
"""

import os
import sys
import logging
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.ERROR)

from data_reader import RecDataReader, DiscriminatorDataReader
from datasets import RecDataset, DiscriminatorDataset
from models.GNN import LightGCN_PCFR
from models.Discriminators import BinaryDiscriminator
from utils.generic import batch_to_gpu
from utils.constants import LABEL, USER, ITEM
from utils.metrics import ndcg_at_k, value_unfairness, user_oriented_unfairness

# ── Config ─────────────────────────────────────────────────────────────────────

DATASET      = 'insurance'
DATA_PATH    = '../dataset/'
ALL_ATTRS    = ['u_gender', 'u_occupation', 'u_activity', 'u_marital_status']

U_VEC_SIZE   = 64
N_GNN_LAYERS = 2
RANDOM_SEED  = 2020
BATCH_SIZE   = 256
VT_NUM_NEG   = 10
NUM_WORKER   = 0

DISC_EPOCHS  = 300
DISC_LR      = 0.001
DISC_L2      = 1e-4

MODEL_DIR = '../model/'

# Checkpoints for Q1 cross-leakage
SINGLE_ATTR_CKPTS = {
    attr: os.path.join(MODEL_DIR, f'GNN_PCFR_insurance_{attr}', 'model.pt')
    for attr in ALL_ATTRS
}
MULTI_ATTR_CKPT = os.path.join(MODEL_DIR, 'GNN_PCFR_multiattr_insurance', 'model.pt')

# For Q2 quality/fairness: use u_gender checkpoint as consistent baseline
Q2_CKPTS = [
    ('GNN_PCFR (u_gender ckpt)', SINGLE_ATTR_CKPTS['u_gender'], [ALL_ATTRS[0]]),
    ('GNN_PCFR_MultiAttr',       MULTI_ATTR_CKPT,               ALL_ATTRS),
]


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(ckpt_path, feature_cols):
    dr       = RecDataReader(path=DATA_PATH, dataset_name=DATASET,
                             feature_columns=feature_cols, sep='\t')
    train_dp = RecDataset(data_reader=dr, stage='train',
                          batch_size=BATCH_SIZE, num_neg=VT_NUM_NEG)
    valid_dp = RecDataset(data_reader=dr, stage='valid',
                          batch_size=BATCH_SIZE, num_neg=VT_NUM_NEG)
    dp_dict  = {'train': train_dp, 'valid': valid_dp}

    torch.manual_seed(RANDOM_SEED)
    model = LightGCN_PCFR(
        data_processor_dict=dp_dict,
        user_num=len(dr.user_ids_set),
        item_num=len(dr.item_ids_set),
        u_vector_size=U_VEC_SIZE,
        i_vector_size=U_VEC_SIZE,
        n_gnn_layers=N_GNN_LAYERS,
        random_seed=RANDOM_SEED,
        dropout=0.0,
        model_path=ckpt_path,
    )
    state = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    return model, dr


# ── Q1: Cross-attribute leakage ────────────────────────────────────────────────

def eval_leakage(model, eval_attr):
    """
    Train a fresh discriminator attacking model.get_user_vectors() for eval_attr.
    Returns best attacker AUC over DISC_EPOCHS epochs.
    """
    disc_dr = DiscriminatorDataReader(path=DATA_PATH, dataset_name=DATASET,
                                      feature_columns=[eval_attr], sep='\t',
                                      test_ratio=0.2)
    tr_dp = DiscriminatorDataset(data_reader=disc_dr, stage='train', batch_size=BATCH_SIZE)
    te_dp = DiscriminatorDataset(data_reader=disc_dr, stage='test',  batch_size=BATCH_SIZE)
    tr_ld = DataLoader(tr_dp, batch_size=BATCH_SIZE, shuffle=True,
                       collate_fn=DiscriminatorDataset.collate_fn)
    te_ld = DataLoader(te_dp, batch_size=BATCH_SIZE,
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
        for b in tr_ld:
            b = batch_to_gpu(b)
            with torch.no_grad():
                vecs = model.get_user_vectors(b['X'] - 1)
            lbl  = b['features'][:, 0]
            opt.zero_grad()
            disc(vecs, lbl).backward()
            opt.step()

        disc.eval()
        preds, labels = [], []
        for b in te_ld:
            b = batch_to_gpu(b)
            with torch.no_grad():
                vecs = model.get_user_vectors(b['X'] - 1)
                lbl  = b['features'][:, 0]
            preds.append(disc.predict(vecs)['output'].squeeze().cpu().numpy())
            labels.append(lbl.cpu().numpy())

        if preds:
            try:
                auc = roc_auc_score(np.concatenate(labels), np.concatenate(preds))
                best_auc = max(best_auc, auc)
            except Exception:
                pass

    return best_auc


def run_q1():
    print("\n" + "="*80)
    print("  Q1 — Cross-Attribute Leakage  |  GNN-PCFR")
    print("  Attacker probes the filtered GNN user embedding")
    print("="*80)

    results = {}
    rows = [(attr, SINGLE_ATTR_CKPTS[attr]) for attr in ALL_ATTRS
            if os.path.exists(SINGLE_ATTR_CKPTS[attr])]
    if os.path.exists(MULTI_ATTR_CKPT):
        rows.append(('all_attrs', MULTI_ATTR_CKPT))

    for trained_attr, ckpt in rows:
        feature_cols = ALL_ATTRS if trained_attr == 'all_attrs' else [trained_attr]
        print(f"\n▶ Trained on {trained_attr}")
        model, _ = load_model(ckpt, feature_cols)
        aucs = {}
        for eval_attr in ALL_ATTRS:
            sys.stdout.write(f"  probe {eval_attr:28s} ... ")
            sys.stdout.flush()
            auc = eval_leakage(model, eval_attr)
            aucs[eval_attr] = auc
            mark = " ←" if eval_attr == trained_attr else ""
            print(f"{auc:.4f}{mark}")
        results[trained_attr] = aucs
        del model

    # Summary table
    print("\n" + "="*80)
    print(f"{'Trained on':>20}  " + "  ".join(f"{a:>16}" for a in ALL_ATTRS))
    print("-"*80)
    for trained_attr, aucs in results.items():
        row = f"{trained_attr:>20}  " + \
              "  ".join(f"{aucs[a]:>16.4f}" for a in ALL_ATTRS)
        print(row)
    print("="*80)
    return results


# ── Q2: Quality + Outcome Fairness ─────────────────────────────────────────────

def collect_predictions(model, test_loader):
    """Run model on test set, return DataFrame with uid, iid, score, label, attrs."""
    records = []
    with torch.no_grad():
        for batch in test_loader:
            batch  = batch_to_gpu(batch)
            preds  = model.predict(batch)['prediction'].cpu().numpy()
            labels = batch[LABEL].cpu().numpy()
            uids   = batch['X'][:, 0].cpu().numpy()
            iids   = batch['X'][:, 1].cpu().numpy()
            feats  = batch['features'].cpu().numpy()
            for i in range(len(preds)):
                row = {'uid': int(uids[i]), 'iid': int(iids[i]),
                       'score': float(preds[i]), 'label': float(labels[i])}
                for j, attr in enumerate(ALL_ATTRS):
                    row[attr] = int(feats[i, j])
                records.append(row)
    return pd.DataFrame(records)


def ndcg3_f1(df):
    ndcgs, f1s = [], []
    for uid, g in df.sort_values('score', ascending=False).groupby('uid'):
        labels = g['label'].tolist()
        ndcgs.append(ndcg_at_k(labels, k=3, method=1))
        overlap = float(np.sum(g['label'].values[:3]))
        total   = float(np.sum(g['label'].values))
        f1s.append(2 * overlap / (3 + total) if total > 0 else 0.0)
    return float(np.mean(ndcgs)), float(np.mean(f1s))


def ugf_metric(df, attr):
    g0 = df[df[attr] == 0].rename(columns={'uid': USER, 'iid': ITEM, 'label': LABEL})
    g1 = df[df[attr] == 1].rename(columns={'uid': USER, 'iid': ITEM, 'label': LABEL})
    if g0.empty or g1.empty:
        return float('nan')
    return user_oriented_unfairness(g0, g1, metric='ndcg@3')


def valunf_metric(df, attr):
    g0 = df[df[attr] == 0].rename(columns={'uid': USER, 'iid': ITEM, 'label': LABEL})
    g1 = df[df[attr] == 1].rename(columns={'uid': USER, 'iid': ITEM, 'label': LABEL})
    if g0.empty or g1.empty:
        return float('nan')
    return value_unfairness(g0, g1)


def run_q2():
    print("\n" + "="*80)
    print("  Q2 — Quality + Outcome Fairness  |  GNN-PCFR vs BiasedMF-PCFR")
    print("  All models evaluated on same test set (consistent comparison)")
    print("="*80)

    results = []
    for label, ckpt, feature_cols in Q2_CKPTS:
        if not os.path.exists(ckpt):
            print(f"  ✗ Checkpoint not found for {label}: {ckpt}")
            continue

        print(f"\n  [{label}] ...", end=' ', flush=True)
        model, dr = load_model(ckpt, ALL_ATTRS)

        test_dp = RecDataset(data_reader=dr, stage='test',
                             batch_size=BATCH_SIZE, num_neg=VT_NUM_NEG)
        test_loader = DataLoader(test_dp, batch_size=None, num_workers=NUM_WORKER,
                                 pin_memory=False, collate_fn=test_dp.collate_fn)

        df       = collect_predictions(model, test_loader)
        n3, f1   = ndcg3_f1(df)
        row = {'framework': label, 'ndcg@3': n3, 'f1@3': f1}
        for attr in ALL_ATTRS:
            row[f'ugf_{attr}']    = ugf_metric(df, attr)
            row[f'valunf_{attr}'] = valunf_metric(df, attr)
        results.append(row)
        print(f"NDCG@3={n3:.4f}")
        del model

    # Print tables
    print("\n" + "="*80)
    print("  RECOMMENDATION QUALITY")
    print(f"  {'Framework':<28} {'NDCG@3':>7} {'F1@3':>7}")
    print(f"  {'-'*44}")
    for r in results:
        print(f"  {r['framework']:<28} {r['ndcg@3']:>7.4f} {r['f1@3']:>7.4f}")

    for key, title in [('ugf', 'UGF ↓'), ('valunf', 'Value Unfairness ↓')]:
        print(f"\n{'─'*80}")
        print(f"  {title}")
        hdr = f"  {'Framework':<28}" + "".join(f" {a:>18}" for a in ALL_ATTRS) + f" {'Avg':>7}"
        print(hdr)
        for r in results:
            vals = [r[f'{key}_{a}'] for a in ALL_ATTRS]
            avg  = float(np.nanmean(vals))
            row  = f"  {r['framework']:<28}" + \
                   "".join(f" {v:>18.4f}" for v in vals) + \
                   f" {avg:>7.4f}"
            print(row)

    print("\n" + "="*80)
    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    missing = [attr for attr, p in SINGLE_ATTR_CKPTS.items() if not os.path.exists(p)]
    if missing:
        print(f"[!] Missing checkpoints for: {missing}")
        print("    Run  python gnn_pcfr_train.py  first.")
        sys.exit(1)

    q1_results = run_q1()
    q2_results = run_q2()


if __name__ == '__main__':
    main()
