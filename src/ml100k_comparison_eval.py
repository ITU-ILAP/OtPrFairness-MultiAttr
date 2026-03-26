# coding=utf-8
"""
Quality + Outcome Fairness Comparison — ml100k
Same protocol as multi_attr_eval.py but for ml100k.

Evaluates all saved checkpoints on:
  - NDCG@3, F1@3 (recommendation quality)
  - UGF per attribute (user-oriented group fairness)
  - ValUnf per attribute (value unfairness)

Usage (from src/):
    python ml100k_comparison_eval.py
"""

import os, sys, logging
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.ERROR)

from data_reader import RecDataReader
from datasets import RecDataset
from models.BiasedMF import (BiasedMF, BiasedMF_PCFR, BiasedMF_FairRec,
                              BiasedMF_FairPO, BiasedMF_FOCF_AbsUnf, BiasedMF_FOCF_ValUnf)
from models.GNN import LightGCN_PCFR
from utils.generic import batch_to_gpu
from utils.constants import LABEL, USER, ITEM
from utils.metrics import ndcg_at_k, value_unfairness, user_oriented_unfairness

DATASET     = 'ml100k'
DATA_PATH   = '../dataset/'
ALL_ATTRS   = ['u_gender', 'u_age', 'u_occupation', 'u_activity']
BATCH_SIZE  = 1024
VT_NUM_NEG  = 10
U_VEC_SIZE  = 64
RANDOM_SEED = 2020
NUM_WORKER  = 0
MODEL_DIR   = '../model/'

MODELS = [
    ('None',          f'cmp_BiasedMF_None_ml100k_u_gender/model.pt',         BiasedMF),
    ('FOCF_ValUnf',   f'cmp_BiasedMF_FOCF_ValUnf_ml100k_u_gender/model.pt',  BiasedMF_FOCF_ValUnf),
    ('FOCF_AbsUnf',   f'cmp_BiasedMF_FOCF_AbsUnf_ml100k_u_gender/model.pt',  BiasedMF_FOCF_AbsUnf),
    ('PCFR',          f'cmp_BiasedMF_PCFR_ml100k_u_gender/model.pt',          BiasedMF_PCFR),
    ('FairRec',       f'cmp_BiasedMF_FairRec_ml100k_u_gender/model.pt',       BiasedMF_FairRec),
    ('FairPO',        f'cmp_BiasedMF_FairPO_ml100k_u_gender/model.pt',        BiasedMF_FairPO),
    ('PCFR_Multi',    f'BiasedMF_PCFR_multiattr_ml100k/model.pt',             BiasedMF_PCFR),
    ('GNN_PCFR',      f'GNN_PCFR_ml100k_u_gender/model.pt',                   LightGCN_PCFR),
    ('GNN_Multi',     f'GNN_PCFR_multiattr_ml100k/model.pt',                  LightGCN_PCFR),
]


def build_model(label, model_class, dp_dict, user_num, item_num, ckpt):
    torch.manual_seed(RANDOM_SEED)
    m = model_class(data_processor_dict=dp_dict, user_num=user_num, item_num=item_num,
                    u_vector_size=U_VEC_SIZE, i_vector_size=U_VEC_SIZE,
                    random_seed=RANDOM_SEED, dropout=0.0, model_path=ckpt)
    state = torch.load(ckpt, map_location='cpu')
    m.load_state_dict(state)
    m.eval()
    return m


def evaluate_model(model, test_loader):
    records = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch_to_gpu(batch)
            out   = model.predict(batch)
            preds  = out['prediction'].cpu().numpy()
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
    for _, g in df.sort_values('score', ascending=False).groupby('uid'):
        labels = g['label'].tolist()
        ndcgs.append(ndcg_at_k(labels, k=3, method=1))
        overlap = float(np.sum(g['label'].values[:3]))
        total   = float(np.sum(g['label'].values))
        f1s.append(2 * overlap / (3 + total) if total > 0 else 0.0)
    return float(np.mean(ndcgs)), float(np.mean(f1s))


def ugf(df, attr):
    g0 = df[df[attr] == 0].rename(columns={'uid': USER, 'iid': ITEM, 'label': LABEL})
    g1 = df[df[attr] == 1].rename(columns={'uid': USER, 'iid': ITEM, 'label': LABEL})
    if g0.empty or g1.empty: return float('nan')
    return user_oriented_unfairness(g0, g1, metric='ndcg@3')


def valunf(df, attr):
    g0 = df[df[attr] == 0].rename(columns={'uid': USER, 'iid': ITEM, 'label': LABEL})
    g1 = df[df[attr] == 1].rename(columns={'uid': USER, 'iid': ITEM, 'label': LABEL})
    if g0.empty or g1.empty: return float('nan')
    return value_unfairness(g0, g1)


def main():
    torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)

    print("Loading data ...")
    dr = RecDataReader(path=DATA_PATH, dataset_name=DATASET,
                       feature_columns=ALL_ATTRS, sep='\t')
    user_num = len(dr.user_ids_set)
    item_num = len(dr.item_ids_set)

    dp_dict = {
        'train': RecDataset(data_reader=dr, stage='train',
                            batch_size=BATCH_SIZE, num_neg=1),
        'valid': RecDataset(data_reader=dr, stage='valid',
                            batch_size=BATCH_SIZE, num_neg=VT_NUM_NEG),
    }
    test_dp = RecDataset(data_reader=dr, stage='test',
                         batch_size=BATCH_SIZE, num_neg=VT_NUM_NEG)
    test_loader = DataLoader(test_dp, batch_size=None, num_workers=NUM_WORKER,
                             pin_memory=False, collate_fn=test_dp.collate_fn)

    results = []
    for label, rel_path, cls in MODELS:
        ckpt = os.path.join(MODEL_DIR, rel_path)
        if not os.path.exists(ckpt):
            print(f"  [SKIP] {label} — checkpoint not found")
            continue
        print(f"  [{label}] ...", end=' ', flush=True)
        model = build_model(label, cls, dp_dict, user_num, item_num, ckpt)
        df    = evaluate_model(model, test_loader)
        n3, f1 = ndcg3_f1(df)
        row = {'framework': label, 'ndcg@3': n3, 'f1@3': f1}
        for attr in ALL_ATTRS:
            row[f'ugf_{attr}']    = ugf(df, attr)
            row[f'valunf_{attr}'] = valunf(df, attr)
        results.append(row)
        print(f"NDCG@3={n3:.4f}")
        del model

    # ── Print tables ────────────────────────────────────────────────────────
    W = 94
    print("\n" + "="*W)
    print("  RECOMMENDATION QUALITY  —  ml100k")
    print(f"  {'Framework':<18} {'NDCG@3':>7} {'F1@3':>7}")
    print(f"  {'-'*34}")
    for r in results:
        print(f"  {r['framework']:<18} {r['ndcg@3']:>7.4f} {r['f1@3']:>7.4f}")

    for mk, ml in [('ugf', 'UGF (User-Oriented Group Fairness) ↓'),
                   ('valunf', 'Value Unfairness ↓')]:
        print("\n" + "="*W)
        print(f"  {ml}  —  ml100k")
        hdr = f"  {'Framework':<18}" + "".join(f" {a:>20}" for a in ALL_ATTRS) + f" {'Avg':>7}"
        print(hdr); print(f"  {'-'*90}")
        for r in results:
            vals = [r[f'{mk}_{a}'] for a in ALL_ATTRS]
            avg  = float(np.nanmean(vals))
            print(f"  {r['framework']:<18}" + "".join(f" {v:>20.4f}" for v in vals) + f" {avg:>7.4f}")

    print("\n" + "="*W)


if __name__ == '__main__':
    main()
