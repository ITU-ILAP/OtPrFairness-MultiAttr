# coding=utf-8
"""
AutoEncoder Evaluation Script
================================
Evaluates AutoEncoder (None, PCFR, PCFR_MultiAttr) on both insurance and ml100k
across all three dimensions:
  1. Recommendation quality  (NDCG@3, F1@3)
  2. Outcome fairness        (UGF, ValUnf per attribute)
  3. Process fairness        (cross-attribute leakage AUC)

Usage (from src/):
    python autoencoder_eval.py
    python autoencoder_eval.py --dataset insurance
    python autoencoder_eval.py --dataset ml100k
    python autoencoder_eval.py --skip_leakage   # quality + fairness only
"""

import os, sys, argparse, logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, ndcg_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.ERROR)

from data_reader import RecDataReader, DiscriminatorDataReader
from datasets import RecDataset, DiscriminatorDataset
from models.AutoEncoder import AutoEncoder, AutoEncoder_PCFR
from models.Discriminators import BinaryDiscriminator
from utils.generic import batch_to_gpu

# ── Config ─────────────────────────────────────────────────────────────────────

DATASET_ATTRS = {
    'insurance': ['u_gender', 'u_occupation', 'u_activity', 'u_marital_status'],
    'ml100k':    ['u_gender', 'u_age', 'u_occupation', 'u_activity'],
}

DATA_PATH    = '../dataset/'
MODEL_DIR    = '../model/'
U_VEC_SIZE   = 64
AE_HIDDEN    = 256
BATCH_SIZE   = 512
VT_NUM_NEG   = 100
RANDOM_SEED  = 2020
NUM_WORKER   = 0
DISC_EPOCHS  = 300
DISC_LR      = 1e-3
DISC_L2      = 1e-4


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(ckpt_path, dataset, feat_cols, with_filter):
    dr = RecDataReader(path=DATA_PATH, dataset_name=dataset,
                       feature_columns=feat_cols, sep='\t')
    train_dp = RecDataset(data_reader=dr, stage='train',
                          batch_size=BATCH_SIZE, num_neg=1)
    test_dp  = RecDataset(data_reader=dr, stage='test',
                          batch_size=BATCH_SIZE, num_neg=VT_NUM_NEG)
    dp_dict  = {'train': train_dp, 'test': test_dp}

    torch.manual_seed(RANDOM_SEED)
    cls   = AutoEncoder_PCFR if with_filter else AutoEncoder
    model = cls(data_processor_dict=dp_dict,
                user_num=len(dr.user_ids_set),
                item_num=len(dr.item_ids_set),
                u_vector_size=U_VEC_SIZE, i_vector_size=U_VEC_SIZE,
                ae_hidden=AE_HIDDEN, recon_weight=0.1, ae_dropout=0.5,
                random_seed=RANDOM_SEED, dropout=0.2, model_path=ckpt_path)
    state = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, dr, test_dp


# ── Recommendation quality ─────────────────────────────────────────────────────

def eval_quality(model, test_dp):
    loader = DataLoader(test_dp, batch_size=None, num_workers=NUM_WORKER,
                        collate_fn=test_dp.collate_fn)
    ndcgs, f1s = [], []
    n = 1 + VT_NUM_NEG
    with torch.no_grad():
        for batch in loader:
            batch  = batch_to_gpu(batch)
            preds  = model.predict(batch)['prediction']
            labels = batch_to_gpu(batch)['Y'] if 'Y' in batch else None
            for i in range(len(preds) // n):
                scores = preds[i * n: (i + 1) * n].cpu().numpy()
                rank   = int((scores > scores[0]).sum()) + 1
                # NDCG@3
                if rank <= 3:
                    ndcgs.append(1.0 / np.log2(rank + 1))
                else:
                    ndcgs.append(0.0)
                # F1@3 (hit-based)
                f1s.append(1.0 if rank <= 3 else 0.0)
    ndcg = float(np.mean(ndcgs))
    f1   = float(np.mean(f1s))
    return ndcg, f1


# ── Outcome fairness ───────────────────────────────────────────────────────────

def eval_outcome_fairness(model, dataset, all_attrs):
    """Compute UGF and ValUnf for all attributes using the test set."""
    # Load user attribute data
    user_df = pd.read_csv(
        os.path.join(DATA_PATH, dataset, 'user.csv'), sep='\t')
    test_df = pd.read_csv(
        os.path.join(DATA_PATH, dataset, 'test.csv'), sep='\t')

    dr = RecDataReader(path=DATA_PATH, dataset_name=dataset,
                       feature_columns=[all_attrs[0]], sep='\t')
    test_dp = RecDataset(data_reader=dr, stage='test',
                         batch_size=BATCH_SIZE, num_neg=VT_NUM_NEG)
    loader  = DataLoader(test_dp, batch_size=None, num_workers=NUM_WORKER,
                         collate_fn=test_dp.collate_fn)

    uid_list, iid_list, pred_list, label_list = [], [], [], []
    n = 1 + VT_NUM_NEG
    with torch.no_grad():
        for batch in loader:
            batch  = batch_to_gpu(batch)
            preds  = model.predict(batch)['prediction']
            x      = batch['X']
            y      = batch['Y']
            uid_list.extend(x[:, 0].cpu().numpy())
            iid_list.extend(x[:, 1].cpu().numpy())
            pred_list.extend(preds.cpu().numpy())
            label_list.extend(y.cpu().numpy())

    df = pd.DataFrame({'uid': uid_list, 'pred': pred_list, 'label': label_list})
    # merge user attributes
    df = df.merge(user_df[['uid'] + all_attrs], on='uid', how='left')

    ugf_dict    = {}
    valunf_dict = {}

    for attr in all_attrs:
        df_valid = df.dropna(subset=[attr])
        df_valid = df_valid[df_valid[attr].isin([0, 1])]
        g0 = df_valid[df_valid[attr] == 0]
        g1 = df_valid[df_valid[attr] == 1]
        if len(g0) == 0 or len(g1) == 0:
            ugf_dict[attr] = float('nan')
            valunf_dict[attr] = float('nan')
            continue

        # UGF: gap in mean NDCG@3 across groups
        def group_ndcg(gdf):
            hits = []
            for uid, grp in gdf.groupby('uid'):
                scores = grp['pred'].values
                rank   = int((scores > scores[0]).sum()) + 1
                hits.append(1.0 / np.log2(rank + 1) if rank <= 3 else 0.0)
            return float(np.mean(hits)) if hits else 0.0

        ugf_dict[attr] = abs(group_ndcg(g0) - group_ndcg(g1))

        # ValUnf: signed gap in mean (pred - label) across groups
        bias0 = (g0['pred'] - g0['label']).mean()
        bias1 = (g1['pred'] - g1['label']).mean()
        valunf_dict[attr] = abs(float(bias0) - float(bias1))

    return ugf_dict, valunf_dict


# ── Process fairness (leakage) ─────────────────────────────────────────────────

def eval_leakage(model, dataset, eval_attr):
    disc_dr = DiscriminatorDataReader(path=DATA_PATH, dataset_name=dataset,
                                      feature_columns=[eval_attr], sep='\t',
                                      test_ratio=0.2)
    train_dl = DataLoader(
        DiscriminatorDataset(data_reader=disc_dr, stage='train',
                             batch_size=BATCH_SIZE),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER,
        collate_fn=DiscriminatorDataset.collate_fn)
    test_dl  = DataLoader(
        DiscriminatorDataset(data_reader=disc_dr, stage='test',
                             batch_size=BATCH_SIZE),
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
                vecs = model.get_user_vectors(batch['X'] - 1)
            lbl = batch['features'][:, 0]
            opt.zero_grad(); disc(vecs, lbl).backward(); opt.step()

        disc.eval()
        preds, lbls = [], []
        for batch in test_dl:
            batch = batch_to_gpu(batch)
            with torch.no_grad():
                vecs = model.get_user_vectors(batch['X'] - 1)
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


# ── Main ───────────────────────────────────────────────────────────────────────

def run_dataset(dataset, skip_leakage):
    ALL_ATTRS = DATASET_ATTRS[dataset]

    # Models to evaluate: (label, ckpt_path, feat_cols, with_filter)
    models_to_eval = []

    none_path = os.path.join(MODEL_DIR, f'AE_None_{dataset}', 'model.pt')
    if os.path.exists(none_path):
        models_to_eval.append(('AE_None', none_path, [ALL_ATTRS[0]], False))

    for attr in ALL_ATTRS:
        p = os.path.join(MODEL_DIR, f'AE_PCFR_{dataset}_{attr}', 'model.pt')
        if os.path.exists(p):
            models_to_eval.append((f'AE_PCFR/{attr}', p, [attr], True))

    multi_path = os.path.join(MODEL_DIR, f'AE_PCFR_multiattr_{dataset}', 'model.pt')
    if os.path.exists(multi_path):
        models_to_eval.append(('AE_PCFR_Multi', multi_path, ALL_ATTRS, True))

    if not models_to_eval:
        print(f"  No checkpoints found for {dataset}. Run autoencoder_train.py first.")
        return

    print(f"\n{'='*80}")
    print(f"  AutoEncoder Results  |  {dataset}")
    print(f"{'='*80}")

    # ── 1. Recommendation Quality ──────────────────────────────────────────
    print(f"\n  RECOMMENDATION QUALITY")
    print(f"  {'Model':<20} {'NDCG@3':>8} {'F1@3':>8}")
    print(f"  {'-'*38}")
    quality = {}
    for label, ckpt, feat_cols, with_filter in models_to_eval:
        model, dr, test_dp = load_model(ckpt, dataset, feat_cols, with_filter)
        ndcg, f1 = eval_quality(model, test_dp)
        quality[label] = (ndcg, f1)
        print(f"  {label:<20} {ndcg:>8.4f} {f1:>8.4f}")
        del model

    # ── 2. Outcome Fairness ────────────────────────────────────────────────
    print(f"\n  UGF (User-Oriented Group Fairness) ↓")
    header = f"  {'Model':<20} " + " ".join(f"{a:>18}" for a in ALL_ATTRS) + f" {'Avg':>8}"
    print(header)
    print(f"  {'-'*( 20 + 19*len(ALL_ATTRS) + 9)}")
    ugf_results, valunf_results = {}, {}
    for label, ckpt, feat_cols, with_filter in models_to_eval:
        model, _, _ = load_model(ckpt, dataset, feat_cols, with_filter)
        ugf, valunf = eval_outcome_fairness(model, dataset, ALL_ATTRS)
        ugf_results[label]    = ugf
        valunf_results[label] = valunf
        vals = [ugf.get(a, float('nan')) for a in ALL_ATTRS]
        avg  = float(np.nanmean(vals))
        row  = f"  {label:<20} " + " ".join(f"{v:>18.4f}" for v in vals) + f" {avg:>8.4f}"
        print(row)
        del model

    print(f"\n  Value Unfairness ↓")
    print(header)
    print(f"  {'-'*( 20 + 19*len(ALL_ATTRS) + 9)}")
    for label in [m[0] for m in models_to_eval]:
        valunf = valunf_results[label]
        vals   = [valunf.get(a, float('nan')) for a in ALL_ATTRS]
        avg    = float(np.nanmean(vals))
        row    = f"  {label:<20} " + " ".join(f"{v:>18.4f}" for v in vals) + f" {avg:>8.4f}"
        print(row)

    # ── 3. Process Fairness — Leakage ─────────────────────────────────────
    if skip_leakage:
        return

    print(f"\n  CROSS-ATTRIBUTE LEAKAGE (Attacker AUC → 0.50)")
    print(f"  ← = trained attribute  |  0.50 = no leakage")
    header2 = f"  {'Model':<20} {'Trained on':>14} " + " ".join(f"{a:>18}" for a in ALL_ATTRS)
    print(header2)
    print(f"  {'-'*(20 + 15 + 19*len(ALL_ATTRS))}")

    for label, ckpt, feat_cols, with_filter in models_to_eval:
        model, _, _ = load_model(ckpt, dataset, feat_cols, with_filter)
        trained = feat_cols[0] if len(feat_cols) == 1 else 'all_attrs'
        aucs = {}
        sys.stdout.write(f"\n  {label:<20} {trained:>14}  ")
        sys.stdout.flush()
        for ea in ALL_ATTRS:
            sys.stdout.write(f"\n    probing {ea:<22} ... "); sys.stdout.flush()
            auc = eval_leakage(model, dataset, ea)
            aucs[ea] = auc
            marker = "←" if ea == trained else " "
            sys.stdout.write(f"AUC={auc:.4f} {marker}"); sys.stdout.flush()

        vals = " ".join(
            f"{aucs[a]:>17.4f}{'←' if a == trained else ' '}" for a in ALL_ATTRS)
        print(f"\n  {label:<20} {trained:>14}  {vals}")
        del model

    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='both',
                        choices=['insurance', 'ml100k', 'both'])
    parser.add_argument('--skip_leakage', action='store_true',
                        help='Skip the slow leakage eval, only do quality + fairness.')
    args = parser.parse_args()

    datasets = ['insurance', 'ml100k'] if args.dataset == 'both' else [args.dataset]
    torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)

    for ds in datasets:
        run_dataset(ds, args.skip_leakage)


if __name__ == '__main__':
    main()
