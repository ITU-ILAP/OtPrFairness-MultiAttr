# coding=utf-8
"""
Baseline training for ml100k dataset.

Trains all 6 frameworks (None, FOCF_ValUnf, FOCF_AbsUnf, PCFR, FairRec, FairPO)
on BiasedMF / ml100k / u_gender — same protocol as run_comparison.py on insurance.

Checkpoints saved to:
    ../model/cmp_BiasedMF_<fw>_ml100k_u_gender/model.pt

Usage (from src/):
    python ml100k_baseline_train.py
    # Runtime: ~30-40 min on CPU
"""

import subprocess
import sys
import os

DATASET    = 'ml100k'
FEATURE    = 'u_gender'
MODEL      = 'BiasedMF'
EPOCHS     = '100'
METRIC     = 'ndcg@3,f1@3'
VT_NUM_NEG = '10'
BATCH      = '1024'
LR         = '1e-3'
L2         = '1e-4'

FRAMEWORKS = ['None', 'FOCF_ValUnf', 'FOCF_AbsUnf', 'PCFR', 'FairRec', 'FairPO']

BASE_CMD = [
    'python', './main.py',
    '--data_processor',  'RecDataset',
    '--dataset',         DATASET,
    '--feature_columns', FEATURE,
    '--optimizer',       'Adam',
    '--metric',          METRIC,
    '--l2',              L2,
    '--lr',              LR,
    '--batch_size',      BATCH,
    '--runner',          'RecRunner',
    '--vt_num_neg',      VT_NUM_NEG,
    '--vt_batch_size',   BATCH,
    '--num_worker',      '0',
    '--epoch',           EPOCHS,
    '--eval_disc',
]


def run_framework(fw):
    model_dir  = f'../model/cmp_{MODEL}_{fw}_{DATASET}_{FEATURE}'
    model_path = f'{model_dir}/model.pt'
    os.makedirs(model_dir, exist_ok=True)

    if os.path.exists(model_path):
        print(f'  [SKIP] {fw} — checkpoint already exists at {model_path}')
        return

    cmd = BASE_CMD + [
        '--model_name',         MODEL,
        '--fairness_framework', fw,
        '--model_path',         model_path,
    ]
    if fw == 'FairRec':
        cmd += ['--fairrec_lambda', '0.05']
    if fw == 'FairPO':
        cmd += ['--fairpo_alpha', '1.0', '--fairpo_beta', '1.0']

    sep = '─' * 60
    print(f'\n{sep}')
    print(f'  Training: {MODEL} / {fw} / {DATASET} / {FEATURE}')
    print(f'{sep}')
    sys.stdout.flush()

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f'  ✗ {fw} failed (exit {result.returncode})')
    else:
        print(f'  ✓ {fw} → {model_path}')
    sys.stdout.flush()


if __name__ == '__main__':
    print('=' * 60)
    print(f'  ml100k Baseline Training')
    print(f'  Model={MODEL} | Dataset={DATASET} | Attr={FEATURE}')
    print('=' * 60)

    for fw in FRAMEWORKS:
        run_framework(fw)

    print('\n' + '=' * 60)
    print('  All baselines done.')
    print('  Checkpoints in ../model/cmp_BiasedMF_*_ml100k_u_gender/')
    print('=' * 60)
