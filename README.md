# OtPrFairness-MultiAttr

Implementation for the paper:

> **"Investigating User-Side Fairness in Outcome and Process for Multi-Type Sensitive Attributes in Recommendations"**
> ACM Transactions on Recommender Systems, 2025 — Hong Kong Baptist University

This repo implements and compares six fairness frameworks for recommender systems, including **FairPO** — a new method that jointly enforces both *process fairness* (adversarial learning) and *outcome fairness* (value-unfairness regularization).

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Setup](#setup)
3. [Datasets](#datasets)
4. [Fairness Frameworks](#fairness-frameworks)
5. [Quick Start](#quick-start)
6. [Running Individual Experiments](#running-individual-experiments)
7. [Running a Full Comparison](#running-a-full-comparison)
8. [Key Arguments Reference](#key-arguments-reference)
9. [Fairness Metrics](#fairness-metrics)
10. [Expected Results](#expected-results)
11. [Troubleshooting](#troubleshooting)

---

## Repository Structure

```
OtPrFairness-MultiAttr/
├── dataset/
│   ├── insurance/          # Insurance recommendation dataset
│   └── ml1M/               # MovieLens 1M dataset
├── src/
│   ├── main.py             # Entry point — runs a single experiment
│   ├── run_comparison.py   # Runs all 6 frameworks and prints a comparison table
│   ├── runner.py           # Training loops for all frameworks
│   └── models/
│       ├── BiasedMF.py     # Biased Matrix Factorization + fairness variants
│       ├── PMF.py          # Probabilistic Matrix Factorization
│       ├── DMF.py          # Deep Matrix Factorization
│       └── MLP.py          # Multi-Layer Perceptron
├── cmd/
│   └── BiasedMF/
│       ├── exp_insurance.txt        # Commands for original frameworks on insurance
│       ├── exp_insurance_FairPO.txt # Commands for FairPO on insurance
│       ├── exp_ml1M.txt
│       └── exp_ml1M_FairPO.txt
├── model/                  # Saved checkpoints (auto-created)
└── log/                    # Training logs (auto-created)
```

---

## Setup

### Requirements

- Python 3.8+
- PyTorch (CPU is fine — no GPU required)
- numpy, pandas, scikit-learn, tqdm

### Install

```bash
git clone https://github.com/ITU-ILAP/OtPrFairness-MultiAttr.git
cd OtPrFairness-MultiAttr
pip install torch numpy pandas scikit-learn tqdm
```

---

## Datasets

| Dataset   | Users | Items | Interactions | Sensitive Attributes |
|-----------|-------|-------|--------------|----------------------|
| insurance | 1,511 | 12    | 27,180       | u_gender, u_activity, u_marital_status, u_occupation |
| ml1M      | 6,040 | 3,416 | 999,611      | u_gender, u_age |

Datasets are already in the `dataset/` folder — no additional download or preprocessing needed.

---

## Fairness Frameworks

| Framework     | Type             | Description |
|---------------|------------------|-------------|
| `None`        | Baseline         | No fairness constraint |
| `FOCF_ValUnf` | Outcome fairness | Minimizes value unfairness (signed bias gap between groups) |
| `FOCF_AbsUnf` | Outcome fairness | Minimizes absolute unfairness between groups |
| `PCFR`        | Process fairness | Adversarial filter removes sensitive info from user embeddings |
| `FairRec`     | Process fairness | Dual-branch (learner + filter) with orthogonal regularization |
| `FairPO`      | **Both**         | **New:** adversarial process fairness + value-unfairness outcome penalty |

### FairPO Loss Function

```
L = L_rec + α · L_adv + β · L_outcome

  L_rec     = BPR recommendation loss
  L_adv     = −Σ discriminator(filtered_user_emb, sensitive_label)
  L_outcome = smooth_l1(|mean_bias_group0 − mean_bias_group1|, 0) × batch_size/2
  α         = --fairpo_alpha  (default 1.0)
  β         = --fairpo_beta   (default 1.0)
```

---

## Quick Start

**All commands must be run from the `src/` directory:**

```bash
cd src/
```

### No fairness (baseline)

```bash
python main.py \
  --model_name BiasedMF --fairness_framework None \
  --dataset insurance --feature_columns u_gender \
  --optimizer Adam --metric ndcg@3,f1@3 \
  --lr 1e-3 --l2 1e-4 --batch_size 1024 \
  --epoch 100 --eval_disc
```

### FairPO (process + outcome fairness)

```bash
python main.py \
  --model_name BiasedMF --fairness_framework FairPO \
  --dataset insurance --feature_columns u_gender \
  --optimizer Adam --metric ndcg@3,f1@3 \
  --lr 1e-3 --l2 1e-4 --batch_size 1024 \
  --epoch 100 --eval_disc \
  --fairpo_alpha 1.0 --fairpo_beta 1.0
```

### PCFR (process fairness only)

```bash
python main.py \
  --model_name BiasedMF --fairness_framework PCFR \
  --dataset insurance --feature_columns u_gender \
  --optimizer Adam --metric ndcg@3,f1@3 \
  --lr 1e-3 --l2 1e-4 --batch_size 1024 \
  --epoch 100 --eval_disc
```

### FairRec

```bash
python main.py \
  --model_name BiasedMF --fairness_framework FairRec \
  --dataset insurance --feature_columns u_gender \
  --optimizer Adam --metric ndcg@3,f1@3 \
  --lr 1e-3 --l2 1e-4 --batch_size 1024 \
  --epoch 100 --eval_disc \
  --fairrec_lambda 0.05
```

### FOCF (outcome fairness)

```bash
# Value unfairness variant
python main.py \
  --model_name BiasedMF --fairness_framework FOCF_ValUnf \
  --dataset insurance --feature_columns u_gender \
  --optimizer Adam --metric ndcg@3,f1@3 \
  --lr 1e-3 --l2 1e-4 --batch_size 1024 \
  --epoch 100 --eval_disc

# Absolute unfairness variant
python main.py \
  --model_name BiasedMF --fairness_framework FOCF_AbsUnf \
  --dataset insurance --feature_columns u_gender \
  --optimizer Adam --metric ndcg@3,f1@3 \
  --lr 1e-3 --l2 1e-4 --batch_size 1024 \
  --epoch 100 --eval_disc
```

---

## Running Individual Experiments

Pre-written command files are in `cmd/BiasedMF/`. They run all sensitive attributes for one dataset.

```bash
cd src/

# Run all original frameworks on insurance dataset (all 4 sensitive attributes)
while IFS= read -r line; do
  [[ "$line" == \#* || -z "$line" ]] && continue
  eval "$line"
done < ../cmd/BiasedMF/exp_insurance.txt

# Run FairPO on insurance dataset
while IFS= read -r line; do
  [[ "$line" == \#* || -z "$line" ]] && continue
  eval "$line"
done < ../cmd/BiasedMF/exp_insurance_FairPO.txt
```

> **Tip:** The cmd files use `--epoch 1000` for full replication. For quick testing, add `--epoch 100` to any command.

---

## Running a Full Comparison

The comparison script trains all 6 frameworks and prints a results table:

```bash
cd src/
python run_comparison.py
```

This runs `BiasedMF` on `insurance / u_gender` for 100 epochs each and outputs:

```
====================================================================================================
                                        COMPARISON TABLE
              Model=BiasedMF | Dataset=insurance | Attribute=u_gender | Epochs=100
====================================================================================================
Framework          NDCG@3          F1@3       ValUnf↓       AbsUnf↓       UsrUnf↓  DiscAUC→0.5
----------------------------------------------------------------------------------------------------
None               0.8503        0.4592        0.0125        0.0125        0.0056          N/A
FOCF_ValUnf        0.8502        0.4578        0.0128        0.0128        0.0054          N/A
FOCF_AbsUnf        0.8508        0.4599        0.0119        0.0119        0.0056          N/A
PCFR               0.8337        0.4521        0.0102        0.0102        0.0215       0.5001
FairRec            0.8385        0.4541        0.0104        0.0104        0.0134       0.5000
FairPO             0.8355        0.4592        0.0105        0.0105        0.0201       0.5001
====================================================================================================
```

**Expected runtime:** ~20–30 minutes on CPU.

---

## Key Arguments Reference

| Argument               | Default   | Description |
|------------------------|-----------|-------------|
| `--model_name`         | `BiasedMF`| Model: `BiasedMF`, `PMF`, `DMF`, `MLP` |
| `--fairness_framework` | `None`    | Framework (see table above) |
| `--dataset`            | —         | `insurance` or `ml1M` |
| `--feature_columns`    | —         | Sensitive attribute (e.g. `u_gender`) |
| `--epoch`              | `100`     | Training epochs |
| `--lr`                 | `0.001`   | Learning rate |
| `--l2`                 | `1e-5`    | L2 regularization |
| `--batch_size`         | `128`     | Training batch size |
| `--optimizer`          | `GD`      | `GD` (SGD), `Adam`, `Adagrad` |
| `--metric`             | `RMSE`    | Evaluation metric(s), e.g. `ndcg@3,f1@3` |
| `--eval_disc`          | flag      | Evaluate process fairness with a trained discriminator |
| `--u_vector_size`      | `64`      | User embedding dimension |
| `--i_vector_size`      | `64`      | Item embedding dimension |
| `--vt_num_neg`         | `100`     | Negatives per positive at val/test time |
| `--fairpo_alpha`       | `1.0`     | FairPO: adversarial penalty weight |
| `--fairpo_beta`        | `1.0`     | FairPO: outcome-fairness penalty weight |
| `--fairrec_lambda`     | `0.5`     | FairRec: combined fairness loss weight |
| `--num_worker`         | `0`       | DataLoader workers (keep at 0 on macOS/Windows) |

---

## Fairness Metrics

| Metric | Better | Description |
|--------|--------|-------------|
| **ValUnf** | lower ↓ | Signed gap in mean (prediction − label) between groups |
| **AbsUnf** | lower ↓ | Absolute gap in mean \|prediction − label\| between groups |
| **UsrUnf** | lower ↓ | Gap in mean NDCG between the two user groups |
| **CGU** | higher ↑ | Calibrated Group-wise Utility: relative utility difference between groups vs baseline |
| **DiscAUC** | → 0.5 | AUC of a discriminator predicting the sensitive attribute from embeddings; 0.5 = perfectly fair (process fairness) |

---

## Expected Results

`BiasedMF`, `insurance`, `u_gender`, 100 epochs:

| Framework   | NDCG@3 | ValUnf | UsrUnf | DiscAUC |
|-------------|--------|--------|--------|---------|
| None        | 0.8503 | 0.0125 | 0.0056 | —       |
| FOCF_ValUnf | 0.8502 | 0.0128 | 0.0054 | —       |
| FOCF_AbsUnf | 0.8508 | 0.0119 | 0.0056 | —       |
| PCFR        | 0.8337 | 0.0102 | 0.0215 | 0.5001  |
| FairRec     | 0.8385 | 0.0104 | 0.0134 | 0.5000  |
| FairPO      | 0.8355 | 0.0105 | 0.0201 | 0.5001  |

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `AttributeError: module 'numpy' has no attribute 'float'` | Already patched. Update NumPy if it appears elsewhere: replace `np.float` with `float`. |
| `AssertionError: Torch not compiled with CUDA enabled` | Already patched. The code auto-detects CPU/GPU. |
| `RuntimeError: DataLoader worker died` | Use `--num_worker 0` (default in all cmd files). |
| Out of memory | Reduce `--batch_size` to `256` or `--u_vector_size` to `32`. |
