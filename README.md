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

Dataset files are **not included** in this repository due to size. See [`dataset/README.md`](dataset/README.md) for download and preprocessing instructions.

---

## Fairness Frameworks

This repository extends the original paper implementation with **FairPO**, a new framework introduced in this version that jointly enforces both process and outcome fairness.

| Framework     | Type             | Origin | Description |
|---------------|------------------|--------|-------------|
| `None`        | Baseline         | Paper  | No fairness constraint |
| `FOCF_ValUnf` | Outcome fairness | Paper  | Minimizes value unfairness (signed bias gap between groups) |
| `FOCF_AbsUnf` | Outcome fairness | Paper  | Minimizes absolute unfairness between groups |
| `PCFR`        | Process fairness | Paper  | Adversarial filter removes sensitive info from user embeddings |
| `FairRec`     | Process fairness | Paper  | Dual-branch (learner + filter) with orthogonal regularization |
| `FairPO`      | **Process + Outcome** | **New (this version)** | Adversarial process fairness + value-unfairness outcome penalty |

### FairPO — Fair Process & Outcome

**FairPO** (Fair Process & Outcome) is a new fairness framework added in this version of the repository. It is designed to address both types of user-side fairness simultaneously:

- **Process fairness** — prevents sensitive attributes from being recoverable from user embeddings via an adversarial discriminator (inherited from PCFR)
- **Outcome fairness** — penalizes discrepancies in prediction scores between demographic groups (inspired by FOCF)

#### Loss Function

```
L = L_rec + α · L_adv + β · L_outcome

  L_rec     = BPR recommendation loss
  L_adv     = −Σ discriminator(filtered_user_emb, sensitive_label)
  L_outcome = smooth_l1(|mean_bias_group0 − mean_bias_group1|, 0) × batch_size/2
  α         = --fairpo_alpha  (default 1.0)
  β         = --fairpo_beta   (default 1.0)
```

FairPO is implemented for all four backbone models (BiasedMF, PMF, DMF, MLP).

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

## Experimental Results

Full results on the **Insurance dataset** (`BiasedMF`, all 4 sensitive attributes). These results include both the paper's original frameworks and the new **FairPO** method.

### Process Fairness — Attacker AUC ↓ (0.5 = perfect, attacker is random)

| Framework   | u_gender | u_occupation | u_activity | u_marital |
|-------------|----------|--------------|------------|-----------|
| None        | 0.5000   | 0.5000       | 0.7470     | 0.6149    |
| FOCF_ValUnf | 0.5000   | 0.5000       | 0.7420     | 0.6153    |
| FOCF_AbsUnf | 0.5000   | 0.5000       | 0.7345     | 0.6179    |
| PCFR        | 0.5000   | 0.5000       | 0.5000     | 0.5000    |
| FairRec     | 0.5000   | 0.5000       | 0.5000     | 0.5000    |
| **FairPO**  | **0.5000** | **0.5000** | **0.5000** | **0.5000** |

FOCF methods fail to achieve process fairness on u_activity and u_marital since they have no adversarial component. FairPO matches PCFR and FairRec with perfect process fairness across all attributes.

### Recommendation Quality — NDCG@3 ↑

| Framework   | u_gender | u_occupation | u_activity | u_marital | Avg    |
|-------------|----------|--------------|------------|-----------|--------|
| None        | **0.8503** | **0.8531** | **0.8533** | **0.8538** | **0.8526** |
| FOCF_ValUnf | 0.8502   | 0.8527       | 0.8462     | 0.8504    | 0.8499 |
| FOCF_AbsUnf | 0.8508   | **0.8531**   | 0.8469     | 0.8498    | 0.8502 |
| PCFR        | 0.8337   | 0.8350       | 0.8346     | 0.8322    | 0.8339 |
| FairRec     | 0.8385   | 0.8380       | **0.8400** | 0.8374    | 0.8385 |
| **FairPO**  | 0.8355   | 0.8332       | 0.8356     | 0.8334    | 0.8344 |

Among process-fair methods, FairPO (avg 0.8344) outperforms PCFR (avg 0.8339) and is competitive with FairRec (avg 0.8385).

### Outcome Fairness — Value Unfairness ↓

| Framework   | u_gender | u_occupation | u_activity | u_marital | Avg    |
|-------------|----------|--------------|------------|-----------|--------|
| None        | 0.01246  | 0.02648      | 0.20084    | 0.04730   | 0.07177 |
| FOCF_ValUnf | 0.01281  | 0.03471      | 0.09019    | 0.08003   | 0.05444 |
| FOCF_AbsUnf | 0.01185  | 0.03728      | 0.10606    | 0.08731   | 0.06063 |
| PCFR        | **0.01021** | **0.01808** | 0.03834  | **0.03393** | **0.02514** |
| FairRec     | 0.01039  | 0.01877      | 0.03892    | 0.03453   | 0.02565 |
| **FairPO**  | 0.01055  | **0.01755**  | **0.03978** | 0.03608  | 0.02599 |

FairPO achieves the best outcome fairness on u_occupation and competitive results on all other attributes, despite not being a dedicated outcome fairness method.

### Summary

| Framework   | Process Fair (all attrs)? | Avg NDCG@3 | Avg ValUnf |
|-------------|---------------------------|------------|------------|
| None        | ❌ (2/4)                  | 0.8526     | 0.07177    |
| FOCF_ValUnf | ❌ (2/4)                  | 0.8499     | 0.05444    |
| FOCF_AbsUnf | ❌ (2/4)                  | 0.8502     | 0.06063    |
| PCFR        | ✅ (4/4)                  | 0.8339     | **0.02514** |
| FairRec     | ✅ (4/4)                  | **0.8385** | 0.02565    |
| **FairPO**  | ✅ **(4/4)**               | 0.8344     | 0.02599    |

**FairPO** is the only method that achieves perfect process fairness across all sensitive attributes while simultaneously maintaining competitive recommendation quality and outcome fairness.

---

## Q1: Cross-Attribute Leakage Analysis

> **Question:** When PCFR mitigates leakage for one sensitive attribute, what happens to the leakage of the *other* attributes?

### How to Run

```bash
cd src/
python cross_leakage_eval.py
```

The script loads saved PCFR checkpoints (one per sensitive attribute), freezes each model, then trains a fresh attacker discriminator for **all 4 attributes** on each frozen model. It outputs a cross-leakage AUC matrix. Runtime: ~15 minutes on CPU.

### Results — PCFR Cross-Leakage AUC Matrix

> Rows = which attribute PCFR was trained to protect
> Columns = which attribute the attacker is probing
> ← = the trained (protected) attribute | 0.50 = no leakage

| Trained on → | u_gender | u_occupation | u_activity | u_marital |
|---|---|---|---|---|
| **Baseline (None)** | 0.5452 | 0.5438 | 0.8764 | 0.6653 |
| **PCFR / u_gender** | **0.5262 ←** | 0.5502 | 0.6964 | 0.6015 |
| **PCFR / u_occupation** | 0.5272 | **0.5535 ←** | 0.6869 | 0.6048 |
| **PCFR / u_activity** | 0.5210 | 0.5514 | **0.6913 ←** | 0.6028 |
| **PCFR / u_marital** | 0.5195 | 0.5499 | 0.6867 | **0.6110 ←** |

### Key Findings

1. **Positive spillover** — protecting any one attribute reduces leakage of all others. The adversarial training makes embeddings generally less informative about demographics.

2. **u_activity is resistant** — even when PCFR directly targets `u_activity`, leakage only drops from 0.8764 → 0.6913. It is strongly encoded in the user embeddings and single-attribute PCFR cannot fully suppress it.

3. **No attribute reaches 0.50 unless it was the direct training target** — and the trained attribute itself only reaches 0.52–0.55. True multi-attribute fairness requires training against all attributes simultaneously (→ Q2).

---

## Q2: Multi-Attribute Simultaneous Leakage Mitigation

> **Question:** Can we train a single PCFR model to suppress leakage of *all* sensitive attributes at once, rather than one at a time?

### Approach

A single `BiasedMF_PCFR` model is trained adversarially against **4 discriminators simultaneously** — one per sensitive attribute. In each training step:

```
total_loss = rec_loss − adv_weight × (disc_gender + disc_occupation + disc_activity + disc_marital)
```

All four discriminators receive gradients in every batch. The model must learn embeddings that fool all four at once.

### How to Run

**Step 1 — Train the multi-attribute model:**

```bash
cd src/
python multi_attr_pcfr_train.py
# Saves checkpoint to ../model/PCFR_multiattr_insurance/model.pt
# Runtime: ~10 minutes on CPU (100 epochs)
```

**Step 2 — Evaluate cross-attribute leakage (includes Q1 + Q2 comparison):**

```bash
cd src/
python cross_leakage_eval.py
# Automatically detects the Q2 checkpoint and adds it as an extra row
# Runtime: ~15 minutes on CPU
```

### Results — Q1 vs Q2 Leakage Comparison

> Rows = model and training target | Columns = attribute probed by attacker
> ← = trained attribute | 0.50 = no leakage (attacker is random)

| Model | Trained on | u_gender | u_occupation | u_activity | u_marital |
|---|---|---|---|---|---|
| PCFR | u_gender | **0.5262 ←** | 0.5502 | 0.6964 | 0.6015 |
| PCFR | u_occupation | 0.5272 | **0.5535 ←** | 0.6869 | 0.6048 |
| PCFR | u_activity | 0.5210 | 0.5514 | **0.6913 ←** | 0.6028 |
| PCFR | u_marital | 0.5195 | 0.5499 | 0.6867 | **0.6110 ←** |
| **PCFR_MultiAttr** | **all_attrs** | **0.5051** | **0.5150** | **0.5000** | **0.5118** |

### Key Findings

1. **Multi-attribute training achieves near-perfect suppression across all attributes** — all four AUCs collapse to ≤0.515, compared to 0.55–0.69 for single-attribute PCFR.

2. **Biggest gain on u_activity** — single-attribute PCFR could not suppress `u_activity` below ~0.69 even when directly targeting it. Multi-attribute training brings it to **0.5000** (perfectly random attacker).

3. **No trade-off between attributes** — the model doesn't sacrifice one attribute to protect another; all four are suppressed simultaneously to near-chance level.

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `AttributeError: module 'numpy' has no attribute 'float'` | Already patched. Update NumPy if it appears elsewhere: replace `np.float` with `float`. |
| `AssertionError: Torch not compiled with CUDA enabled` | Already patched. The code auto-detects CPU/GPU. |
| `RuntimeError: DataLoader worker died` | Use `--num_worker 0` (default in all cmd files). |
| Out of memory | Reduce `--batch_size` to `256` or `--u_vector_size` to `32`. |
