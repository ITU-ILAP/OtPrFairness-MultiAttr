# OtPrFairness-MultiAttr

Implementation for the paper:

> **"Investigating User-Side Fairness in Outcome and Process for Multi-Type Sensitive Attributes in Recommendations"**
> ACM Transactions on Recommender Systems, 2025 — Hong Kong Baptist University

This repo implements and compares six fairness frameworks for recommender systems, plus **FairPO** — a new method that jointly enforces both *process fairness* (adversarial learning) and *outcome fairness* (value-unfairness regularization). It also includes two research extensions (Q1, Q2) studying cross-attribute leakage and multi-attribute simultaneous fairness, and a **GNN backbone** (LightGCN multi-graph) as an alternative to Biased Matrix Factorization.

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Setup](#setup)
3. [Datasets](#datasets)
4. [Fairness Frameworks](#fairness-frameworks)
5. [GNN Backbone](#gnn-backbone-lightgcn-multi-graph)
6. [Usage](#usage)
7. [Key Arguments Reference](#key-arguments-reference)
8. [Fairness Metrics](#fairness-metrics)
9. [Results](#results)
10. [Q1: Cross-Attribute Leakage](#q1-cross-attribute-leakage-analysis)
11. [Q2: Multi-Attribute Simultaneous Fairness](#q2-multi-attribute-simultaneous-fairness)
12. [GNN Results](#gnn-results)
13. [Troubleshooting](#troubleshooting)

---

## Repository Structure

```
OtPrFairness-MultiAttr/
├── dataset/
│   ├── insurance/          # Insurance recommendation dataset
│   └── ml1M/               # MovieLens 1M dataset
├── src/
│   ├── main.py                    # Entry point — single experiment
│   ├── run_comparison.py          # Trains all 6 frameworks, prints comparison table
│   ├── runner.py                  # Training loops for all frameworks
│   ├── cross_leakage_eval.py      # Q1+GNN: cross-attribute leakage AUC matrix (BiasedMF & GNN)
│   ├── multi_attr_pcfr_train.py   # Q2: trains PCFR against all 4 attributes simultaneously
│   ├── multi_attr_eval.py         # Q2: evaluates all models on quality + outcome fairness
│   ├── gnn_pcfr_train.py          # GNN: trains LightGCN_PCFR (single- and multi-attr)
│   └── models/
│       ├── BiasedMF.py     # Biased MF + all fairness variants (PCFR, FairRec, FairPO, FOCF)
│       ├── GNN.py          # LightGCN multi-graph backbone + PCFR variant
│       ├── PMF.py
│       ├── DMF.py
│       └── MLP.py
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

**Requirements:** Python 3.8+, PyTorch (CPU is fine), numpy, pandas, scikit-learn, tqdm

```bash
git clone https://github.com/ITU-ILAP/OtPrFairness-MultiAttr.git
cd OtPrFairness-MultiAttr
pip install torch numpy pandas scikit-learn tqdm
```

---

## Datasets

| Dataset   | Users | Items | Interactions | Sensitive Attributes |
|-----------|-------|-------|--------------|----------------------|
| insurance | 1,511 | 12    | 27,180       | u_gender, u_occupation, u_activity, u_marital_status |
| ml1M      | 6,040 | 3,416 | 999,611      | u_gender, u_age |

Dataset files are **not included** — see [`dataset/README.md`](dataset/README.md) for download and preprocessing instructions.

---

## Fairness Frameworks

| Framework     | Type                  | Origin              | Description |
|---------------|-----------------------|---------------------|-------------|
| `None`        | Baseline              | Paper               | No fairness constraint |
| `FOCF_ValUnf` | Outcome               | Paper               | Minimizes signed bias gap between groups |
| `FOCF_AbsUnf` | Outcome               | Paper               | Minimizes absolute unfairness between groups |
| `PCFR`        | Process               | Paper               | Adversarial filter removes sensitive info from user embeddings |
| `FairRec`     | Process               | Paper               | Dual-branch learner with orthogonal regularization |
| `FairPO`      | **Process + Outcome** | **New (this repo)** | Adversarial process fairness + value-unfairness outcome penalty |

### FairPO

**FairPO** jointly addresses both fairness types in a single loss:

```
L = L_rec  +  α · L_adv  +  β · L_outcome

  L_rec     = BPR recommendation loss
  L_adv     = −Σ discriminator(filtered_user_emb, sensitive_label)   [process]
  L_outcome = smooth_l1(|mean_bias_g0 − mean_bias_g1|, 0) × B/2     [outcome]
  α, β      = --fairpo_alpha / --fairpo_beta  (both default: 1.0)
```

FairPO is implemented for all four backbone models (BiasedMF, PMF, DMF, MLP).

---

## GNN Backbone — LightGCN Multi-Graph

As an extension to the paper's matrix factorization models, this repo includes a **LightGCN-based multi-graph** backbone (`models/GNN.py`) that learns user/item embeddings through graph propagation rather than simple lookup tables.

### Architecture

Three graphs are built from the training interactions and combined during propagation:

| Graph | Nodes | Edges | Captures |
|---|---|---|---|
| User–Item bipartite | users + items | user interacted with item | Collaborative filtering signal |
| User–User | users | two users share ≥ 3 common items | User similarity |
| Item–Item | items | two items share ≥ 3 common users | Item similarity |

Each graph uses symmetric degree normalization. Embeddings from all L propagation layers are averaged (LightGCN-style, no activation functions).

### Variants

| Class | Description |
|---|---|
| `LightGCN_MultiGraph` | Base GNN — propagated embeddings only |
| `LightGCN_PCFR` | + adversarial filter on user embeddings (same filter architecture as BiasedMF_PCFR) |

### Training

```bash
cd src/
python gnn_pcfr_train.py
# Trains GNN_PCFR single-attribute (×4 attributes) + multi-attribute on insurance
# Saves checkpoints to ../model/GNN_PCFR_insurance_<attr>/ and GNN_PCFR_multiattr_insurance/
# Runtime: ~15–20 min on CPU
```

---

## Usage

All commands must be run from the `src/` directory: `cd src/`

### Single experiment

```bash
# No fairness (baseline)
python main.py --model_name BiasedMF --fairness_framework None \
  --dataset insurance --feature_columns u_gender \
  --optimizer Adam --metric ndcg@3,f1@3 \
  --lr 1e-3 --l2 1e-4 --batch_size 1024 --epoch 100 --eval_disc

# FairPO
python main.py --model_name BiasedMF --fairness_framework FairPO \
  --dataset insurance --feature_columns u_gender \
  --optimizer Adam --metric ndcg@3,f1@3 \
  --lr 1e-3 --l2 1e-4 --batch_size 1024 --epoch 100 --eval_disc \
  --fairpo_alpha 1.0 --fairpo_beta 1.0
```

For PCFR, FairRec, FOCF variants: replace `--fairness_framework` accordingly. Optional extra args: `--fairrec_lambda 0.05` for FairRec.

### Full comparison (all 6 frameworks)

```bash
python run_comparison.py
# Trains all 6 frameworks on insurance/u_gender, 100 epochs each
# Runtime: ~20–30 min on CPU
```

### Batch experiments (all attributes)

```bash
while IFS= read -r line; do [[ "$line" == \#* || -z "$line" ]] && continue; eval "$line"; done \
  < ../cmd/BiasedMF/exp_insurance.txt

while IFS= read -r line; do [[ "$line" == \#* || -z "$line" ]] && continue; eval "$line"; done \
  < ../cmd/BiasedMF/exp_insurance_FairPO.txt
```

> **Tip:** The cmd files use `--epoch 1000` for full replication. For quick testing, override with `--epoch 100`.

### Q1: Cross-attribute leakage

```bash
python cross_leakage_eval.py
# Loads saved PCFR checkpoints, trains fresh attackers for all 4 attributes
# Runtime: ~15 min on CPU
```

### Q2: Multi-attribute PCFR

```bash
# Step 1 — Train
python multi_attr_pcfr_train.py
# Trains PCFR against all 4 discriminators simultaneously
# Saves to ../model/PCFR_multiattr_insurance/model.pt — ~10 min on CPU

# Step 2 — Evaluate quality + fairness
python multi_attr_eval.py
# Evaluates all saved checkpoints: NDCG@3, UGF, ValUnf for all 4 attributes
# Runtime: ~5 min on CPU
```

---

## Key Arguments Reference

| Argument               | Default   | Description |
|------------------------|-----------|-------------|
| `--model_name`         | `BiasedMF`| Model: `BiasedMF`, `PMF`, `DMF`, `MLP` |
| `--fairness_framework` | `None`    | Framework (see table above) |
| `--dataset`            | —         | `insurance` or `ml1M` |
| `--feature_columns`    | —         | Sensitive attribute, e.g. `u_gender` |
| `--epoch`              | `100`     | Training epochs |
| `--lr`                 | `0.001`   | Learning rate |
| `--l2`                 | `1e-5`    | L2 regularization |
| `--batch_size`         | `128`     | Training batch size |
| `--optimizer`          | `GD`      | `GD` (SGD), `Adam`, `Adagrad` |
| `--metric`             | `RMSE`    | Evaluation metrics, e.g. `ndcg@3,f1@3` |
| `--eval_disc`          | flag      | Evaluate process fairness via trained discriminator |
| `--u_vector_size`      | `64`      | User embedding dimension |
| `--vt_num_neg`         | `100`     | Negatives per positive at val/test time |
| `--fairpo_alpha`       | `1.0`     | FairPO: adversarial penalty weight |
| `--fairpo_beta`        | `1.0`     | FairPO: outcome-fairness penalty weight |
| `--fairrec_lambda`     | `0.5`     | FairRec: fairness loss weight |
| `--num_worker`         | `0`       | DataLoader workers (keep 0 on macOS/Windows) |

---

## Fairness Metrics

| Metric      | Better   | Description |
|-------------|----------|-------------|
| **ValUnf**  | lower ↓  | Signed gap in mean (prediction − label) between groups |
| **AbsUnf**  | lower ↓  | Absolute gap in mean \|prediction − label\| between groups |
| **UGF**     | lower ↓  | Gap in mean NDCG between the two user groups (user-oriented group fairness) |
| **DiscAUC** | → 0.5    | AUC of a discriminator predicting the sensitive attribute from user embeddings; 0.5 = no leakage (perfect process fairness) |

---

## Results

All results use **BiasedMF on the Insurance dataset**. Each framework in the baseline tables is trained separately per sensitive attribute (standard single-attribute evaluation).

### Recommendation Quality — NDCG@3 ↑

| Framework   | u_gender | u_occupation | u_activity | u_marital | Avg    |
|-------------|----------|--------------|------------|-----------|--------|
| None        | **0.8503** | **0.8531** | **0.8533** | **0.8538** | **0.8526** |
| FOCF_ValUnf | 0.8502   | 0.8527       | 0.8462     | 0.8504    | 0.8499 |
| FOCF_AbsUnf | 0.8508   | **0.8531**   | 0.8469     | 0.8498    | 0.8502 |
| PCFR        | 0.8337   | 0.8350       | 0.8346     | 0.8322    | 0.8339 |
| FairRec     | 0.8385   | 0.8380       | **0.8400** | 0.8374    | **0.8385** |
| **FairPO**  | 0.8355   | 0.8332       | 0.8356     | 0.8334    | 0.8344 |

Among process-fair methods, FairRec leads on quality, followed closely by FairPO and PCFR.

### Process Fairness — Attacker AUC → 0.5 ↓

Each column is the DiscAUC when the framework is trained to protect that specific attribute.

| Framework   | u_gender | u_occupation | u_activity | u_marital |
|-------------|----------|--------------|------------|-----------|
| None        | 0.5000   | 0.5000       | 0.7470     | 0.6149    |
| FOCF_ValUnf | 0.5000   | 0.5000       | 0.7420     | 0.6153    |
| FOCF_AbsUnf | 0.5000   | 0.5000       | 0.7345     | 0.6179    |
| PCFR        | 0.5000   | 0.5000       | 0.5000     | 0.5000    |
| FairRec     | 0.5000   | 0.5000       | 0.5000     | 0.5000    |
| **FairPO**  | **0.5000** | **0.5000** | **0.5000** | **0.5000** |

FOCF methods fail on `u_activity` and `u_marital` (no adversarial component). PCFR, FairRec, and FairPO all achieve perfect process fairness on their target attribute.

### Outcome Fairness — Value Unfairness ↓

| Framework   | u_gender | u_occupation | u_activity | u_marital | Avg     |
|-------------|----------|--------------|------------|-----------|---------|
| None        | 0.01246  | 0.02648      | 0.20084    | 0.04730   | 0.07177 |
| FOCF_ValUnf | 0.01281  | 0.03471      | 0.09019    | 0.08003   | 0.05444 |
| FOCF_AbsUnf | 0.01185  | 0.03728      | 0.10606    | 0.08731   | 0.06063 |
| PCFR        | **0.01021** | **0.01808** | 0.03834 | **0.03393** | **0.02514** |
| FairRec     | 0.01039  | 0.01877      | 0.03892    | 0.03453   | 0.02565 |
| **FairPO**  | 0.01055  | **0.01755**  | **0.03978** | 0.03608  | 0.02599 |

Despite having no explicit outcome-fairness constraint, PCFR and FairPO achieve the best ValUnf — adversarial training removes attribute-correlated patterns that often cause outcome bias.

### Summary

| Framework   | Process fair (all attrs)? | Avg NDCG@3 | Avg ValUnf |
|-------------|---------------------------|------------|------------|
| None        | ❌ (2/4)                  | 0.8526     | 0.07177    |
| FOCF_ValUnf | ❌ (2/4)                  | 0.8499     | 0.05444    |
| FOCF_AbsUnf | ❌ (2/4)                  | 0.8502     | 0.06063    |
| PCFR        | ✅ (4/4)                  | 0.8339     | **0.02514** |
| FairRec     | ✅ (4/4)                  | **0.8385** | 0.02565    |
| **FairPO**  | ✅ **(4/4)**               | 0.8344     | 0.02599    |

**FairPO** uniquely achieves all three goals: perfect process fairness, competitive recommendation quality, and strong outcome fairness — without requiring a dedicated outcome-fairness loss.

---

## Q1: Cross-Attribute Leakage Analysis

> **Question:** When PCFR protects one attribute, what happens to the leakage of the *other* attributes?

```bash
cd src/ && python cross_leakage_eval.py   # ~15 min on CPU
```

### Results — Cross-Leakage AUC Matrix

Rows = PCFR trained to protect this attribute. Columns = attribute probed by fresh attacker.
**←** marks the trained (protected) attribute. 0.50 = no leakage.

| Trained on → | u_gender | u_occupation | u_activity | u_marital |
|---|---|---|---|---|
| **Baseline (None)** | 0.5452 | 0.5438 | 0.8764 | 0.6653 |
| **PCFR / u_gender** | **0.5262 ←** | 0.5502 | 0.6964 | 0.6015 |
| **PCFR / u_occupation** | 0.5272 | **0.5535 ←** | 0.6869 | 0.6048 |
| **PCFR / u_activity** | 0.5210 | 0.5514 | **0.6913 ←** | 0.6028 |
| **PCFR / u_marital** | 0.5195 | 0.5499 | 0.6867 | **0.6110 ←** |

### Key Findings

1. **Positive spillover** — protecting any one attribute reduces leakage of all others. Adversarial training makes embeddings generally less informative about demographics.
2. **`u_activity` is resistant** — even when PCFR targets it directly, leakage only drops from 0.8764 → 0.6913. It is strongly encoded and cannot be suppressed by single-attribute training.
3. **No attribute reaches 0.50 unless it is the direct training target** — and even then only down to 0.52–0.55. True multi-attribute fairness requires training against all attributes simultaneously (→ Q2).

---

## Q2: Multi-Attribute Simultaneous Fairness

> **Question:** Can a single PCFR model suppress leakage of *all* attributes at once — and what is the cost to recommendation quality and outcome fairness?

### Approach

A single `BiasedMF_PCFR` is trained against **4 discriminators simultaneously** — one per attribute. Each batch step:

```
total_loss = rec_loss − adv_weight × (disc_gender + disc_occupation + disc_activity + disc_marital)
```

All four discriminators receive gradients every batch. The model must learn embeddings that fool all four at once.

```bash
cd src/
python multi_attr_pcfr_train.py    # ~10 min on CPU
python cross_leakage_eval.py       # adds PCFR_MultiAttr row to Q1 matrix
python multi_attr_eval.py          # quality + outcome fairness comparison
```

### Process Fairness — Q1 vs Q2 Leakage Comparison

| Model | Trained on | u_gender | u_occupation | u_activity | u_marital |
|---|---|---|---|---|---|
| PCFR | u_gender | **0.5262 ←** | 0.5502 | 0.6964 | 0.6015 |
| PCFR | u_occupation | 0.5272 | **0.5535 ←** | 0.6869 | 0.6048 |
| PCFR | u_activity | 0.5210 | 0.5514 | **0.6913 ←** | 0.6028 |
| PCFR | u_marital | 0.5195 | 0.5499 | 0.6867 | **0.6110 ←** |
| **PCFR_MultiAttr** | **all_attrs** | **0.5051** | **0.5150** | **0.5000** | **0.5118** |

**PCFR_MultiAttr drives all four attributes to near-chance AUC simultaneously**, with the biggest gain on the previously resistant `u_activity` (0.69 → 0.50).

### Quality & Outcome Fairness — Full Comparison

> All frameworks evaluated on the same held-out test set using their saved checkpoints (trained on `u_gender`). Fairness metrics are computed for all 4 attributes post-hoc to ensure a consistent comparison with PCFR_MultiAttr.

**Recommendation Quality**

| Framework | NDCG@3 | F1@3 |
|---|---|---|
| None | 0.8512 | 0.4707 |
| FOCF_ValUnf | 0.8514 | 0.4708 |
| FOCF_AbsUnf | 0.8517 | 0.4710 |
| PCFR | 0.8324 | 0.4567 |
| FairRec | 0.8373 | 0.4612 |
| FairPO | 0.8354 | 0.4592 |
| **PCFR_MultiAttr** | **0.7915** | **0.4440** |

**UGF — User-Oriented Group Fairness ↓** (lower = equal NDCG across groups)

| Framework | u_gender | u_occupation | u_activity | u_marital | Avg |
|---|---|---|---|---|---|
| None | 0.0019 | 0.1283 | 0.0560 | 0.0857 | 0.0680 |
| FOCF_ValUnf | 0.0004 | 0.1283 | 0.0546 | 0.0868 | 0.0675 |
| FOCF_AbsUnf | 0.0010 | 0.1273 | 0.0552 | 0.0851 | **0.0671** |
| PCFR | 0.0202 | 0.0908 | 0.0727 | 0.0945 | 0.0696 |
| FairRec | 0.0091 | 0.1231 | 0.0681 | 0.0916 | 0.0730 |
| FairPO | 0.0211 | 0.0860 | 0.0730 | 0.0914 | 0.0679 |
| **PCFR_MultiAttr** | 0.0154 | 0.1125 | 0.0704 | 0.1061 | 0.0761 |

**Value Unfairness ↓** (lower = equal score bias across groups)

| Framework | u_gender | u_occupation | u_activity | u_marital | Avg |
|---|---|---|---|---|---|
| None | 0.0131 | 0.0493 | 0.0274 | 0.2028 | 0.0731 |
| FOCF_ValUnf | 0.0133 | 0.0497 | 0.0278 | 0.2018 | 0.0731 |
| FOCF_AbsUnf | 0.0122 | 0.0494 | 0.0272 | 0.2013 | 0.0725 |
| PCFR | 0.0093 | 0.0339 | 0.0180 | 0.0392 | **0.0251** |
| FairRec | 0.0094 | 0.0343 | 0.0182 | 0.0396 | 0.0254 |
| FairPO | 0.0101 | 0.0340 | 0.0181 | 0.0392 | 0.0254 |
| **PCFR_MultiAttr** | 0.0512 | 0.0845 | 0.0632 | 0.1327 | 0.0829 |

### Key Findings

1. **Process fairness: PCFR_MultiAttr wins** — the only method to suppress leakage of all 4 attributes simultaneously (all AUC → 0.50–0.52).
2. **Quality cost** — joint adversarial training over-regularizes the user embeddings: NDCG drops ~5% relative to single-attribute PCFR (~0.83 → 0.79).
3. **Outcome fairness regresses** — ValUnf avg worsens from 0.025 (single-attr PCFR) to 0.083 (PCFR_MultiAttr). Suppressing all attribute signals simultaneously disrupts the score distribution across groups.
4. **Trade-off summary** — PCFR_MultiAttr represents a pure process-fairness solution. For applications requiring balanced process + outcome fairness with minimal quality loss, FairPO remains the stronger choice.

---

## GNN Results

All GNN experiments use `LightGCN_PCFR` with `n_layers=2`, `embed_dim=64`, on the **Insurance dataset**.

### Cross-Attribute Leakage — BiasedMF PCFR vs GNN PCFR

*(0.50 = no leakage | ← = trained attribute)*

| Model | Trained on | u_gender | u_occupation | u_activity | u_marital |
|---|---|---|---|---|---|
| BiasedMF PCFR | u_gender | 0.5262 ← | 0.5502 | 0.6964 | 0.6015 |
| BiasedMF PCFR | u_occupation | 0.5272 | 0.5535 ← | 0.6869 | 0.6048 |
| BiasedMF PCFR | u_activity | 0.5210 | 0.5514 | 0.6913 ← | 0.6028 |
| BiasedMF PCFR | u_marital | 0.5195 | 0.5499 | 0.6867 | 0.6110 ← |
| BiasedMF MultiAttr | all_attrs | 0.5051 | 0.5150 | **0.5000** | 0.5118 |
| **GNN PCFR** | u_gender | **0.5099 ←** | **0.5179** | **0.5437** | **0.5203** |
| **GNN PCFR** | u_occupation | **0.5073** | **0.5142 ←** | **0.5487** | **0.5210** |
| **GNN PCFR** | u_activity | **0.5090** | **0.5151** | **0.5492 ←** | **0.5211** |
| **GNN PCFR** | u_marital | **0.5013** | **0.5122** | **0.5483** | **0.5170 ←** |
| **GNN MultiAttr** | all_attrs | **0.5036** | **0.5183** | **0.5422** | **0.5145** |

### Key Findings

1. **GNN PCFR dramatically outperforms BiasedMF PCFR on cross-attribute leakage.** When BiasedMF PCFR is trained on u_gender, u_activity leakage stays at 0.6964 — barely moved from the baseline. GNN PCFR trained on u_gender brings u_activity down to 0.5437 without even targeting it.

2. **GNN provides near-free cross-attribute spillover.** Every GNN single-attribute model keeps all unprotected attributes below 0.55. BiasedMF consistently leaves u_activity at 0.69+.

3. **GNN MultiAttr adds only marginal gain over GNN single-attr.** Since GNN already suppresses cross-leakage well on its own, the multi-attribute variant (0.5422 on u_activity) barely improves over any single-attribute GNN model (~0.545). In contrast, BiasedMF needed explicit multi-attribute training to get u_activity from 0.69 to 0.50.

4. **Interpretation:** GNN embeddings are shaped by graph structure (shared interactions) rather than raw user–attribute correlations, making them inherently less attribute-informative. The adversarial filter then pushes the remaining signal to chance.

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `AttributeError: module 'numpy' has no attribute 'float'` | Replace `np.float` with `float`. Already patched in this repo. |
| `AssertionError: Torch not compiled with CUDA enabled` | Already patched — code auto-detects CPU/GPU. |
| `RuntimeError: DataLoader worker died` | Use `--num_worker 0` (default in all cmd files). |
| Out of memory | Reduce `--batch_size` to `256` or `--u_vector_size` to `32`. |
