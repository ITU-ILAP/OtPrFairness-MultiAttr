# OtPrFairness-MultiAttr

Implementation for:

> **"Investigating User-Side Fairness in Outcome and Process for Multi-Type Sensitive Attributes in Recommendations"**
> ACM Transactions on Recommender Systems, 2025 — Hong Kong Baptist University

Extends the paper with **FairPO** (joint process + outcome fairness), a **GNN backbone** (LightGCN multi-graph), and experiments on two datasets (insurance, ml100k).

See [`results/`](results/) for all experimental results.

---

## Setup

```bash
git clone https://github.com/ITU-ILAP/OtPrFairness-MultiAttr.git
cd OtPrFairness-MultiAttr
pip install torch numpy pandas scikit-learn tqdm
```

All commands below must be run from `src/`: `cd src/`

---

## Datasets

| Dataset   | Users | Items | Interactions | Sensitive Attributes |
|-----------|-------|-------|--------------|----------------------|
| insurance | 1,511 | 12    | 27,180       | u_gender, u_occupation, u_activity, u_marital_status |
| ml100k    | 943   | 1,682 | 100,000      | u_gender, u_age, u_occupation, u_activity |

Dataset files are not included — see [`dataset/README.md`](dataset/README.md) for instructions.

---

## Fairness Frameworks

| Framework     | Type                  | Description |
|---------------|-----------------------|-------------|
| `None`        | Baseline              | No fairness constraint |
| `FOCF_ValUnf` | Outcome               | Minimizes signed bias gap between groups |
| `FOCF_AbsUnf` | Outcome               | Minimizes absolute unfairness between groups |
| `PCFR`        | Process               | Adversarial filter on user embeddings |
| `FairRec`     | Process               | Dual-branch learner with orthogonal regularization |
| `FairPO`      | Process + Outcome     | Adversarial filter + value-unfairness penalty (new) |

**FairPO loss:** `L = L_rec + α·L_adv + β·L_outcome` where `L_outcome = smooth_l1(|mean_bias_g0 − mean_bias_g1|, 0)`. Args: `--fairpo_alpha` (default 1.0), `--fairpo_beta` (default 1.0).

---

## GNN Backbone

`models/GNN.py` implements a LightGCN multi-graph backbone with three graphs built from training interactions:

- **User–Item** bipartite (collaborative filtering)
- **User–User** (users sharing ≥ 3 common items)
- **Item–Item** (items sharing ≥ 3 common users)

Two variants: `LightGCN_MultiGraph` (base) and `LightGCN_PCFR` (+ adversarial filter).

---

## Running Experiments

### Single experiment

```bash
python main.py --model_name BiasedMF --fairness_framework None \
  --dataset insurance --feature_columns u_gender \
  --optimizer Adam --metric ndcg@3,f1@3 \
  --lr 1e-3 --l2 1e-4 --batch_size 1024 --epoch 100 --eval_disc
```

Replace `--fairness_framework` with any framework name. Add `--fairrec_lambda 0.05` for FairRec, `--fairpo_alpha 1.0 --fairpo_beta 1.0` for FairPO.

### All 6 frameworks on insurance (comparison table)

```bash
python run_comparison.py          # ~20–30 min on CPU
```

### All frameworks × all attributes (batch)

```bash
# Insurance
while IFS= read -r line; do [[ "$line" == \#* || -z "$line" ]] && continue; eval "$line"; done \
  < ../cmd/BiasedMF/exp_insurance.txt

# Insurance — FairPO
while IFS= read -r line; do [[ "$line" == \#* || -z "$line" ]] && continue; eval "$line"; done \
  < ../cmd/BiasedMF/exp_insurance_FairPO.txt
```

> The cmd files use `--epoch 1000`. Add `--epoch 100` for quick testing.

### Q1 — Cross-attribute leakage (insurance)

```bash
python cross_leakage_eval.py      # ~15 min on CPU
```

Loads saved PCFR and GNN checkpoints, trains a fresh attacker for each attribute, outputs a leakage AUC matrix.

### Q2 — Multi-attribute PCFR (insurance)

```bash
python multi_attr_pcfr_train.py   # train   ~10 min
python multi_attr_eval.py         # evaluate ~5 min
```

### GNN — Insurance

```bash
python gnn_pcfr_train.py          # trains GNN_PCFR ×4 attrs + multi-attr  ~15–20 min
python cross_leakage_eval.py      # includes GNN rows automatically
python multi_attr_eval.py         # quality + fairness comparison
```

### ml100k

```bash
python ml100k_baseline_train.py   # all 6 baseline frameworks         ~30 min
python gnn_pcfr_train.py --dataset ml100k  # GNN PCFR ×5 models      ~8 hrs
python ml100k_leakage_eval.py     # cross-attribute leakage eval      ~1 hr
python ml100k_comparison_eval.py  # quality + outcome fairness        ~5 min
```

---

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--model_name` | `BiasedMF` | `BiasedMF`, `PMF`, `DMF`, `MLP` |
| `--fairness_framework` | `None` | See frameworks table above |
| `--dataset` | — | `insurance` or `ml100k` |
| `--feature_columns` | — | Sensitive attribute(s), e.g. `u_gender` |
| `--epoch` | `100` | Training epochs |
| `--lr` | `0.001` | Learning rate |
| `--l2` | `1e-5` | L2 regularization |
| `--batch_size` | `128` | Training batch size |
| `--optimizer` | `GD` | `GD`, `Adam`, `Adagrad` |
| `--metric` | `RMSE` | e.g. `ndcg@3,f1@3` |
| `--eval_disc` | flag | Evaluate process fairness via discriminator |
| `--u_vector_size` | `64` | Embedding dimension |
| `--vt_num_neg` | `100` | Negatives per positive at test time |
| `--num_worker` | `0` | Keep at 0 on macOS/Windows |

---

## Fairness Metrics

| Metric | Direction | Description |
|---|---|---|
| NDCG@3 | ↑ | Ranking quality |
| ValUnf | ↓ | Signed gap in mean (prediction − label) between groups |
| UGF | ↓ | Gap in mean NDCG between user groups |
| DiscAUC | → 0.5 | Attacker AUC for predicting sensitive attribute from embeddings |

---

## Troubleshooting

| Error | Fix |
|---|---|
| `numpy has no attribute 'float'` | Already patched. Update NumPy if it appears elsewhere. |
| `Torch not compiled with CUDA` | Already patched — auto-detects CPU/GPU. |
| `DataLoader worker died` | Use `--num_worker 0` (default). |
| Out of memory | Reduce `--batch_size` to `256` or `--u_vector_size` to `32`. |
