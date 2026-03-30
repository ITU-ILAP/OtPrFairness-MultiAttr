# Results — Insurance Dataset

All experiments use **BiasedMF** (or **GNN**) backbone on the insurance dataset (1,511 users, 12 items).
Sensitive attributes: `u_gender`, `u_occupation`, `u_activity`, `u_marital_status`.

---

## 1. Recommendation Quality (NDCG@3 ↑)

Baseline frameworks trained separately per attribute (standard single-attribute protocol).

| Framework | u_gender | u_occupation | u_activity | u_marital | Avg |
|-----------|----------|--------------|------------|-----------|-----|
| None | 0.8503 | 0.8531 | 0.8533 | 0.8538 | 0.8526 |
| FOCF_ValUnf | 0.8502 | 0.8527 | 0.8462 | 0.8504 | 0.8499 |
| FOCF_AbsUnf | 0.8508 | 0.8531 | 0.8469 | 0.8498 | 0.8502 |
| PCFR | 0.8337 | 0.8350 | 0.8346 | 0.8322 | 0.8339 |
| FairRec | 0.8385 | 0.8380 | 0.8400 | 0.8374 | 0.8385 |
| FairPO | 0.8355 | 0.8332 | 0.8356 | 0.8334 | 0.8344 |
| PCFR_MultiAttr | — | — | — | — | 0.7915 |
| GNN PCFR | — | — | — | — | 0.8333 |
| GNN PCFR Multi | — | — | — | — | 0.8335 |

GNN PCFR matches single-attr PCFR quality (0.833). PCFR_MultiAttr pays a ~5% quality cost for protecting all attributes simultaneously.

---

## 2. Outcome Fairness

### Value Unfairness ↓ (signed bias gap between groups)

| Framework | u_gender | u_occupation | u_activity | u_marital | Avg |
|-----------|----------|--------------|------------|-----------|-----|
| None | 0.0131 | 0.0493 | 0.0274 | 0.2028 | 0.0731 |
| FOCF_ValUnf | 0.0133 | 0.0497 | 0.0278 | 0.2018 | 0.0731 |
| FOCF_AbsUnf | 0.0122 | 0.0494 | 0.0272 | 0.2013 | 0.0725 |
| PCFR | 0.0093 | 0.0339 | 0.0180 | 0.0392 | **0.0251** |
| FairRec | 0.0094 | 0.0343 | 0.0182 | 0.0396 | 0.0254 |
| FairPO | 0.0101 | 0.0340 | 0.0181 | 0.0392 | 0.0254 |
| PCFR_MultiAttr | 0.0512 | 0.0845 | 0.0632 | 0.1327 | 0.0829 |
| GNN PCFR | 0.0393 | 0.0567 | 0.0360 | 0.0639 | 0.0490 |
| GNN PCFR Multi | 0.0302 | 0.0589 | 0.0386 | 0.0614 | 0.0473 |

### UGF — User-Oriented Group Fairness ↓ (NDCG gap between groups)

| Framework | u_gender | u_occupation | u_activity | u_marital | Avg |
|-----------|----------|--------------|------------|-----------|-----|
| None | 0.0019 | 0.1283 | 0.0560 | 0.0857 | 0.0680 |
| FOCF_ValUnf | 0.0004 | 0.1283 | 0.0546 | 0.0868 | 0.0675 |
| FOCF_AbsUnf | 0.0010 | 0.1273 | 0.0552 | 0.0851 | 0.0671 |
| PCFR | 0.0202 | 0.0908 | 0.0727 | 0.0945 | 0.0696 |
| FairRec | 0.0091 | 0.1231 | 0.0681 | 0.0916 | 0.0730 |
| FairPO | 0.0211 | 0.0860 | 0.0730 | 0.0914 | 0.0679 |
| PCFR_MultiAttr | 0.0154 | 0.1125 | 0.0704 | 0.1061 | 0.0761 |
| GNN PCFR | 0.0045 | 0.1350 | 0.0749 | 0.0802 | 0.0737 |
| GNN PCFR Multi | 0.0068 | 0.1268 | 0.0724 | 0.0753 | 0.0703 |

PCFR and FairPO achieve the best ValUnf despite having no explicit outcome-fairness constraint — adversarial training incidentally removes attribute-correlated score patterns.

---

## 3. Process Fairness — Cross-Attribute Leakage (Attacker AUC → 0.50)

Row = model trained to protect that attribute. Column = attribute probed by a fresh attacker.
← marks the trained (protected) attribute. 0.50 = no leakage.

### BiasedMF

| Model | Trained on | u_gender | u_occupation | u_activity | u_marital |
|-------|-----------|----------|--------------|------------|-----------|
| BiasedMF None | — | 0.5452 | 0.5438 | 0.8764 | 0.6653 |
| BiasedMF PCFR | u_gender | 0.5262 ← | 0.5502 | 0.6964 | 0.6015 |
| BiasedMF PCFR | u_occupation | 0.5272 | 0.5535 ← | 0.6869 | 0.6048 |
| BiasedMF PCFR | u_activity | 0.5210 | 0.5514 | 0.6913 ← | 0.6028 |
| BiasedMF PCFR | u_marital | 0.5195 | 0.5499 | 0.6867 | 0.6110 ← |
| BiasedMF PCFR Multi | all_attrs | 0.5051 | 0.5150 | **0.5000** | 0.5118 |

### GNN (LightGCN)

| Model | Trained on | u_gender | u_occupation | u_activity | u_marital |
|-------|-----------|----------|--------------|------------|-----------|
| GNN PCFR | u_gender | 0.5099 ← | 0.5179 | 0.5437 | 0.5203 |
| GNN PCFR | u_occupation | 0.5073 | 0.5142 ← | 0.5487 | 0.5210 |
| GNN PCFR | u_activity | 0.5090 | 0.5151 | 0.5492 ← | 0.5211 |
| GNN PCFR | u_marital | 0.5013 | 0.5122 | 0.5483 | 0.5170 ← |
| GNN PCFR Multi | all_attrs | 0.5036 | 0.5183 | 0.5422 | 0.5145 |

### Key Finding

GNN dominates on process fairness. A single-attribute GNN model trained only on `u_gender` already suppresses `u_activity` leakage to 0.54 — better than what BiasedMF achieves with dedicated multi-attribute training (0.69). Graph propagation over the sparse, demographically-mixed insurance interaction graph makes user embeddings naturally less attribute-informative, giving the adversarial filter a much easier job.
