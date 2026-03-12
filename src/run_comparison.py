#!/usr/bin/env python3
"""
Comparison script: runs all 6 fairness frameworks (None, FOCF_ValUnf, FOCF_AbsUnf,
PCFR, FairRec, FairPO) on BiasedMF / insurance / u_gender and prints a side-by-side
results table.
"""
import subprocess
import re
import sys
import os

DATASET      = "insurance"
FEATURE      = "u_gender"
MODEL        = "BiasedMF"
EPOCHS       = "100"
METRIC       = "ndcg@3,f1@3"
VT_NUM_NEG   = "10"
BATCH        = "1024"
LR           = "1e-3"
L2           = "1e-4"

FRAMEWORKS = ["None", "FOCF_ValUnf", "FOCF_AbsUnf", "PCFR", "FairRec", "FairPO"]

BASE_CMD = [
    "python", "./main.py",
    "--data_processor",  "RecDataset",
    "--dataset",         DATASET,
    "--feature_columns", FEATURE,
    "--optimizer",       "Adam",
    "--metric",          METRIC,
    "--l2",              L2,
    "--lr",              LR,
    "--batch_size",      BATCH,
    "--runner",          "RecRunner",
    "--vt_num_neg",      VT_NUM_NEG,
    "--vt_batch_size",   BATCH,
    "--num_worker",      "0",
    "--epoch",           EPOCHS,
    "--eval_disc",
]

def run_framework(fw):
    model_dir = f"../model/cmp_{MODEL}_{fw}_{DATASET}_{FEATURE}"
    model_path = f"{model_dir}/model.pt"
    os.makedirs(model_dir, exist_ok=True)

    cmd = BASE_CMD + [
        "--model_name",         MODEL,
        "--fairness_framework",  fw,
        "--model_path",          model_path,
    ]
    if fw == "FairRec":
        cmd += ["--fairrec_lambda", "0.05"]
    if fw == "FairPO":
        cmd += ["--fairpo_alpha", "1.0", "--fairpo_beta", "1.0"]

    print(f"\n{'='*60}")
    print(f"Running  framework={fw}  ...")
    print(f"{'='*60}")
    sys.stdout.flush()

    result = subprocess.run(cmd, capture_output=True, text=True)
    out = result.stdout + result.stderr
    for line in out.splitlines()[-15:]:
        print(" ", line)
    return out


def extract(out, pattern, default="N/A"):
    m = re.search(pattern, out)
    return m.group(1) if m else default


results = {}
for fw in FRAMEWORKS:
    out = run_framework(fw)
    results[fw] = {
        "ndcg":     extract(out, r"Test After Training = ([\d.]+)"),
        "f1":       extract(out, r"Test After Training = [\d.]+,([\d.]+)"),
        "val_unf":  extract(out, r"Value unfairness: ([\d.]+)"),
        "abs_unf":  extract(out, r"Absolute unfairness: ([\d.]+)"),
        "usr_unf":  extract(out, r"User-oriented unfairness: ([\d.]+)"),
        "cgu":      extract(out, r"Calibrated group-wise utility: (-?[\d.]+)"),
        "disc_auc": extract(out, r"u_gender=([\d.]+)auc"),
    }

COL = 14
print("\n\n" + "=" * 100)
print(f"{'COMPARISON TABLE':^100}")
print(f"Model={MODEL} | Dataset={DATASET} | Attribute={FEATURE} | Epochs={EPOCHS}")
print("=" * 100)
header = f"{'Framework':<14} {'NDCG@3':>{COL}} {'F1@3':>{COL}} {'ValUnf↓':>{COL}} {'AbsUnf↓':>{COL}} {'UsrUnf↓':>{COL}} {'CGU↑':>{COL}} {'DiscAUC→0.5':>{COL}}"
print(header)
print("-" * 100)
for fw, r in results.items():
    row = (f"{fw:<14} {r['ndcg']:>{COL}} {r['f1']:>{COL}} {r['val_unf']:>{COL}} "
           f"{r['abs_unf']:>{COL}} {r['usr_unf']:>{COL}} {r['cgu']:>{COL}} {r['disc_auc']:>{COL}}")
    print(row)
print("=" * 100)
print("Metrics: NDCG@3 / F1@3 = recommendation quality (higher ↑ is better)")
print("         ValUnf / AbsUnf / UsrUnf = outcome unfairness (lower ↓ is better)")
print("         CGU = Calibrated Group-wise Utility (higher ↑ is better)")
print("         DiscAUC = process unfairness (closer to 0.5 is better)")
