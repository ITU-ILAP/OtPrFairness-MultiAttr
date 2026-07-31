"""6-way method comparison: Friedman + Nemenyi + paired Wilcoxon.

Rebuilt from the handoff doc description (original /tmp/six_way.py lost).
Cells: quality -> (backbone, seed); leakage/fairness -> (backbone, seed, attr).
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats

try:
    import scikit_posthocs as sp
    HAVE_SP = True
except ImportError:
    HAVE_SP = False

METHODS = {  # canonical name -> framework label in CSV
    "None": "None", "PCFR": "PCFR", "CRAFTV1": "CRAFT",
    "FairRec": "FairRec", "CRAFTV3": "CRAFTV3_lam30", "CRAFTV4": "CRAFTV4_lam30",
}

q = pd.read_csv("results/eval_quality.csv")
f = pd.read_csv("results/eval_fairness.csv")
for df in (q, f):
    df["framework"] = df["framework"].fillna("None")

def cell_matrix(dataset, metric):
    """Return DataFrame: rows = cells, columns = methods (only methods present)."""
    cols = {}
    if metric == "ndcg":
        d = q[q.dataset == dataset]
        for name, fw in METHODS.items():
            g = d[d.framework == fw].groupby(["backbone", "seed"])["ndcg_at_3"].mean()
            if len(g): cols[name] = g
    else:
        d = f[f.dataset == dataset]
        attrs = sorted(d.eval_attr.unique())
        for name, fw in METHODS.items():
            dd = d[d.framework == fw]
            if not len(dd): continue
            vals = {}
            for (bb, sd), gg in dd.groupby(["backbone", "seed"]):
                for a in attrs:
                    if name == "None":
                        on = gg[gg.eval_attr == a]
                        offv = gg[gg.eval_attr != a]["leakage_auc"].mean()
                    else:
                        sub = gg[gg.protected_attr == a]
                        if not len(sub): continue
                        on = sub[sub.eval_attr == a]
                        offv = sub[sub.eval_attr != a]["leakage_auc"].mean()
                    if not len(on): continue
                    r = on.iloc[0]
                    v = {"on_leak": r.leakage_auc, "off_leak": offv,
                         "ugf": r.ugf, "val_unfair": r.value_unfairness}[metric]
                    vals[(bb, sd, a)] = v
            if vals: cols[name] = pd.Series(vals)
    M = pd.DataFrame(cols).dropna()
    return M

def analyze(dataset):
    print(f"\n{'='*70}\n  {dataset}\n{'='*70}")
    for metric in ["ndcg", "on_leak", "off_leak", "ugf", "val_unfair"]:
        M = cell_matrix(dataset, metric)
        if M.shape[1] < 3 or len(M) < 5:
            print(f"\n-- {metric}: insufficient data ({M.shape})"); continue
        higher_better = metric == "ndcg"
        ranks = M.rank(axis=1, ascending=not higher_better).mean()
        fr_stat, fr_p = stats.friedmanchisquare(*[M[c] for c in M.columns])
        print(f"\n-- {metric}  (n={len(M)} cells, {M.shape[1]} methods)  "
              f"Friedman chi2={fr_stat:.1f} p={fr_p:.2e}")
        order = ranks.sort_values()
        print("   mean rank (1=best): " +
              "  ".join(f"{m}={r:.2f}" for m, r in order.items()))
        if HAVE_SP and fr_p < 0.05:
            nem = sp.posthoc_nemenyi_friedman(M.values)
            nem.index = nem.columns = M.columns
            best = order.index[0]
            sig = [f"{m}(p={nem.loc[best, m]:.3f})" for m in M.columns
                   if m != best and nem.loc[best, m] < 0.05]
            ns = [m for m in M.columns if m != best and nem.loc[best, m] >= 0.05]
            print(f"   Nemenyi vs best({best}): sig-worse: {', '.join(sig) or 'none'}"
                  + (f" | not-sig: {', '.join(ns)}" if ns else ""))
        # key paired Wilcoxon tests
        for a, b in [("CRAFTV3", "PCFR"), ("CRAFTV3", "FairRec"),
                     ("CRAFTV4", "CRAFTV3"), ("CRAFTV3", "None")]:
            if a in M.columns and b in M.columns:
                d = M[a] - M[b]
                if np.allclose(d, 0): continue
                w = stats.wilcoxon(M[a], M[b])
                print(f"   Wilcoxon {a} vs {b}: mean diff={d.mean():+.4f} p={w.pvalue:.2e}")

for ds in ["ml100k", "insurance", "ml1m"]:
    analyze(ds)
print()
