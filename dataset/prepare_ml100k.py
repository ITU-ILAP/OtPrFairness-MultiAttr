# coding=utf-8
"""
Preprocess MovieLens 100k into the TSV format expected by this codebase.

Input:  /tmp/ml-100k/  (downloaded from grouplens.org)
Output: ./ml100k/      (same structure as ml1M/)

Sensitive attributes:
  u_gender   : F=0, M=1
  u_age      : <35 → 0 (young),  >=35 → 1 (older)
  u_occupation: student → 0,  all others → 1
  u_activity : interactions < median → 0,  >= median → 1

Train/Val/Test split: 80 / 10 / 10  (by interaction, stratified per user,
same approach used for ml1M)
"""

import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 2020
SRC          = '/tmp/ml-100k'
DST          = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml100k')
ATTRS        = ['u_gender', 'u_age', 'u_occupation', 'u_activity']

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.makedirs(DST, exist_ok=True)

# ── 1. Load ratings ───────────────────────────────────────────────────────────
print("Loading ratings ...")
ratings = pd.read_csv(
    os.path.join(SRC, 'u.data'), sep='\t',
    names=['uid', 'iid', 'rating', 'timestamp']
)
# Binarise: rating >= 4 → positive (label=1), else label=0
ratings['label'] = (ratings['rating'] >= 4).astype(int)
ratings = ratings.drop(columns=['rating', 'timestamp'])

# Re-index users and items to be 1-based and contiguous
uid_map = {u: i+1 for i, u in enumerate(sorted(ratings['uid'].unique()))}
iid_map = {v: i+1 for i, v in enumerate(sorted(ratings['iid'].unique()))}
ratings['uid'] = ratings['uid'].map(uid_map)
ratings['iid'] = ratings['iid'].map(iid_map)

print(f"  Users: {ratings['uid'].nunique()}, Items: {ratings['iid'].nunique()}, "
      f"Interactions: {len(ratings)}")

# ── 2. Load user demographics ─────────────────────────────────────────────────
print("Loading user info ...")
users = pd.read_csv(
    os.path.join(SRC, 'u.user'), sep='|',
    names=['uid_orig', 'age', 'gender', 'occupation', 'zip']
)
users['uid'] = users['uid_orig'].map(uid_map)
users = users.dropna(subset=['uid'])
users['uid'] = users['uid'].astype(int)

# u_gender: F=0, M=1
users['u_gender'] = (users['gender'] == 'M').astype(int)

# u_age: <35 → 0 (young), >=35 → 1 (older)
users['u_age'] = (users['age'] >= 35).astype(int)

# u_occupation: student → 0, all others → 1
users['u_occupation'] = (users['occupation'] != 'student').astype(int)

user_attr = users[['uid', 'u_gender', 'u_age', 'u_occupation']].set_index('uid')

# ── 3. u_activity: median split of interaction count ─────────────────────────
counts = ratings.groupby('uid').size()
median_count = counts.median()
activity_map = (counts >= median_count).astype(int).rename('u_activity')

# ── 4. Attach attributes to ratings ──────────────────────────────────────────
ratings = ratings.join(user_attr, on='uid').join(activity_map, on='uid')
ratings = ratings[['uid', 'iid', 'label'] + ATTRS]

print(f"  Attribute distributions:")
for a in ATTRS:
    v = ratings.drop_duplicates('uid')[a].value_counts().sort_index()
    print(f"    {a}: {dict(v)}")

# ── 5. Train / Validation / Test split (80/10/10 per user) ───────────────────
print("Splitting train/val/test ...")
train_rows, val_rows, test_rows = [], [], []

for uid, grp in ratings.groupby('uid'):
    grp = grp.sample(frac=1, random_state=RANDOM_SEED)  # shuffle
    n = len(grp)
    n_test = max(1, int(n * 0.10))
    n_val  = max(1, int(n * 0.10))
    test_rows.append(grp.iloc[:n_test])
    val_rows.append(grp.iloc[n_test:n_test + n_val])
    train_rows.append(grp.iloc[n_test + n_val:])

all_df   = ratings
train_df = pd.concat(train_rows).reset_index(drop=True)
val_df   = pd.concat(val_rows).reset_index(drop=True)
test_df  = pd.concat(test_rows).reset_index(drop=True)

print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# ── 6. Save main TSV files ────────────────────────────────────────────────────
print("Saving TSV files ...")
all_df.to_csv(os.path.join(DST, 'ml100k.all.tsv'),        sep='\t', index=False)
train_df.to_csv(os.path.join(DST, 'ml100k.train.tsv'),    sep='\t', index=False)
val_df.to_csv(os.path.join(DST, 'ml100k.validation.tsv'), sep='\t', index=False)
test_df.to_csv(os.path.join(DST, 'ml100k.test.tsv'),      sep='\t', index=False)

# ── 7. Attacker files (uid + attr, 80/20 split by user) ──────────────────────
print("Saving attacker files ...")
user_attrs_df = all_df.drop_duplicates('uid')[['uid'] + ATTRS].reset_index(drop=True)
user_attrs_df = user_attrs_df.sample(frac=1, random_state=RANDOM_SEED)
split = int(len(user_attrs_df) * 0.8)
att_train = user_attrs_df.iloc[:split]
att_test  = user_attrs_df.iloc[split:]

for attr in ATTRS:
    att_train[['uid', attr]].to_csv(
        os.path.join(DST, f'ml100k_{attr}.attacker.train.tsv'), sep='\t', index=False)
    att_test[['uid', attr]].to_csv(
        os.path.join(DST, f'ml100k_{attr}.attacker.test.tsv'),  sep='\t', index=False)

print(f"\nDone. Files written to {DST}/")
print("  pkl cache files will be auto-generated on first model run.")
