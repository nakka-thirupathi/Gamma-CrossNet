import argparse, pandas as pd
from sklearn.model_selection import GroupShuffleSplit

p=argparse.ArgumentParser()
p.add_argument("--csv",required=True); p.add_argument("--out",default="splits")
a=p.parse_args()
df=pd.read_csv(a.csv)
# Split by case_id to prevent leakage. This is a deterministic group split,
# not guaranteed class-stratified; inspect class distributions after creation.
gss=GroupShuffleSplit(n_splits=1,test_size=0.30,random_state=42)
tr_idx, tmp_idx=next(gss.split(df,groups=df.case_id))
train=df.iloc[tr_idx]; tmp=df.iloc[tmp_idx]
gss2=GroupShuffleSplit(n_splits=1,test_size=0.50,random_state=42)
va_rel,te_rel=next(gss2.split(tmp,groups=tmp.case_id))
import os
os.makedirs(a.out,exist_ok=True)
train.to_csv(f"{a.out}/train.csv",index=False)
tmp.iloc[va_rel].to_csv(f"{a.out}/val.csv",index=False)
tmp.iloc[te_rel].to_csv(f"{a.out}/test.csv",index=False)
print("Wrote splits. Verify patient-level disjointness and class counts before training.")
