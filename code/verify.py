"""independent integrity / leakage audit. Writes qc_report.md."""
import os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJ, "data"); OUT = os.path.join(PROJ, "outputs")
m = pd.read_csv(os.path.join(DATA, "master_features.csv"), low_memory=False)
fam = json.load(open(os.path.join(DATA, "feature_families.json")))
mm = m[m["MMSE"].notna() & m["fs_ICV_mm3"].notna()].copy()
R = ["# Verification / Leakage Audit\n"]

# A. unique subject per cohort row
for name, d in [("dx3", mm[mm.baseline_dx.isin(["CN","MCI","AD"])]),
                ("adcn", mm[mm.baseline_dx.isin(["CN","AD"])]),
                ("conv36", mm[(mm.baseline_dx=="MCI") & mm.conv_36.isin(["sMCI","pMCI"])])]:
    uniq = d["RID"].is_unique
    R.append("- [%s] rows=%d, unique RID=%s  => one row/subject: %s"
             % (name, len(d), d["RID"].nunique(), "PASS" if uniq else "FAIL"))

# B. no leakage columns in predictor families
leak = {"conversion_month","first_AD_month","max_followup_m","conv_24","conv_36",
        "conv_48","baseline_dx","DX","fs_offset_days","baseline_date"}
allfeat = set(fam["demo"]+fam["cognition"]+fam["freesurfer"])
bad = allfeat & leak
R.append("- predictor families contain outcome/future columns: %s (%s)"
         % (sorted(bad) if bad else "none", "FAIL" if bad else "PASS"))

# C. independent re-derivation of conv_36 from trajectory
traj = pd.read_csv(os.path.join(DATA, "dx_trajectory.csv"))
base = pd.read_csv(os.path.join(DATA, "manifest_subject_baseline.csv"))
mci = base[base.baseline_dx=="MCI"]
firstAD = traj[traj.DX=="AD"].groupby("RID")["month"].min()
fu = traj.groupby("RID")["month"].max()
p=s=0
for rid in mci.RID:
    fa = firstAD.get(rid, np.nan); f = fu.get(rid, 0)
    if pd.notna(fa) and fa <= 39: p += 1
    elif f >= 30: s += 1
saved = pd.read_csv(os.path.join(DATA,"cohort_conversion.csv"))
sv = saved["conv_36"].value_counts()
R.append("- conv_36 re-derived pMCI=%d sMCI=%d vs saved pMCI=%d sMCI=%d : %s"
         % (p, s, sv.get("pMCI",0), sv.get("sMCI",0),
            "PASS" if (p==sv.get("pMCI",0) and s==sv.get("sMCI",0)) else "CHECK"))

# D. independent reproduction: dx3 clinical logistic, GroupKFold, different seed
d = mm[mm.baseline_dx.isin(["CN","MCI","AD"])]
cols = fam["demo"]+fam["cognition"]
X=d[cols]; y=d["baseline_dx"].values; g=d["RID"].values
pipe=Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler()),
               ("c",LogisticRegression(max_iter=3000,class_weight="balanced"))])
bas=[]
for tr,te in GroupKFold(5).split(X,y,g):
    assert not (set(g[tr])&set(g[te]))
    pipe.fit(X.iloc[tr],y[tr]); bas.append(balanced_accuracy_score(y[te],pipe.predict(X.iloc[te])))
R.append("- dx3 clinical logistic independent GroupKFold bAcc=%.3f (main run ~0.931): %s"
         % (np.mean(bas), "PASS" if abs(np.mean(bas)-0.931)<0.03 else "CHECK"))

# E. label-permutation leakage test (should collapse to chance ~0.33)
rng=np.random.default_rng(0); yp=rng.permutation(y); bas2=[]
for tr,te in StratifiedKFold(5,shuffle=True,random_state=1).split(X,yp):
    pipe.fit(X.iloc[tr],yp[tr]); bas2.append(balanced_accuracy_score(yp[te],pipe.predict(X.iloc[te])))
R.append("- label-permutation bAcc=%.3f (expected ~0.33 chance): %s"
         % (np.mean(bas2), "PASS" if abs(np.mean(bas2)-0.333)<0.05 else "CHECK"))

# F. FS baseline alignment
off = base["fs_offset_days"].dropna()
R.append("- FS baseline scan offset: median=%.0f d, 95%%<=%.0f d, max=%.0f d (window 365)"
         % (off.median(), off.quantile(.95), off.max()))

open(os.path.join(OUT,"qc_report.md"),"w").write("\n".join(R)+"\n")
print("\n".join(R))
