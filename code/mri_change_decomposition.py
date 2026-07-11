"""Isolate the incremental value of STRUCTURAL MRI CHANGE (atrophy rate) for
MCI->AD conversion, leakage-safe landmark [0,L] -> (L,H].
Feature families let us test the user's main hypothesis:
  does MRI *change* add value over (a) clinical baseline, (b) the cross-sectional
  MRI snapshot, (c) cognitive change?  Shared folds -> aligned OOF -> paired bootstrap."""
import os, numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,"..","data")
def L_(f): return pd.read_csv(os.path.join(DATA,f),low_memory=False)
SLACK=3.0
base=L_("manifest_subject_baseline.csv"); base["baseline_date"]=pd.to_datetime(base["baseline_date"])
bdate=dict(zip(base.RID,base.baseline_date))
man=L_("master_features.csv"); lf=L_("longitudinal_features.csv")
COG=[("mmse","MMSE"),("cdr","CDRSB"),("adas","ADAS13")]; cogtabs={t:L_(f"cog_{t}.csv") for t,_ in COG}
fs=L_("freesurfer_fsx7.csv"); icv=pd.to_numeric(fs["ST10CV"],errors="coerce")
fs["HIPP"]=(pd.to_numeric(fs["ST29SV"],errors="coerce")+pd.to_numeric(fs["ST88SV"],errors="coerce"))/icv*1000
fs["VENT"]=(pd.to_numeric(fs["ST37SV"],errors="coerce")+pd.to_numeric(fs["ST96SV"],errors="coerce"))/icv*1000
fs["ENT"]=(pd.to_numeric(fs["ST24TA"],errors="coerce")+pd.to_numeric(fs["ST83TA"],errors="coerce"))/2
def eslope(df,datecol,mm,prefix,L):
    df=df.copy(); df[datecol]=pd.to_datetime(df[datecol],errors="coerce"); df["bd"]=df.RID.map(bdate)
    df=df.dropna(subset=[datecol,"bd"]); df["m"]=(df[datecol]-df["bd"]).dt.days/30.4375
    df=df[(df.m>=-1)&(df.m<=L+SLACK)]; out=[]
    for rid,g in df.groupby("RID"):
        yrs=g.m.values/12.0; v=pd.to_numeric(g[mm],errors="coerce").values; ok=np.isfinite(v)&np.isfinite(yrs)
        out.append({"RID":rid, f"{prefix}{mm}_es": np.polyfit(yrs[ok],v[ok],1)[0] if (ok.sum()>=2 and np.ptp(yrs[ok])>0.1) else np.nan})
    return pd.DataFrame(out)
demo=["age","sex_female","PTEDUCAT","APOE4"]; cog_bl=["MMSE","CDRSB","ADAS13"]; fs_bl=["fs_HIPP_bl","fs_VENT_bl","fs_ENT_bl"]
cogcols=[f"cog_{c}_es" for c in cog_bl]; fscols=[f"fs_{m}_es" for m in ["HIPP","VENT","ENT"]]
def build(L,H):
    cog_es=None
    for t,c in COG:
        s=eslope(cogtabs[t],"VISDATE",c,"cog_",L); cog_es=s if cog_es is None else cog_es.merge(s,on="RID",how="outer")
    fes=None
    for m in ["HIPP","VENT","ENT"]:
        s=eslope(fs[["RID","EXAMDATE",m]],"EXAMDATE",m,"fs_",L); fes=s if fes is None else fes.merge(s,on="RID",how="outer")
    mci=base[base.baseline_dx=="MCI"].copy()
    def lab(r):
        c=r.first_AD_month
        if pd.notna(c) and c<=L+SLACK: return None
        if pd.notna(c) and c<=H+SLACK: return 1
        if (pd.isna(c) or c>H+SLACK) and r.max_followup_m>=H-6: return 0
        return None
    mci["y"]=mci.apply(lab,axis=1); coh=mci[mci.y.notna()][["RID","y"]]
    X=man.merge(coh,on="RID",how="inner").merge(lf[["RID"]+fs_bl],on="RID",how="left").merge(cog_es,on="RID",how="left").merge(fes,on="RID",how="left")
    X=X[X["cog_CDRSB_es"].notna()].copy()
    return X, X.y.astype(int).values
FAMILIES={
 "clinical_bl":              demo+cog_bl,
 "clinical+fs_bl(snapshot)": demo+cog_bl+fs_bl,
 "clinical+fs_change":       demo+cog_bl+fscols,
 "clinical+fs_bl+fs_change": demo+cog_bl+fs_bl+fscols,
 "fs_bl_only(level)":        fs_bl,
 "fs_change_only(rate)":     fscols,
 "clinical+cog_change":      demo+cog_bl+cogcols,
 "full(+cog+fs change)":     demo+cog_bl+fs_bl+cogcols+fscols,
}
def oof(X,y,cols,seed=7):
    pipe=Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler()),("c",LogisticRegression(max_iter=4000,class_weight="balanced"))])
    rkf=RepeatedStratifiedKFold(n_splits=5,n_repeats=2,random_state=seed)
    P=np.zeros(len(y)); C=np.zeros(len(y))
    for tr,te in rkf.split(X[cols].values,y):
        pipe.fit(X[cols].iloc[tr],y[tr]); P[te]+=pipe.predict_proba(X[cols].iloc[te])[:,1]; C[te]+=1
    return P/np.maximum(C,1)
def paired(y,pa,pb,B=1000):
    rng=np.random.default_rng(0); d=[]
    for _ in range(B):
        idx=rng.integers(0,len(y),len(y))
        if len(np.unique(y[idx]))<2: continue
        d.append(roc_auc_score(y[idx],pa[idx])-roc_auc_score(y[idx],pb[idx]))
    d=np.array(d); return np.percentile(d,2.5),np.percentile(d,97.5),float((d<=0).mean())

# Primary window 12/48 with full decomposition + paired tests
L,H=12,48
_cache=os.path.join(HERE,"..","outputs","mri_change_X_1248.csv")
if os.path.exists(_cache):
    _d=pd.read_csv(_cache); y=_d["y"].astype(int).values; X=_d
else:
    X,y=build(L,H); X["y"]=y; X.to_csv(_cache,index=False)
print(f"=== PRIMARY landmark L={L} H={H} | n={len(y)} pMCI={y.sum()} sMCI={(y==0).sum()} ===")
probs={}; res=[]
for name,cols in FAMILIES.items():
    p=oof(X,y,cols); probs[name]=p
    au=roc_auc_score(y,p)
    rng=np.random.default_rng(1); bs=[roc_auc_score(y[i],p[i]) for i in (rng.integers(0,len(y),len(y)) for _ in range(500)) if len(np.unique(y[i]))>1]
    lo,hi=np.percentile(bs,[2.5,97.5])
    res.append((name,au,lo,hi)); print(f"  {name:28} AUC={au:.3f} [{lo:.3f}-{hi:.3f}]",flush=True)
print("\n-- Paired tests (does structural CHANGE add?) --")
tests=[("clinical+fs_change","clinical_bl","MRG change vs clinical baseline"),
       ("clinical+fs_bl+fs_change","clinical+fs_bl(snapshot)","MRG change, beyond cross-sectional MRI"),
       ("fs_change_only(rate)","fs_bl_only(level)","atrofi HIZI vs atrofi DUZEYI (tek basina)"),
       ("full(+cog+fs change)","clinical+cog_change","MRG change, beyond cognitive change")]
prows=[]
for a,b,desc in tests:
    lo,hi,pp=paired(y,probs[a],probs[b]); d=roc_auc_score(y,probs[a])-roc_auc_score(y,probs[b])
    sig="EVET" if lo>0 else "hayir"
    print(f"  {desc:42} dAUC={d:+.3f} [{lo:+.3f},{hi:+.3f}] p={pp:.3f} -> adds value: {sig}")
    prows.append(dict(comparison=desc,dAUC=round(d,4),lo=round(lo,4),hi=round(hi,4),p=round(pp,4),adds=sig))
pd.DataFrame(res,columns=["featureset","AUC","lo","hi"]).to_csv(os.path.join(HERE,"..","outputs","mri_change_primary.csv"),index=False)
pd.DataFrame(prows).to_csv(os.path.join(HERE,"..","outputs","mri_change_paired.csv"),index=False)
print("saved outputs/mri_change_primary.csv + mri_change_paired.csv")
