"""Leakage-safe early-change conversion across landmark(L)/horizon(H) windows.
Predictors measured within [0,L]; outcome conversion in (L,H]. Saves table.
"""
import os, numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,"..","data")
def L_(f): return pd.read_csv(os.path.join(DATA,f),low_memory=False)
SLACK=3.0
base=L_("manifest_subject_baseline.csv"); base["baseline_date"]=pd.to_datetime(base["baseline_date"])
bdate=dict(zip(base.RID,base.baseline_date))
man=L_("master_features.csv"); lf=L_("longitudinal_features.csv")
COG=[("mmse","MMSE"),("cdr","CDRSB"),("adas","ADAS13")]
cogtabs={t:L_(f"cog_{t}.csv") for t,_ in COG}
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
def run(L,H):
    # early-change features for this L
    cog_es=None
    for t,c in COG:
        s=eslope(cogtabs[t],"VISDATE",c,"cog_",L); cog_es=s if cog_es is None else cog_es.merge(s,on="RID",how="outer")
    fes=None
    for m in ["HIPP","VENT","ENT"]:
        s=eslope(fs[["RID","EXAMDATE",m]],"EXAMDATE",m,"fs_",L); fes=s if fes is None else fes.merge(s,on="RID",how="outer")
    cogcols=[f"cog_{c}_es" for c in cog_bl]; fscols=[f"fs_{m}_es" for m in ["HIPP","VENT","ENT"]]
    mci=base[base.baseline_dx=="MCI"].copy()
    def lab(r):
        c=r.first_AD_month
        if pd.notna(c) and c<=L+SLACK: return None
        if pd.notna(c) and c<=H+SLACK: return 1
        if (pd.isna(c) or c>H+SLACK) and r.max_followup_m>=H-6: return 0
        return None
    mci["y"]=mci.apply(lab,axis=1); coh=mci[mci.y.notna()][["RID","y"]]
    X=man.merge(coh,on="RID",how="inner").merge(lf[["RID"]+fs_bl],on="RID",how="left").merge(cog_es,on="RID",how="left").merge(fes,on="RID",how="left")
    X=X[X["cog_CDRSB_es"].notna()].copy(); y=X.y.astype(int).values; rid=X.RID.values
    def auc(cols):
        pipe=Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler()),("c",LogisticRegression(max_iter=4000,class_weight="balanced"))])
        rkf=RepeatedStratifiedKFold(n_splits=5,n_repeats=2,random_state=7); yt=[];pp=[]
        for tr,te in rkf.split(X[cols].values,y):
            pipe.fit(X[cols].iloc[tr],y[tr]); yt+=list(y[te]); pp+=list(pipe.predict_proba(X[cols].iloc[te])[:,1])
        return roc_auc_score(np.array(yt),np.array(pp))
    return len(X),int(y.sum()),int((y==0).sum()), auc(demo+cog_bl+fs_bl), auc(demo+cog_bl+fs_bl+cogcols), auc(demo+cog_bl+fs_bl+cogcols+fscols)

rows=[]
for L,H in [(6,36),(12,36),(12,48),(24,48)]:
    n,p,s,a0,a1,a2=run(L,H)
    rows.append(dict(landmark_mo=L,horizon_mo=H,n=n,pMCI=p,sMCI=s,AUC_baseline=round(a0,3),AUC_plus_cogchange=round(a1,3),AUC_plus_cog_fs_change=round(a2,3)))
    print("L=%2d H=%2d | n=%3d (pMCI %3d/sMCI %3d) | baseline %.3f | +cogΔ %.3f | +cog+FSΔ %.3f"%(L,H,n,p,s,a0,a1,a2))
pd.DataFrame(rows).to_csv(os.path.join(HERE,"..","outputs","landmark_sensitivity.csv"),index=False)
print("saved outputs/landmark_sensitivity.csv")
