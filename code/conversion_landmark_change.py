"""Leakage-safe MCI->AD conversion via EARLY change (landmark). At landmark L, among
MCI not yet converted, use baseline + change over [0,L] to predict conversion in
(L,H]. All predictors <= L; outcome after L -> no leakage.
"""
import os, numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve, confusion_matrix

HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,"..","data")
def L_(f): return pd.read_csv(os.path.join(DATA,f),low_memory=False)
LM=12.0; H=48.0; SLACK=3.0

man=L_("master_features.csv")
base=L_("manifest_subject_baseline.csv"); base["baseline_date"]=pd.to_datetime(base["baseline_date"])
bdate=dict(zip(base.RID,base.baseline_date))
lf=L_("longitudinal_features.csv")

def early_slopes(df,datecol,meas,prefix):
    df=df.copy(); df[datecol]=pd.to_datetime(df[datecol],errors="coerce")
    df["bd"]=df.RID.map(bdate); df=df.dropna(subset=[datecol,"bd"])
    df["m"]=(df[datecol]-df["bd"]).dt.days/30.4375
    df=df[(df.m>=-1)&(df.m<=LM+SLACK)]
    out=[]
    for rid,g in df.groupby("RID"):
        yrs=g.m.values/12.0; rec={"RID":rid}
        for mm in meas:
            v=pd.to_numeric(g[mm],errors="coerce").values; ok=np.isfinite(v)&np.isfinite(yrs)
            rec[f"{prefix}{mm}_eslope"]=np.polyfit(yrs[ok],v[ok],1)[0] if (ok.sum()>=2 and np.ptp(yrs[ok])>0.1) else np.nan
        out.append(rec)
    return pd.DataFrame(out)

cogsl=None
for t,c in [("mmse","MMSE"),("cdr","CDRSB"),("adas","ADAS13")]:
    s=early_slopes(L_(f"cog_{t}.csv"),"VISDATE",[c],"cog_")[["RID",f"cog_{c}_eslope"]]
    cogsl=s if cogsl is None else cogsl.merge(s,on="RID",how="outer")
fs=L_("freesurfer_fsx7.csv"); icv=pd.to_numeric(fs["ST10CV"],errors="coerce")
fs["HIPP"]=(pd.to_numeric(fs["ST29SV"],errors="coerce")+pd.to_numeric(fs["ST88SV"],errors="coerce"))/icv*1000
fs["VENT"]=(pd.to_numeric(fs["ST37SV"],errors="coerce")+pd.to_numeric(fs["ST96SV"],errors="coerce"))/icv*1000
fs["ENT"]=(pd.to_numeric(fs["ST24TA"],errors="coerce")+pd.to_numeric(fs["ST83TA"],errors="coerce"))/2
fs_es=["fs_HIPP_eslope","fs_VENT_eslope","fs_ENT_eslope"]
fssl=early_slopes(fs[["RID","EXAMDATE","HIPP","VENT","ENT"]],"EXAMDATE",["HIPP","VENT","ENT"],"fs_")[["RID"]+fs_es]

mci=base[base.baseline_dx=="MCI"].copy()
def label(r):
    c=r.first_AD_month
    if pd.notna(c) and c<=LM+SLACK: return "early"
    if pd.notna(c) and c<=H+SLACK: return "pMCI"
    if (pd.isna(c) or c>H+SLACK) and r.max_followup_m>=H-6: return "sMCI"
    return np.nan
mci["lab"]=mci.apply(label,axis=1)
coh=mci[mci.lab.isin(["pMCI","sMCI"])][["RID","lab"]]

demo=["age","sex_female","PTEDUCAT","APOE4"]; cog_bl=["MMSE","CDRSB","ADAS13"]; fs_bl=["fs_HIPP_bl","fs_VENT_bl","fs_ENT_bl"]
cog_es=["cog_MMSE_eslope","cog_CDRSB_eslope","cog_ADAS13_eslope"]
X=man.merge(coh,on="RID",how="inner").merge(lf[["RID"]+fs_bl],on="RID",how="left")
X=X.merge(cogsl,on="RID",how="left").merge(fssl,on="RID",how="left")
X=X[X["cog_CDRSB_eslope"].notna()].copy()
y=(X.lab.values=="pMCI").astype(int); rid=X.RID.values
print("Landmark cohort (L=%g mo, H=%g mo): n=%d  pMCI=%d  sMCI=%d"%(LM,H,len(X),y.sum(),(y==0).sum()))

def cvauc(cols):
    pipe=Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler()),
                   ("c",LogisticRegression(max_iter=4000,class_weight="balanced"))])
    rkf=RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=7); ba=[];yt=[];pp=[]
    for k,(tr,te) in enumerate(rkf.split(X[cols].values,y)):
        assert not(set(rid[tr])&set(rid[te]))
        pipe.fit(X[cols].iloc[tr],y[tr]); p=pipe.predict_proba(X[cols].iloc[te])[:,1]
        ba.append(balanced_accuracy_score(y[te],(p>=0.5).astype(int)))
        if k<5: yt+=list(y[te]); pp+=list(p)
    yt=np.array(yt);pp=np.array(pp); auc=roc_auc_score(yt,pp)
    fpr,tpr,th=roc_curve(yt,pp); t=th[np.argmax(tpr-fpr)]
    tn,fp,fn,tp=confusion_matrix(yt,(pp>=t).astype(int)).ravel()
    return np.mean(ba),auc,tp/(tp+fn),tn/(tn+fp)

print("\npMCI vs sMCI  (baseline-only  vs  + EARLY CHANGE [0-%g mo]):"%LM)
rows=[]
for name,cols in [("Baseline (klinik+FS)",demo+cog_bl+fs_bl),
                  ("Baseline + early cognitive change",demo+cog_bl+fs_bl+cog_es),
                  ("Baseline + early cognitive+FS change",demo+cog_bl+fs_bl+cog_es+fs_es)]:
    ba,auc,sens,spec=cvauc(cols); rows.append(dict(model=name,n=len(X),bAcc=ba,AUC=auc,sens=sens,spec=spec))
    print("  %-40s AUC=%.3f bAcc=%.3f sens=%.2f spec=%.2f"%(name,auc,ba,sens,spec))
pd.DataFrame(rows).to_csv(os.path.join(HERE,"..","outputs","landmark_conversion_results.csv"),index=False)
print("saved outputs/landmark_conversion_results.csv")
