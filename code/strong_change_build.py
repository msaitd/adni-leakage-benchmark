"""Build + cache a BROAD AD-signature structural panel: early change (slope within
[0,L]) AND baseline value, for the leakage-safe MCI->AD landmark. Strong test."""
import os,sys,numpy as np,pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,"..","data")
def L_(f): return pd.read_csv(os.path.join(DATA,f),low_memory=False)
L=int(sys.argv[1]); H=int(sys.argv[2]); SLACK=3.0
base=L_("manifest_subject_baseline.csv"); base["baseline_date"]=pd.to_datetime(base["baseline_date"])
bdate=dict(zip(base.RID,base.baseline_date)); man=L_("master_features.csv")
COG=[("mmse","MMSE"),("cdr","CDRSB"),("adas","ADAS13")]; cogtabs={t:L_(f"cog_{t}.csv") for t,_ in COG}
fs=L_("freesurfer_fsx7.csv"); n=lambda c: pd.to_numeric(fs[c],errors="coerce"); icv=n("ST10CV")
fs["HIPP"]=(n("ST29SV")+n("ST88SV"))/icv*1000; fs["VENT"]=(n("ST37SV")+n("ST96SV"))/icv*1000
fs["ENT"]=(n("ST24TA")+n("ST83TA"))/2; fs["AMYG"]=(n("ST12SV")+n("ST71SV"))/icv*1000
fs["FUSI"]=(n("ST26TA")+n("ST85TA"))/2; fs["PRECUN"]=(n("ST52TA")+n("ST111TA"))/2
fs["PARAHIP"]=(n("ST44TA")+n("ST103TA"))/2
MEAS=["HIPP","VENT","ENT","AMYG","FUSI","PRECUN","PARAHIP"]
def early(df,datecol,mm,pref):
    df=df.copy(); df[datecol]=pd.to_datetime(df[datecol],errors="coerce"); df["bd"]=df.RID.map(bdate)
    df=df.dropna(subset=[datecol,"bd"]); df["m"]=(df[datecol]-df["bd"]).dt.days/30.4375
    df=df[(df.m>=-1)&(df.m<=L+SLACK)]; out=[]
    for rid,g in df.groupby("RID"):
        yrs=g.m.values/12.0; v=pd.to_numeric(g[mm],errors="coerce").values; ok=np.isfinite(v)&np.isfinite(yrs)
        base_v=v[ok][np.argmin(yrs[ok])] if ok.sum()>=1 else np.nan
        slope=np.polyfit(yrs[ok],v[ok],1)[0] if (ok.sum()>=2 and np.ptp(yrs[ok])>0.1) else np.nan
        out.append({"RID":rid,f"{pref}{mm}_es":slope,f"{pref}{mm}_bl":base_v})
    return pd.DataFrame(out)
F=None
for m in MEAS:
    s=early(fs[["RID","EXAMDATE",m]],"EXAMDATE",m,"fs_"); F=s if F is None else F.merge(s,on="RID",how="outer")
C=None
for t,c in COG:
    s=early(cogtabs[t],"VISDATE",c,"cog_"); C=s if C is None else C.merge(s,on="RID",how="outer")
mci=base[base.baseline_dx=="MCI"].copy()
def lab(r):
    c=r.first_AD_month
    if pd.notna(c) and c<=L+SLACK: return None
    if pd.notna(c) and c<=H+SLACK: return 1
    if (pd.isna(c) or c>H+SLACK) and r.max_followup_m>=H-6: return 0
    return None
mci["y"]=mci.apply(lab,axis=1); coh=mci[mci.y.notna()][["RID","y"]]
demo=["age","sex_female","PTEDUCAT","APOE4"]
X=man[["RID"]+demo+["MMSE","CDRSB","ADAS13"]].merge(coh,on="RID",how="inner").merge(F,on="RID",how="left").merge(C,on="RID",how="left")
X=X[X["cog_CDRSB_es"].notna()].copy()
X.to_csv(os.path.join(HERE,"..","outputs",f"strong_change_X_{L}_{H}.csv"),index=False)
print(f"cache L={L} H={H}: n={len(X)} pMCI={int(X.y.sum())} sMCI={int((X.y==0).sum())} | kolon={X.shape[1]}")
