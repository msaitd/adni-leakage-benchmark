"""Per-subject CHANGE (slope/delta) features from already-extracted longitudinal
tables (FreeSurfer all visits + cognition all visits). No new CAT12 needed.
Output: data/longitudinal_features.csv  (baseline + annual slope per measure)
"""
import os, numpy as np, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,"..","data")
def L(f): return pd.read_csv(os.path.join(DATA,f),low_memory=False)

# ---- FreeSurfer per-visit summary ROIs (ICV-normalized vols; entorhinal thickness) ----
fs=L("freesurfer_fsx7.csv"); fs["EXAMDATE"]=pd.to_datetime(fs["EXAMDATE"],errors="coerce")
fs=fs.dropna(subset=["EXAMDATE"])
icv=pd.to_numeric(fs["ST10CV"],errors="coerce")
fs["HIPP"]=(pd.to_numeric(fs["ST29SV"],errors="coerce")+pd.to_numeric(fs["ST88SV"],errors="coerce"))/icv*1000
fs["VENT"]=(pd.to_numeric(fs["ST37SV"],errors="coerce")+pd.to_numeric(fs["ST96SV"],errors="coerce"))/icv*1000
fs["ENT"] =(pd.to_numeric(fs["ST24TA"],errors="coerce")+pd.to_numeric(fs["ST83TA"],errors="coerce"))/2
fsm=fs[["RID","EXAMDATE","HIPP","VENT","ENT"]].copy()

def slopes(df,datecol,measures,prefix):
    rows=[]
    for rid,g in df.groupby("RID"):
        g=g.sort_values(datecol)
        yrs=(g[datecol]-g[datecol].min()).dt.days.values/365.25
        rec={"RID":rid}
        for m in measures:
            v=pd.to_numeric(g[m],errors="coerce").values
            ok=np.isfinite(v)&np.isfinite(yrs)
            rec[f"{prefix}{m}_bl"]=v[ok][0] if ok.sum()>=1 else np.nan
            if ok.sum()>=2 and np.ptp(yrs[ok])>0.3:
                rec[f"{prefix}{m}_slope"]=np.polyfit(yrs[ok],v[ok],1)[0]   # per year
            else:
                rec[f"{prefix}{m}_slope"]=np.nan
        rec[f"{prefix}n_visits"]=int(len(g))
        rec[f"{prefix}followup_y"]=float(yrs.max())
        rows.append(rec)
    return pd.DataFrame(rows)

fsl=slopes(fsm,"EXAMDATE",["HIPP","VENT","ENT"],"fs_")

# ---- cognition slopes ----
cogf=[]
for t,cols in [("mmse",["MMSE"]),("cdr",["CDRSB"]),("adas",["ADAS13"])]:
    d=L(f"cog_{t}.csv"); d["VISDATE"]=pd.to_datetime(d["VISDATE"],errors="coerce")
    d=d.dropna(subset=["VISDATE"])
    cogf.append(slopes(d,"VISDATE",cols,"cog_"))
cog=cogf[0]
for c in cogf[1:]: cog=cog.merge(c.drop(columns=[x for x in ["cog_n_visits","cog_followup_y"] if x in c]),on="RID",how="outer")

# ---- merge with baseline manifest (labels, demo, baseline FS/cog already there) ----
man=L("master_features.csv")
out=man.merge(fsl,on="RID",how="left").merge(cog,on="RID",how="left")
out.to_csv(os.path.join(DATA,"longitudinal_features.csv"),index=False)
sl=[c for c in out.columns if c.endswith("_slope")]
print("longitudinal_features.csv:",len(out),"subjects |",len(sl),"slope features")
print("subjects with FS slope (>=2 FS visits):", out["fs_HIPP_slope"].notna().sum())
print("subjects with MMSE slope:", out["cog_MMSE_slope"].notna().sum())
# quick descriptive: mean annual change by baseline dx
import numpy as np
g=out.groupby("baseline_dx")[["fs_HIPP_slope","fs_VENT_slope","fs_ENT_slope","cog_MMSE_slope","cog_CDRSB_slope","cog_ADAS13_slope"]].mean()
print("\nMean ANNUAL change by baseline diagnosis:"); print(g.round(3))
