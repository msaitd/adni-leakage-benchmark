"""baseline CSF/PET/plasma biomarkers from ADNIMERGE2 -> data/biomarkers_baseline.csv"""
import os, re, warnings, numpy as np, pandas as pd, pyreadr; warnings.filterwarnings("ignore")
PROJ=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); ROOT=os.path.dirname(PROJ)
RDA=os.path.join(ROOT,"adni images","ADNIMERGE2","data"); OUT=os.path.join(PROJ,"data")
def load(n): r=pyreadr.read_r(os.path.join(RDA,n+".rda")); return r[list(r.keys())[0]]
def vmonth(v):
    v=str(v).lower().strip()
    if v in ("bl","sc","scmri","init","blmri","m0","v01","v02"): return 0
    mm=re.match(r"m(\d+)",v); return int(mm.group(1)) if mm else 999
def baseline(df,cols):
    df=df.copy(); df["RID"]=pd.to_numeric(df["RID"],errors="coerce"); df=df.dropna(subset=["RID"]); df["RID"]=df["RID"].astype(int)
    df["_m"]=df["VISCODE2"].map(vmonth); df=df.sort_values("_m")
    for c in cols: df[c]=pd.to_numeric(df[c],errors="coerce")
    return df.groupby("RID")[cols].first().reset_index()
csf=baseline(load("UPENNBIOMK_ROCHE_ELECSYS"),["ABETA42","PTAU","TAU"]); csf["PTAU_ABETA42"]=csf.PTAU/csf.ABETA42
amy=baseline(load("UCBERKELEY_AMY_6MM"),["SUMMARY_SUVR","CENTILOIDS"])
tau=baseline(load("UCBERKELEY_TAU_6MM"),["META_TEMPORAL_SUVR","CTX_ENTORHINAL_SUVR"])
pl=baseline(load("UPENN_PLASMA_FUJIREBIO_QUANTERIX"),["pT217_AB42_F","AB42_AB40_F"])
bm=csf.merge(amy,on="RID",how="outer").merge(tau,on="RID",how="outer").merge(pl,on="RID",how="outer").rename(columns={
 "ABETA42":"CSF_ABETA42","PTAU":"CSF_PTAU","TAU":"CSF_TAU","PTAU_ABETA42":"CSF_PTAU_ABETA42","SUMMARY_SUVR":"AMYPET_SUVR",
 "CENTILOIDS":"AMYPET_CENTILOID","META_TEMPORAL_SUVR":"TAUPET_METATEMP","CTX_ENTORHINAL_SUVR":"TAUPET_ENTORHINAL",
 "pT217_AB42_F":"PLASMA_PTAU217_AB42","AB42_AB40_F":"PLASMA_AB42_AB40"})
bm.to_csv(os.path.join(OUT,"biomarkers_baseline.csv"),index=False); print("saved biomarkers_baseline.csv", bm.shape)
