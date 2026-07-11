"""Build process_list_followup.csv: EARLY follow-up scan of imaged baseline-MCI subjects
(earliest in [LO,HI] months) + leakage-safe landmark conversion label (H-month horizon).
Change over [0, t_fu] months; conversion over (t_fu, H] months -> no leakage. Needs the F: inventory + data/."""
import os, glob, pandas as pd, numpy as np
LO,HI,H=6,24,36   # early follow-up window (months) and horizon (months)
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
ADNI=os.environ.get("ADNI_ROOT",r"F:\ADNI")
A=pd.concat([pd.read_csv(c) for c in glob.glob(os.path.join(ADNI,"*.csv"))],ignore_index=True)
A["iid"]="I"+A["Image Data ID"].astype(str).str.replace(r"^I","",regex=True)
A["PTID"]=A["Subject"].astype(str).str.strip().str.replace(" ","_")
A["date"]=pd.to_datetime(A["Acq Date"],errors="coerce"); A=A.dropna(subset=["date"]).drop_duplicates("iid")
conv=pd.read_csv(os.path.join(ROOT,"data","cohort_conversion.csv"),low_memory=False)
mci=conv[conv.baseline_dx=="MCI"].copy(); mci["PTID"]=mci.PTID.astype(str); mci["bdate"]=pd.to_datetime(mci["baseline_date"],errors="coerce")
man=pd.read_csv(os.path.join(ROOT,"gpu_deep","fixed_deep_manifest.csv")); man["PTID"]=man.PTID.astype(str)
img_mci=set(man[man.baseline_dx=="MCI"].PTID); base_iid=dict(zip(man.PTID,"I"+man.ImageUID.astype(int).astype(str)))
X=A.merge(mci[["PTID","bdate","first_AD_month","max_followup_m"]],on="PTID",how="inner")
X=X[X.PTID.isin(img_mci)].copy(); X["m"]=(X["date"]-X["bdate"]).dt.days/30.4375
fu=X[(X.m>=LO)&(X.m<=HI)].sort_values("m").drop_duplicates("PTID")   # earliest early scan
def lab(r):
    c=r.first_AD_month; L=r.m
    if pd.notna(c) and c<=L+3: return None
    if pd.notna(c) and c<=H+3: return "pMCI"
    if (pd.isna(c) or c>H+3) and r.max_followup_m>=H-6: return "sMCI"
    return None
fu["label"]=fu.apply(lab,axis=1); fu=fu[fu.label.notna() & fu.PTID.isin(base_iid)]
out=pd.DataFrame({"image_id":fu.iid.values,"ptid":fu.PTID.values,"is_baseline":"True",
                  "followup_month":fu.m.round(1).values,"baseline_image_id":fu.PTID.map(base_iid).values,"label":fu.label.values})
out.to_csv(os.path.join(HERE,"process_list_followup.csv"),index=False)
print(f"process_list_followup.csv: {len(out)} follow-up scans | pMCI={int((out.label=='pMCI').sum())} sMCI={int((out.label=='sMCI').sum())} | window[{LO},{HI}]mo H={H}mo")
