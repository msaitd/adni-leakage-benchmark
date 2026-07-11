"""SHAP for CN/MCI/AD diagnosis + FreeSurfer region-importance figure."""
import os, re, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,"..","data"); FIG=os.path.join(HERE,"..","figures")
m=pd.read_csv(os.path.join(DATA,"master_features.csv"),low_memory=False); fam=json.load(open(os.path.join(DATA,"feature_families.json")))
NICE={"age":"Age","sex_female":"Female sex","PTEDUCAT":"Education","APOE4":"APOE e4 alleles","MMSE":"MMSE","CDRSB":"CDR-SB","CDGLOBAL":"CDR global","ADAS11":"ADAS-Cog11","ADAS13":"ADAS-Cog13","FAQ":"FAQ","fs_ICV_mm3":"ICV"}
def pretty(c):
    if c in NICE: return NICE[c]
    if c.startswith("fs_"):
        p=c.split("_",2); kind=p[1]; rest=p[2] if len(p)>2 else ""
        mm=re.search(r"of(.+)$",rest); reg=mm.group(1) if mm else rest
        reg=re.sub(r"(?<!^)(?=[A-Z])"," ",reg).strip()
        km={"vol":"vol","thk":"thick","area":"area","thksd":"thick-sd","hsv":"subfield"}
        return f"{reg} [{km.get(kind,kind)}]"
    return c
def region(c):  # bare anatomical region for FS columns
    if not c.startswith("fs_"): return None
    rest=c.split("_",2)[2] if len(c.split("_",2))>2 else ""
    mm=re.search(r"of(.+)$",rest); reg=mm.group(1) if mm else rest
    return re.sub(r"(?<!^)(?=[A-Z])"," ",reg).strip()

d=m[m.MMSE.notna()&m.fs_ICV_mm3.notna()&m.baseline_dx.isin(["CN","MCI","AD"])].copy()
cols=[c for c in fam["demo"]+fam["cognition"]+fam["freesurfer"] if c in d.columns]
Ximp=SimpleImputer(strategy="median").fit_transform(d[cols]); y=d.baseline_dx.values
names=[pretty(c) for c in cols]
model=ExtraTreesClassifier(n_estimators=120,n_jobs=2,class_weight="balanced_subsample",random_state=0).fit(Ximp,y)
expl=shap.TreeExplainer(model)
rng=np.random.default_rng(0); samp=rng.choice(len(Ximp),min(150,len(Ximp)),replace=False)
sv=expl.shap_values(Ximp[samp])              # (n,feat,3)
sv=np.array(sv)
imp=np.abs(sv).mean(axis=(0,2)) if sv.ndim==3 else np.abs(sv).mean(0)   # mean|SHAP| over samples+classes
order=np.argsort(imp)[::-1][:15][::-1]
plt.figure(figsize=(7,5.5)); plt.barh([names[i] for i in order], imp[order], color="#2c7fb8")
plt.xlabel("mean |SHAP| (CN/MCI/AD)"); plt.title("SHAP feature importance — CN/MCI/AD diagnosis")
plt.tight_layout(); plt.savefig(os.path.join(FIG,"figS4_shap_diagnosis_bar.png"),dpi=150,bbox_inches="tight"); plt.savefig(os.path.join(FIG,"figS4_shap_diagnosis_bar.svg"),bbox_inches="tight"); plt.close()

# FreeSurfer region-importance (aggregate |SHAP| over all FS features per anatomical region)
regimp={}
for i,c in enumerate(cols):
    r=region(c)
    if r: regimp[r]=regimp.get(r,0.0)+imp[i]
rs=pd.Series(regimp).sort_values(ascending=False).head(15)[::-1]
plt.figure(figsize=(7.5,5.8)); plt.barh(rs.index, rs.values, color="#f0a500")
plt.xlabel("aggregated mean |SHAP| across measures"); plt.title("FreeSurfer brain-region importance — CN/MCI/AD\n(sum of volume/thickness/area SHAP per region)")
plt.tight_layout(); plt.savefig(os.path.join(FIG,"figS5_roi_importance.png"),dpi=150,bbox_inches="tight"); plt.savefig(os.path.join(FIG,"figS5_roi_importance.svg"),bbox_inches="tight"); plt.close()
pd.Series(regimp).sort_values(ascending=False).round(5).to_csv(os.path.join(HERE,"..","outputs","roi_importance_dx3.csv"))
print("top SHAP (dx3):", [names[i] for i in order[::-1][:6]])
print("top FreeSurfer regions:", list(rs.index[::-1][:8]))
print("saved figS4_shap_diagnosis_bar, figS5_roi_importance, roi_importance_dx3.csv")
