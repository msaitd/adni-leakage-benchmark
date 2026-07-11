"""time-to-AD survival analysis among baseline MCI (Cox PH + c-index + KM)."""
import os, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import k_fold_cross_validation
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
HERE=os.path.dirname(os.path.abspath(__file__)); D=os.path.join(HERE,"..","data"); O=os.path.join(HERE,"..","outputs"); F=os.path.join(HERE,"..","figures")
base=pd.read_csv(os.path.join(D,"manifest_subject_baseline.csv"),low_memory=False)
mf=pd.read_csv(os.path.join(D,"master_features.csv"),low_memory=False)
lf=pd.read_csv(os.path.join(D,"longitudinal_features.csv"),low_memory=False)
mci=base[base.baseline_dx=="MCI"].copy()
mci["duration"]=np.where(mci.first_AD_month.notna(), mci.first_AD_month, mci.max_followup_m)
mci["event"]=mci.first_AD_month.notna().astype(int)
mci=mci[mci.duration>0]
pred=["age","sex_female","PTEDUCAT","APOE4","MMSE","CDRSB","ADAS13","FAQ"]
NICE={"age":"Age","sex_female":"Female sex","PTEDUCAT":"Education","APOE4":"APOE e4 alleles","MMSE":"MMSE","CDRSB":"CDR-SB","ADAS13":"ADAS-Cog13","FAQ":"FAQ","fs_HIPP_bl":"Hippocampus","fs_ENT_bl":"Entorhinal thick","fs_VENT_bl":"Lateral ventricle"}
mci["sex_female"]=(mci.SEX=="Female").astype(float)
X=mci.merge(lf[["RID","fs_HIPP_bl","fs_ENT_bl","fs_VENT_bl"]],on="RID",how="left")
cols=pred+["fs_HIPP_bl","fs_ENT_bl","fs_VENT_bl"]
Xi=pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X[cols]),columns=cols)
Xi=pd.DataFrame(StandardScaler().fit_transform(Xi),columns=cols)
Xi["duration"]=X["duration"].values; Xi["event"]=X["event"].values
print("MCI survival cohort n=%d, converters=%d, median follow-up=%.0f mo"%(len(Xi),Xi.event.sum(),np.median(Xi.duration)))
cph=CoxPHFitter(penalizer=0.1).fit(Xi,"duration","event")
s=cph.summary.copy(); s["feature"]=[NICE.get(i,i) for i in s.index]
hr=s[["feature","exp(coef)","exp(coef) lower 95%","exp(coef) upper 95%","p"]].rename(columns={"exp(coef)":"HR","exp(coef) lower 95%":"lo","exp(coef) upper 95%":"hi"})
hr=hr.sort_values("HR",ascending=False).round(3); hr.to_csv(os.path.join(O,"survival_cox_hr.csv"),index=False)
from sklearn.model_selection import KFold
from lifelines.utils import concordance_index
_ci=[]
for _tr,_te in KFold(5,shuffle=True,random_state=42).split(Xi):
    _f=CoxPHFitter(penalizer=0.1).fit(Xi.iloc[_tr],"duration","event")
    _ph=_f.predict_partial_hazard(Xi.iloc[_te])
    _ci.append(concordance_index(Xi.iloc[_te]["duration"], -_ph, Xi.iloc[_te]["event"]))
cidx=float(np.mean(_ci))
print("Cross-validated Harrell c-index = %.3f"%cidx)
print(hr.to_string(index=False))
json.dump({"cv_c_index":round(float(cidx),3),"n":int(len(Xi)),"converters":int(Xi.event.sum())},open(os.path.join(O,"survival_cindex.json"),"w"))

# KM by APOE e4 dose
plt.figure(figsize=(7,4.8)); kmf=KaplanMeierFitter()
Xa=X.copy(); Xa["e4"]=Xa["APOE4"]
for g,lab,col in [(0,"APOE e4 = 0","#2c7fb8"),(1,"APOE e4 = 1","#f0a500"),(2,"APOE e4 = 2","#d7301f")]:
    sub=Xa[Xa.e4==g]
    if len(sub)>5:
        kmf.fit(sub.duration, sub.event, label=f"{lab} (n={len(sub)})"); kmf.plot_survival_function(ci_show=False,color=col)
plt.xlabel("Months from baseline"); plt.ylabel("AD-free probability"); plt.title("Time to AD conversion by APOE e4 dose (baseline MCI)")
plt.xlim(0,72); plt.legend(); plt.savefig(os.path.join(F,"figS7_km_apoe.png"),dpi=150,bbox_inches="tight"); plt.savefig(os.path.join(F,"figS7_km_apoe.svg"),bbox_inches="tight"); plt.close()
print("saved survival_cox_hr.csv, survival_cindex.json, figS7_km_apoe")
