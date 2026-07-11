"""independent end-to-end QC: recompute key numbers, cross-check
against the manuscript text, re-run leakage controls, verify deliverables."""
import os, re, json, glob, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize

HERE=os.path.dirname(os.path.abspath(__file__)); D=os.path.join(HERE,"..","data"); O=os.path.join(HERE,"..","outputs"); F=os.path.join(HERE,"..","figures"); M=os.path.join(HERE,"..","manuscript")
man=open(os.path.join(M,"manuscript.md"),encoding="utf-8").read()
checks=[]
def chk(name, ok, detail=""): checks.append((("PASS" if ok else "FAIL"), name, detail))

mf=pd.read_csv(os.path.join(D,"master_features.csv"),low_memory=False)
common=mf.MMSE.notna()&mf.fs_ICV_mm3.notna()

# --- cohort counts ---
dx=mf[common&mf.baseline_dx.isin(["CN","MCI","AD"])]; vc=dx.baseline_dx.value_counts()
chk("dx3 cohort n=3151", len(dx)==3151, f"got {len(dx)}")
chk("dx3 groups CN1283/MCI1379/AD489", (vc.get('CN')==1283 and vc.get('MCI')==1379 and vc.get('AD')==489), str(vc.to_dict()))
adcn=dx[dx.baseline_dx.isin(["CN","AD"])]; chk("adcn n=1772", len(adcn)==1772, f"got {len(adcn)}")
conv=mf[common&(mf.baseline_dx=='MCI')&mf.conv_36.isin(['pMCI','sMCI'])]; cv=conv.conv_36.value_counts()
chk("conv36 n=832 (pMCI287/sMCI545)", len(conv)==832 and cv.get('pMCI')==287 and cv.get('sMCI')==545, str(cv.to_dict()))

# --- manuscript cites these counts ---
for tok in ["3,151","1,283","1,379","489","1,772","287","545"]:
    chk(f"manuscript mentions {tok}", tok in man)

# --- headline metrics vs table2 + manuscript ---
t2=pd.read_csv(os.path.join(O,"table2_model_performance.csv"))
def g(task,fs,mdl,col): 
    r=t2[(t2.task==task)&(t2.featureset==fs)&(t2.model==mdl)]; return float(r[col].iloc[0]) if len(r) else np.nan
chk("dx3 clinical logistic bAcc≈0.931", abs(g("dx3","clinical","logistic_l2","bAcc")-0.931)<0.005, f'{g("dx3","clinical","logistic_l2","bAcc"):.3f}')
chk("dx3 clinical AUC≈0.987", abs(g("dx3","clinical","logistic_l2","AUC")-0.987)<0.005, f'{g("dx3","clinical","logistic_l2","AUC"):.3f}')
chk("dx3 freesurfer AUC≈0.722", abs(g("dx3","freesurfer","logistic_l2","AUC")-0.722)<0.01, f'{g("dx3","freesurfer","logistic_l2","AUC"):.3f}')
chk("conv36 clinical AUC≈0.874", abs(g("conv36","clinical","logistic_l2","AUC")-0.874)<0.01, f'{g("conv36","clinical","logistic_l2","AUC"):.3f}')
for tok in ["0.931","0.987","0.722","0.874"]:
    chk(f"manuscript mentions metric {tok}", tok in man)

# --- landmark sensitivity matches ---
ls=pd.read_csv(os.path.join(O,"landmark_sensitivity.csv"))
row=ls[(ls.landmark_mo==12)&(ls.horizon_mo==48)].iloc[0]
chk("landmark 12/48 n=551 pMCI196/sMCI355", row['n']==551 and row['pMCI']==196 and row['sMCI']==355, f"{row['n']}/{row['pMCI']}/{row['sMCI']}")
chk("landmark 12/48 baseline 0.870 -> +change 0.894", abs(row['AUC_baseline']-0.870)<0.01 and abs(row['AUC_plus_cogchange']-0.894)<0.01, f"{row['AUC_baseline']}->{row['AUC_plus_cogchange']}")

# --- XAI artifacts ---
conf=json.load(open(os.path.join(O,"conformal_conversion.json")))
chk("conformal coverage ~0.90+", conf["empirical_coverage"]>=0.88, str(conf))
orr=pd.read_csv(os.path.join(O,"odds_ratios_conversion.csv"))
adas_or=orr[orr.feature.str.contains("ADAS")].OR_per_SD.max()
chk("odds ratio ADAS large (>3)", adas_or>3, f"{adas_or:.2f}")

# --- LEAKAGE re-run ---
for nm,d in [("dx3",dx),("adcn",adcn),("conv36",conv)]:
    chk(f"{nm} one-row-per-subject", d.RID.is_unique, f"rows {len(d)}, uniq {d.RID.nunique()}")
fam=json.load(open(os.path.join(D,"feature_families.json")))
leakcols={"conv_24","conv_36","conv_48","conversion_month","first_AD_month","max_followup_m","baseline_dx"}
chk("predictor families have no outcome cols", not (set(fam['demo']+fam['cognition']+fam['freesurfer'])&leakcols))
# label permutation on dx3 clinical -> chance
cols=fam['demo']+fam['cognition']; X=dx[cols]; y=dx.baseline_dx.values
pipe=Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler()),("c",LogisticRegression(max_iter=3000,class_weight="balanced"))])
rng=np.random.default_rng(0); yp=rng.permutation(y); ba=[]
for tr,te in StratifiedKFold(5,shuffle=True,random_state=1).split(X,yp):
    pipe.fit(X.iloc[tr],yp[tr]); ba.append(balanced_accuracy_score(yp[te],pipe.predict(X.iloc[te])))
chk("label-permutation collapses to chance (~0.33)", abs(np.mean(ba)-0.333)<0.05, f"{np.mean(ba):.3f}")
# independent reproduction dx3 clinical GroupKFold
ba2=[]
for tr,te in GroupKFold(5).split(X,y,dx.RID.values):
    pipe.fit(X.iloc[tr],y[tr]); ba2.append(balanced_accuracy_score(y[te],pipe.predict(X.iloc[te])))
chk("independent GroupKFold reproduces ~0.931", abs(np.mean(ba2)-0.931)<0.02, f"{np.mean(ba2):.3f}")

# --- deliverables inventory ---
for n in range(1,11): chk(f"figure fig{n} exists", any(glob.glob(os.path.join(F,f"fig{n}_*.png"))))
for s in ["figS1","figS2","figS3a","figS3b","figS4","figS5"]: chk(f"{s} exists", bool(glob.glob(os.path.join(F,s+"*.png"))))
for d in ["manuscript_expanded.docx","supplementary.docx","reporting_checklists.docx","title_page_and_cover_letter.docx","journal_recommendation.docx"]:
    chk(f"deliverable {d}", os.path.exists(os.path.join(M,d)) and os.path.getsize(os.path.join(M,d))>3000)
# manuscript internal: all Figure 1..10 referenced; no double %
chk("no double-percent in manuscript", "%%" not in man)
chk("ORCID present", "0000-0002-0336-4825" in open(os.path.join(M,"title_page_and_cover_letter.md"),encoding="utf-8").read())
chk("figures 1..10 all referenced in text", all((f"Figure {i}" in man) for i in range(1,11)))

# report
n_pass=sum(1 for c in checks if c[0]=="PASS"); n_fail=len(checks)-n_pass
rep=["# QC Audit Report\n", f"**{n_pass}/{len(checks)} checks passed; {n_fail} failed.**\n"]
for st,name,det in checks:
    rep.append(f"- [{st}] {name}" + (f" — {det}" if (det and st=='FAIL') else (f" ({det})" if det else "")))
open(os.path.join(O,"qc_audit_report.md"),"w").write("\n".join(rep)+"\n")
print("\n".join(f"[{st}] {name}"+((" -- "+det) if det and st=='FAIL' else "") for st,name,det in checks))
print(f"\n==== {n_pass}/{len(checks)} PASSED, {n_fail} FAILED ====")
