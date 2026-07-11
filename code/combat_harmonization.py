"""ComBat site harmonization of FreeSurfer features
(covariates age+sex, NOT diagnosis). Compare FreeSurfer-only AUC before/after on a
held-out split (subject-level). Sensitivity/robustness analysis."""
import os, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from neuroCombat import neuroCombat
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
HERE=os.path.dirname(os.path.abspath(__file__)); D=os.path.join(HERE,"..","data"); O=os.path.join(HERE,"..","outputs")
m=pd.read_csv(os.path.join(D,"master_features.csv"),low_memory=False); fam=json.load(open(os.path.join(D,"feature_families.json")))
m["site"]=m.PTID.astype(str).str.slice(0,3); fscols=[c for c in fam["freesurfer"] if c in m.columns]

def auc_holdout(d,cols,ycol,classes,seed=0):
    y=d[ycol].astype(str).values
    tr,te=train_test_split(np.arange(len(d)),test_size=0.3,stratify=y,random_state=seed)
    pipe=Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler()),("c",LogisticRegression(max_iter=2000,class_weight="balanced"))])
    pipe.fit(d[cols].values[tr],y[tr]); pr=pipe.predict_proba(d[cols].values[te]); cl=list(pipe.classes_); pr=pr[:,[cl.index(c) for c in classes]]
    yt=y[te]
    return (roc_auc_score((yt==classes[1]).astype(int),pr[:,1]) if len(classes)==2
            else roc_auc_score(label_binarize(yt,classes=classes),pr,average="macro"))

def harmonize(d):
    keep=d.site.value_counts(); keep=keep[keep>=10].index; d=d[d.site.isin(keep)].copy()
    dat=SimpleImputer(strategy="median").fit_transform(d[fscols]).T
    cov=d[["site","age","sex_female"]].copy(); cov["age"]=cov["age"].fillna(cov["age"].median()); cov["sex_female"]=cov["sex_female"].fillna(0)
    out=neuroCombat(dat=pd.DataFrame(dat),covars=cov,batch_col="site",categorical_cols=["sex_female"],continuous_cols=["age"])["data"].T
    dh=d.copy(); dh[fscols]=out; return d,dh,len(keep)

common=m.MMSE.notna()&m.fs_ICV_mm3.notna(); rows=[]
for task,sub,ycol,classes in [("dx3 (CN/MCI/AD)",m[common&m.baseline_dx.isin(["CN","MCI","AD"])],"baseline_dx",["CN","MCI","AD"]),
                              ("conv36 (pMCI/sMCI)",m[common&(m.baseline_dx=="MCI")&m.conv_36.isin(["pMCI","sMCI"])],"conv_36",["sMCI","pMCI"])]:
    draw,dh,ns=harmonize(sub)
    a0=np.mean([auc_holdout(draw,fscols,ycol,classes,s) for s in range(3)])
    a1=np.mean([auc_holdout(dh,fscols,ycol,classes,s) for s in range(3)])
    rows.append(dict(task=task,n=len(draw),n_sites=ns,FS_AUC_raw=round(a0,3),FS_AUC_ComBat=round(a1,3),delta=round(a1-a0,3)))
    print(f"{task}: n={len(draw)} sites={ns}  FS AUC raw={a0:.3f} -> ComBat={a1:.3f} (Δ{a1-a0:+.3f})")
pd.DataFrame(rows).to_csv(os.path.join(O,"combat_sensitivity.csv"),index=False)
print("saved combat_sensitivity.csv")
