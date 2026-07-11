"""add CSF/PET/plasma biomarker family; compare clinical vs
biomarker vs clinical+biomarker for diagnosis and conversion (subject-level CV)."""
import os, json, numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import label_binarize
HERE=os.path.dirname(os.path.abspath(__file__)); D=os.path.join(HERE,"..","data"); O=os.path.join(HERE,"..","outputs")
m=pd.read_csv(os.path.join(D,"master_features.csv"),low_memory=False); fam=json.load(open(os.path.join(D,"feature_families.json")))
bm=pd.read_csv(os.path.join(D,"biomarkers_baseline.csv"))
bcols=[c for c in bm.columns if c!="RID"]
m=m.merge(bm,on="RID",how="left")
clin=[c for c in fam["demo"]+fam["cognition"] if c in m.columns]
def cv(d,cols,ycol,classes):
    y=d[ycol].astype(str).values; X=d[cols].values; rid=d["RID"].values
    pipe=Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler()),("c",LogisticRegression(max_iter=3000,class_weight="balanced"))])
    rkf=RepeatedStratifiedKFold(n_splits=5,n_repeats=2,random_state=7); yt=[];pp=[];ba=[]
    for k,(tr,te) in enumerate(rkf.split(X,y)):
        assert not(set(rid[tr])&set(rid[te])); pipe.fit(X[tr],y[tr]); pr=pipe.predict_proba(X[te]); cl=list(pipe.classes_); pr=pr[:,[cl.index(c) for c in classes]]
        ba.append(balanced_accuracy_score(y[te],np.array(classes)[pr.argmax(1)]))
        if k<5: yt+=list(y[te]); pp+=list(pr)
    yt=np.array(yt); pp=np.array(pp)
    auc=(roc_auc_score((yt==classes[1]).astype(int),pp[:,1]) if len(classes)==2 else roc_auc_score(label_binarize(yt,classes=classes),pp,average="macro"))
    return round(np.mean(ba),3),round(auc,3)
has_bm=m["CSF_ABETA42"].notna()|m["AMYPET_SUVR"].notna()
rows=[]
for task,sub,ycol,classes in [("dx3 (CN/MCI/AD)",m[m.MMSE.notna()&has_bm&m.baseline_dx.isin(["CN","MCI","AD"])],"baseline_dx",["CN","MCI","AD"]),
                              ("conv36 (pMCI/sMCI)",m[m.MMSE.notna()&has_bm&(m.baseline_dx=="MCI")&m.conv_36.isin(["pMCI","sMCI"])],"conv_36",["sMCI","pMCI"])]:
    for name,cols in [("clinical",clin),("biomarker",bcols),("clinical+biomarker",clin+bcols)]:
        ba,auc=cv(sub,cols,ycol,classes); rows.append(dict(task=task,n=len(sub),featureset=name,bAcc=ba,AUC=auc))
        print(f"{task:20s} n={len(sub):4d} {name:20s} bAcc={ba} AUC={auc}")
pd.DataFrame(rows).to_csv(os.path.join(O,"biomarker_model_results.csv"),index=False)
print("saved biomarker_model_results.csv")
