"""21 -- (a) nested tuning sensitivity vs fixed; (b) isotonic calibration. argv: tune | cal"""
import os, sys, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, brier_score_loss
from sklearn.preprocessing import label_binarize
HERE=os.path.dirname(os.path.abspath(__file__)); D=os.path.join(HERE,"..","data"); O=os.path.join(HERE,"..","outputs")
m=pd.read_csv(os.path.join(D,"master_features.csv"),low_memory=False); fam=json.load(open(os.path.join(D,"feature_families.json")))
common=m.MMSE.notna()&m.fs_ICV_mm3.notna()
FS={"clinical":fam["demo"]+fam["cognition"],"multimodal":fam["demo"]+fam["cognition"]+fam["freesurfer"]}
fix=pd.read_csv(os.path.join(O,"table2_model_performance.csv"))
def fixed(task,fs,mdl,col):
    r=fix[(fix.task==task)&(fix.featureset==fs)&(fix.model==mdl)]; return float(r[col].iloc[0]) if len(r) else np.nan
def model_grid(name):
    if name=="logistic_l2":
        return Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler()),("c",LogisticRegression(max_iter=2000,class_weight="balanced"))]),{"c__C":[0.1,1.0,10.0]}
    return Pipeline([("i",SimpleImputer(strategy="median")),("c",HistGradientBoostingClassifier(random_state=0))]),{"c__learning_rate":[0.05,0.12],"c__max_iter":[150]}
def nested(d,cols,ycol,classes,mdl):
    y=d[ycol].astype(str).values; X=d[cols].values; est,grid=model_grid(mdl); yt=[];pp=[]
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=42).split(X,y):
        gs=GridSearchCV(est,grid,scoring="balanced_accuracy",cv=3,n_jobs=-1); gs.fit(X[tr],y[tr])
        pr=gs.predict_proba(X[te]); cl=list(gs.best_estimator_.classes_); pr=pr[:,[cl.index(c) for c in classes]]
        yt+=list(y[te]); pp+=list(pr)
    yt=np.array(yt); pp=np.array(pp)
    auc=(roc_auc_score((yt==classes[1]).astype(int),pp[:,1]) if len(classes)==2 else roc_auc_score(label_binarize(yt,classes=classes),pp,average="macro"))
    return round(balanced_accuracy_score(yt,np.array(classes)[pp.argmax(1)]),3),round(auc,3)

mode=sys.argv[1] if len(sys.argv)>1 else "both"
if mode in ("tune","both"):
    rows=[]
    jobs=[("conv36","clinical",["sMCI","pMCI"],"conv_36"),("dx3","clinical",["CN","MCI","AD"],"baseline_dx")]
    for task,fs,classes,ycol in jobs:
        d=m[common&(m.baseline_dx=="MCI")&m.conv_36.isin(["sMCI","pMCI"])] if task=="conv36" else m[common&m.baseline_dx.isin(["CN","MCI","AD"])]
        for mdl in ["logistic_l2","hist_gb"]:
            if task=="dx3" and mdl=="hist_gb": continue
            ba,auc=nested(d,FS[fs],ycol,classes,mdl)
            rows.append(dict(task=task,featureset=fs,model=mdl,bAcc_tuned=ba,AUC_tuned=auc,bAcc_fixed=round(fixed(task,fs,mdl,"bAcc"),3),AUC_fixed=round(fixed(task,fs,mdl,"AUC"),3)))
            print(f"{task} {fs} {mdl}: tuned {ba}/{auc}  fixed {fixed(task,fs,mdl,'bAcc'):.3f}/{fixed(task,fs,mdl,'AUC'):.3f}")
    pd.DataFrame(rows).to_csv(os.path.join(O,"tuning_sensitivity.csv"),index=False); print("saved tuning_sensitivity.csv")
if mode in ("cal","both"):
    d=m[common&(m.baseline_dx=="MCI")&m.conv_36.isin(["sMCI","pMCI"])]; y=(d.conv_36=="pMCI").astype(int).values; X=d[FS["clinical"]].values
    base=Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler()),("c",LogisticRegression(max_iter=2000,class_weight="balanced"))])
    p_un=cross_val_predict(base,X,y,cv=5,method="predict_proba")[:,1]
    p_cal=cross_val_predict(CalibratedClassifierCV(base,method="isotonic",cv=5),X,y,cv=5,method="predict_proba")[:,1]
    res={"brier_uncalibrated":round(brier_score_loss(y,p_un),3),"brier_isotonic":round(brier_score_loss(y,p_cal),3)}
    json.dump(res,open(os.path.join(O,"calibration_isotonic.json"),"w")); print("Isotonic calibration (conv36 clinical) Brier:",res)
