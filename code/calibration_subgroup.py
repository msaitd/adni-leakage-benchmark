"""calibration curves (binary) + sex/age subgroup fairness from OOF."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, balanced_accuracy_score
HERE=os.path.dirname(os.path.abspath(__file__)); O=os.path.join(HERE,"..","outputs"); FIG=os.path.join(HERE,"..","figures"); D=os.path.join(HERE,"..","data")
m=pd.read_csv(os.path.join(D,"master_features.csv"),low_memory=False)[["RID","sex_female","age"]]

# ---------- calibration (binary) ----------
fig,axes=plt.subplots(1,2,figsize=(10,4.6))
panels=[("adcn","AD","p_AD","clinical","logistic_l2","A. AD vs CN (clinical)"),
        ("conv36","pMCI","p_pMCI","clinical","logistic_l2","B. pMCI vs sMCI (clinical)")]
for ax,(task,pos,pcol,fs,mdl,title) in zip(axes,panels):
    o=pd.read_csv(os.path.join(O,f"oof_{task}.csv")); s=o[(o.featureset==fs)&(o.model==mdl)]
    y=(s.y_true.values==pos).astype(int); p=s[pcol].values
    frac,mean=calibration_curve(y,p,n_bins=10,strategy="quantile"); brier=brier_score_loss(y,p)
    ax.plot([0,1],[0,1],"k--",lw=1,label="ideal"); ax.plot(mean,frac,"o-",color="#d7301f",label=f"model (Brier={brier:.3f})")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency"); ax.set_title(title); ax.legend(loc="upper left",fontsize=9)
fig.suptitle("Calibration (cross-validated out-of-fold)")
fig.savefig(os.path.join(FIG,"fig10_calibration.png"),dpi=150,bbox_inches="tight")
fig.savefig(os.path.join(FIG,"fig10_calibration.svg"),bbox_inches="tight"); plt.close(fig)

# ---------- subgroup fairness ----------
rows=[]
# dx3 multimodal hist_gb: balanced accuracy by sex / age tertile
o=pd.read_csv(os.path.join(O,"oof_dx3.csv")); s=o[(o.featureset=="multimodal")&(o.model=="hist_gb")].merge(m,on="RID",how="left")
s["agegrp"]=pd.qcut(s["age"],3,labels=["young","mid","old"])
for grp,gv in [("sex","sex_female"),("age","agegrp")]:
    for val,sub in s.groupby(gv):
        lab=("Female" if val==1 else "Male") if grp=="sex" else str(val)
        rows.append(dict(task="dx3(CN/MCI/AD)",subgroup=f"{grp}={lab}",n=len(sub),
                         balanced_acc=round(balanced_accuracy_score(sub.y_true,sub.y_pred),3),auc=""))
# conv36 clinical logistic: AUC by sex / age
o=pd.read_csv(os.path.join(O,"oof_conv36.csv")); s=o[(o.featureset=="clinical")&(o.model=="logistic_l2")].merge(m,on="RID",how="left")
s["agegrp"]=pd.qcut(s["age"],3,labels=["young","mid","old"]); s["yb"]=(s.y_true=="pMCI").astype(int)
for grp,gv in [("sex","sex_female"),("age","agegrp")]:
    for val,sub in s.groupby(gv):
        lab=("Female" if val==1 else "Male") if grp=="sex" else str(val)
        try: auc=round(roc_auc_score(sub.yb,sub.p_pMCI),3)
        except Exception: auc=np.nan
        rows.append(dict(task="conv36(pMCI/sMCI)",subgroup=f"{grp}={lab}",n=len(sub),
                         balanced_acc=round(balanced_accuracy_score(sub.yb,(sub.p_pMCI>=0.5).astype(int)),3),auc=auc))
sf=pd.DataFrame(rows); sf.to_csv(os.path.join(O,"subgroup_fairness.csv"),index=False)
print(sf.to_string(index=False))
print("\nsaved fig10_calibration + subgroup_fairness.csv")
