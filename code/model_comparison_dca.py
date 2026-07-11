"""paired bootstrap AUC-difference tests + decision curve analysis."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
HERE=os.path.dirname(os.path.abspath(__file__)); O=os.path.join(HERE,"..","outputs"); F=os.path.join(HERE,"..","figures")
def oof(t): return pd.read_csv(os.path.join(O,f"oof_{t}.csv"))

def boot_diff(y, pa, pb, multiclass=False, classes=None, n=600, seed=0):
    rng=np.random.default_rng(seed); idx=np.arange(len(y))
    def auc(yy,pp):
        if multiclass: return roc_auc_score(label_binarize(yy,classes=classes),pp,average="macro")
        return roc_auc_score(yy,pp)
    base=auc(y,pa)-auc(y,pb); diffs=[]
    for _ in range(n):
        b=rng.choice(idx,len(idx),replace=True)
        try:
            if multiclass and (label_binarize(y[b],classes=classes).sum(0)==0).any(): continue
            if (not multiclass) and len(np.unique(y[b]))<2: continue
            diffs.append(auc(y[b],pa[b])-auc(y[b],pb[b]))
        except Exception: pass
    diffs=np.array(diffs); lo,hi=np.percentile(diffs,[2.5,97.5])
    p=2*min((diffs<=0).mean(),(diffs>=0).mean()); 
    return base,lo,hi,min(p,1.0)

rows=[]
# conv36 (pos=pMCI)
o=oof("conv36")
def get(task,fs,mdl,pcol):
    s=oof(task) if task!="conv36" else o
    s=s[(s.featureset==fs)&(s.model==mdl)].set_index("RID"); return s
def pair(task,A,B,pos):
    a=get(task,*A); b=get(task,*B); j=a.join(b[[f"p_{pos}"]],rsuffix="_b",how="inner")
    y=(j["y_true"].values==pos).astype(int)
    return y, j[f"p_{pos}"].values, j[f"p_{pos}_b"].values
for label,A,B in [("conv36: clinical vs FreeSurfer",("clinical","logistic_l2","p_pMCI"),("freesurfer","hist_gb","p_pMCI")),
                  ("conv36: multimodal vs clinical",("multimodal","hist_gb","p_pMCI"),("clinical","logistic_l2","p_pMCI")),
                  ("conv36: clinical vs cognition",("clinical","logistic_l2","p_pMCI"),("cognition","logistic_l2","p_pMCI"))]:
    y,pa,pb=pair("conv36",A,B,"pMCI"); base,lo,hi,p=boot_diff(y,pa,pb)
    rows.append(dict(comparison=label,AUC_diff=round(base,3),CI=f"{lo:.3f}–{hi:.3f}",p=round(p,3)))
# adcn (pos=AD)
oa=oof("adcn")
def getA(fs,mdl): s=oa[(oa.featureset==fs)&(oa.model==mdl)].set_index("RID"); return s
a=getA("clinical","logistic_l2"); b=getA("freesurfer","logistic_l2"); j=a.join(b[["p_AD"]],rsuffix="_b",how="inner")
y=(j.y_true.values=="AD").astype(int); base,lo,hi,p=boot_diff(y,j.p_AD.values,j.p_AD_b.values)
rows.append(dict(comparison="adcn: clinical vs FreeSurfer",AUC_diff=round(base,3),CI=f"{lo:.3f}–{hi:.3f}",p=round(p,3)))
# dx3 macro (multimodal vs clinical)
od=oof("dx3"); cls=["CN","MCI","AD"]; pc=["p_CN","p_MCI","p_AD"]
a=od[(od.featureset=="multimodal")&(od.model=="hist_gb")].set_index("RID")
b=od[(od.featureset=="clinical")&(od.model=="logistic_l2")].set_index("RID")
j=a.join(b[pc],rsuffix="_b",how="inner"); y=j.y_true.values
base,lo,hi,p=boot_diff(y,j[pc].values,j[[c+"_b" for c in pc]].values,multiclass=True,classes=cls)
rows.append(dict(comparison="dx3 macro: multimodal vs clinical",AUC_diff=round(base,3),CI=f"{lo:.3f}–{hi:.3f}",p=round(p,3)))
comp=pd.DataFrame(rows); comp.to_csv(os.path.join(O,"model_comparison_auc.csv"),index=False)
print(comp.to_string(index=False))

# Decision curve analysis (conv36 clinical)
s=o[(o.featureset=="clinical")&(o.model=="logistic_l2")]; y=(s.y_true.values=="pMCI").astype(int); p=s.p_pMCI.values; N=len(y); prev=y.mean()
ths=np.linspace(0.05,0.6,40); nb_model=[]; nb_all=[]
for t in ths:
    yhat=(p>=t).astype(int); tp=((yhat==1)&(y==1)).sum(); fp=((yhat==1)&(y==0)).sum()
    nb_model.append(tp/N-(fp/N)*(t/(1-t)))
    nb_all.append(prev-(1-prev)*(t/(1-t)))
plt.figure(figsize=(7,4.8))
plt.plot(ths,nb_model,color="#d7301f",lw=2,label="Clinical model")
plt.plot(ths,nb_all,color="#888",ls="--",label="Treat all")
plt.axhline(0,color="k",lw=1,label="Treat none")
plt.ylim(min(-0.02,min(nb_model)-0.01),max(nb_model)+0.02); plt.xlabel("Threshold probability"); plt.ylabel("Net benefit")
plt.title("Decision curve analysis — MCI conversion (clinical model)"); plt.legend()
plt.savefig(os.path.join(F,"figS6_decision_curve.png"),dpi=150,bbox_inches="tight"); plt.savefig(os.path.join(F,"figS6_decision_curve.svg"),bbox_inches="tight"); plt.close()
print("\nsaved model_comparison_auc.csv + figS6_decision_curve")
