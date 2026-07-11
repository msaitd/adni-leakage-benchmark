import os,sys,numpy as np,pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline; from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
HERE=os.path.dirname(os.path.abspath(__file__))
L,H=int(sys.argv[1]),int(sys.argv[2])
X=pd.read_csv(os.path.join(HERE,"..","outputs",f"strong_change_X_{L}_{H}.csv"))
y=X["y"].astype(int).values
demo=["age","sex_female","PTEDUCAT","APOE4"]; cog_bl=["MMSE","CDRSB","ADAS13"]
M=["HIPP","VENT","ENT","AMYG","FUSI","PRECUN","PARAHIP"]
fs_bl=[f"fs_{m}_bl" for m in M]; fs_es=[f"fs_{m}_es" for m in M]; cog_es=[f"cog_{c}_es" for c in cog_bl]
FAM={"clinical_bl":demo+cog_bl,"clinical+fs_bl(snapshot)":demo+cog_bl+fs_bl,
 "clinical+fs_change":demo+cog_bl+fs_es,"clinical+fs_bl+fs_change":demo+cog_bl+fs_bl+fs_es,
 "fs_bl_only(level)":fs_bl,"fs_change_only(rate)":fs_es,"clinical+cog_change":demo+cog_bl+cog_es,
 "full(+cog+fs change)":demo+cog_bl+fs_bl+cog_es+fs_es}
def mk(kind):
    if kind=="lr": return Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler()),("c",LogisticRegression(max_iter=4000,class_weight="balanced"))])
    if kind=="hgb": return Pipeline([("i",SimpleImputer(strategy="median")),("c",HistGradientBoostingClassifier(learning_rate=0.08,max_iter=120,l2_regularization=1.0))])
    return Pipeline([("i",SimpleImputer(strategy="median")),("c",ExtraTreesClassifier(n_estimators=200,class_weight="balanced_subsample",n_jobs=2,random_state=0))])
rkf=RepeatedStratifiedKFold(n_splits=5,n_repeats=2,random_state=7); splits=list(rkf.split(X[demo].values,y))
def oof(cols,kind):
    P=np.zeros(len(y)); C=np.zeros(len(y)); pipe=mk(kind)
    for tr,te in splits:
        pipe.fit(X[cols].iloc[tr],y[tr]); P[te]+=pipe.predict_proba(X[cols].iloc[te])[:,1]; C[te]+=1
    return P/np.maximum(C,1)
# best model per family + store best OOF
best={}; print(f"=== STRONG L={L} H={H} n={len(y)} (pMCI {int(y.sum())}/sMCI {int((y==0).sum())}); genis panel x 3 model ===")
for name,cols in FAM.items():
    cand={k:oof(cols,k) for k in ["lr","hgb","et"]}
    aucs={k:roc_auc_score(y,p) for k,p in cand.items()}
    bk=max(aucs,key=aucs.get); best[name]=(cand[bk],aucs[bk],bk)
    print(f"  {name:28} bestAUC={aucs[bk]:.3f} ({bk})  [lr {aucs['lr']:.3f} hgb {aucs['hgb']:.3f} et {aucs['et']:.3f}]")
def paired(pa,pb,B=800):
    rng=np.random.default_rng(0); d=[]
    for _ in range(B):
        i=rng.integers(0,len(y),len(y))
        if len(np.unique(y[i]))<2: continue
        d.append(roc_auc_score(y[i],pa[i])-roc_auc_score(y[i],pb[i]))
    d=np.array(d); return float(np.mean(d)),np.percentile(d,2.5),np.percentile(d,97.5),float((d<=0).mean())
import numpy as _np
_rows=[(k,v[1],v[2]) for k,v in best.items()]
pd.DataFrame(_rows,columns=["featureset","AUC","best_model"]).to_csv(os.path.join(HERE,"..","outputs",f"strong_change_auc_{L}_{H}.csv"),index=False)
_np.savez(os.path.join(HERE,"..","outputs",f"strong_change_prob_{L}_{H}.npz"),y=y,**{k:v[0] for k,v in best.items()})
print("saved AUC table + probs",flush=True)
print("\n-- Does structural CHANGE add value? (vs the BEST model in each comparison) --")
T=[("clinical+fs_change","clinical_bl","change vs clinical baseline"),
   ("clinical+fs_bl+fs_change","clinical+fs_bl(snapshot)","change, beyond cross-sectional MRI"),
   ("fs_change_only(rate)","fs_bl_only(level)","atrofi HIZI vs DUZEY (tek basina)"),
   ("full(+cog+fs change)","clinical+cog_change","change, beyond cognitive change")]
for a,b,desc in T:
    dm,lo,hi,pp=paired(best[a][0],best[b][0]); d=best[a][1]-best[b][1]
    print(f"  {desc:40} dAUC={d:+.3f} (boot {dm:+.3f}[{lo:+.3f},{hi:+.3f}]) p={pp:.3f} -> {'ADDS' if lo>0 else 'none'}")
