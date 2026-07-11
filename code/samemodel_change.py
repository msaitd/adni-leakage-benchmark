import os,sys,numpy as np,pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline; from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
HERE=os.path.dirname(os.path.abspath(__file__)); KIND=sys.argv[1] if len(sys.argv)>1 else "lr"
demo=["age","sex_female","PTEDUCAT","APOE4"]; cog=["MMSE","CDRSB","ADAS13"]
M=["HIPP","VENT","ENT","AMYG","FUSI","PRECUN","PARAHIP"]
fbl=[f"fs_{m}_bl" for m in M]; fes=[f"fs_{m}_es" for m in M]; ces=[f"cog_{c}_es" for c in cog]
FAM={"A_clinical":demo+cog,"B_clin+fsbl":demo+cog+fbl,"C_clin+fschg":demo+cog+fes,
 "D_clin+fsbl+fschg":demo+cog+fbl+fes,"E_clin+fsbl+cogchg":demo+cog+fbl+ces,
 "F_full":demo+cog+fbl+ces+fes}
def mk():
    if KIND=="lr": return Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler()),("c",LogisticRegression(max_iter=4000,class_weight="balanced"))])
    return Pipeline([("i",SimpleImputer(strategy="median")),("c",HistGradientBoostingClassifier(learning_rate=0.08,max_iter=120,l2_regularization=1.0))])
def paired(y,pa,pb,B=1200):
    rng=np.random.default_rng(0); d=[roc_auc_score(y[i],pa[i])-roc_auc_score(y[i],pb[i]) for i in (rng.integers(0,len(y),len(y)) for _ in range(B)) if len(np.unique(y[i]))>1]
    d=np.array(d); return np.percentile(d,2.5),np.percentile(d,97.5),float((d<=0).mean())
for L,H in [(12,48),(24,48)]:
    X=pd.read_csv(os.path.join(HERE,"..","outputs",f"strong_change_X_{L}_{H}.csv")); y=X["y"].astype(int).values
    sp=list(RepeatedStratifiedKFold(n_splits=5,n_repeats=2,random_state=7).split(X[demo].values,y))
    P={}
    for nm,cols in FAM.items():
        pr=np.zeros(len(y)); c=np.zeros(len(y)); pipe=mk()
        for tr,te in sp: pipe.fit(X[cols].iloc[tr],y[tr]); pr[te]+=pipe.predict_proba(X[cols].iloc[te])[:,1]; c[te]+=1
        P[nm]=pr/np.maximum(c,1)
    print(f"\n=== {KIND} L={L} H={H} n={len(y)} pMCI={int(y.sum())} ===")
    for nm in FAM: print(f"  {nm:22} AUC={roc_auc_score(y,P[nm]):.3f}")
    for a,b,desc in [("C_clin+fschg","A_clinical","fs_change vs klinik"),("D_clin+fsbl+fschg","B_clin+fsbl","fs_change / kesitsel OTESI"),("F_full","E_clin+fsbl+cogchg","fs_change / kesitsel+bilis OTESI")]:
        lo,hi,pp=paired(y,P[a],P[b]); d=roc_auc_score(y,P[a])-roc_auc_score(y,P[b])
        print(f"    {desc:34} dAUC={d:+.3f} [{lo:+.3f},{hi:+.3f}] p={pp:.3f} {'ADDS' if lo>0 else 'none'}")
