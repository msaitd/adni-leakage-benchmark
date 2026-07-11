"""XAI for the MCI-conversion model: SHAP (global beeswarm +
directional bar + local), logistic odds ratios, and split-conformal coverage."""
import os, re, json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,"..","data"); FIG=os.path.join(HERE,"..","figures"); OUT=os.path.join(HERE,"..","outputs")
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

d=m[(m.baseline_dx=="MCI")&m.conv_36.isin(["sMCI","pMCI"])&m.MMSE.notna()&m.fs_ICV_mm3.notna()].copy()
cols=[c for c in fam["demo"]+fam["cognition"]+fam["freesurfer"] if c in d.columns]
Ximp=SimpleImputer(strategy="median").fit_transform(d[cols]); y=(d.conv_36=="pMCI").astype(int).values
names=[pretty(c) for c in cols]
model=ExtraTreesClassifier(n_estimators=300,n_jobs=2,class_weight="balanced_subsample",random_state=0).fit(Ximp,y)
expl=shap.TreeExplainer(model)
rng=np.random.default_rng(0); samp=rng.choice(len(Ximp),min(600,len(Ximp)),replace=False)
sv=expl.shap_values(Ximp[samp])           # (n, feat, 2)
svp=sv[:,:,1] if sv.ndim==3 else sv        # pMCI class
imp=np.abs(svp).mean(0); sign=svp.mean(0); order=np.argsort(imp)[::-1][:15]

# directional global bar
plt.figure(figsize=(7,5.5))
cols_o=order[::-1]; colors=["#d7301f" if sign[i]>0 else "#2c7fb8" for i in cols_o]
plt.barh([names[i] for i in cols_o], imp[cols_o], color=colors)
plt.xlabel("mean |SHAP| (impact on pMCI probability)"); plt.title("SHAP feature importance — MCI conversion\n(red = increases, blue = decreases conversion risk)")
plt.tight_layout(); plt.savefig(os.path.join(FIG,"figS2_shap_conversion_bar.png"),dpi=150,bbox_inches="tight"); plt.savefig(os.path.join(FIG,"figS2_shap_conversion_bar.svg"),bbox_inches="tight"); plt.close()

# beeswarm
try:
    shap.summary_plot(svp, Ximp[samp], feature_names=names, max_display=15, show=False)
    plt.title("SHAP summary — MCI conversion (pMCI)"); plt.tight_layout()
    plt.savefig(os.path.join(FIG,"figS1_shap_conversion_beeswarm.png"),dpi=150,bbox_inches="tight"); plt.close()
    bee="ok"
except Exception as e:
    bee="fail: %s"%e

# local explanation bars for one pMCI and one sMCI (correctly predicted)
pred=model.predict(Ximp[samp])
def local(idx,fname,title):
    s=svp[idx]; o=np.argsort(np.abs(s))[::-1][:10][::-1]
    plt.figure(figsize=(7,4.2)); c=["#d7301f" if s[i]>0 else "#2c7fb8" for i in o]
    plt.barh([names[i] for i in o], s[o], color=c); plt.axvline(0,color="k",lw=.8)
    plt.xlabel("SHAP value (→ pMCI)"); plt.title(title); plt.tight_layout()
    plt.savefig(os.path.join(FIG,fname),dpi=150,bbox_inches="tight"); plt.close()
ip=np.where((y[samp]==1)&(pred==1))[0]; isb=np.where((y[samp]==0)&(pred==0))[0]
if len(ip): local(ip[0],"figS3a_local_pMCI.png","Local SHAP — a correctly identified pMCI")
if len(isb): local(isb[0],"figS3b_local_sMCI.png","Local SHAP — a correctly identified sMCI")

# logistic odds ratios (clinical features, standardized -> OR per 1 SD)
clin=[c for c in fam["demo"]+fam["cognition"] if c in d.columns]
Xc=SimpleImputer(strategy="median").fit_transform(d[clin]); Xc=StandardScaler().fit_transform(Xc)
res=sm.Logit(y, sm.add_constant(Xc)).fit(disp=0)
ci=res.conf_int()
orr=pd.DataFrame({"feature":["intercept"]+[pretty(c) for c in clin],"OR_per_SD":np.exp(res.params),
                  "CI_low":np.exp(ci[:,0]),"CI_high":np.exp(ci[:,1]),"p":res.pvalues}).round(3)
orr=orr[orr.feature!="intercept"].sort_values("OR_per_SD",ascending=False)
orr.to_csv(os.path.join(OUT,"odds_ratios_conversion.csv"),index=False)

# split-conformal (binary) coverage at 90%
Xtr,Xtmp,ytr,ytmp=train_test_split(Ximp,y,test_size=0.4,stratify=y,random_state=1)
Xcal,Xte,ycal,yte=train_test_split(Xtmp,ytmp,test_size=0.5,stratify=ytmp,random_state=1)
cm=ExtraTreesClassifier(n_estimators=300,n_jobs=2,class_weight="balanced_subsample",random_state=0).fit(Xtr,ytr)
pcal=cm.predict_proba(Xcal); scores=1-pcal[np.arange(len(ycal)),ycal]
alpha=0.1; qhat=np.quantile(scores,np.ceil((len(scores)+1)*(1-alpha))/len(scores))
pte=cm.predict_proba(Xte); sets=[ [c for c in [0,1] if (1-pte[i,c])<=qhat] for i in range(len(yte)) ]
cov=np.mean([yte[i] in sets[i] for i in range(len(yte))]); size=np.mean([len(s) for s in sets])
conf=dict(target_coverage=0.90, empirical_coverage=round(float(cov),3), mean_set_size=round(float(size),2), n_test=len(yte))
json.dump(conf, open(os.path.join(OUT,"conformal_conversion.json"),"w"), indent=2)

print("beeswarm:",bee)
print("top SHAP (conv):", [names[i] for i in order[:6]])
print("\nOdds ratios (per 1 SD, conversion):"); print(orr.head(8).to_string(index=False))
print("\nConformal:",conf)
print("saved figS1/figS2/figS3, odds_ratios_conversion.csv, conformal_conversion.json")
