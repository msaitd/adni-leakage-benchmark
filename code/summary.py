"""headline AUC (bootstrap CI) + Youden operating points (binary)."""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import label_binarize

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "outputs")

def oof(t):
    return pd.read_csv(os.path.join(OUT, "oof_%s.csv" % t))

def boot_ci(fn, *a, n=500, seed=0):
    rng = np.random.default_rng(seed)
    base = fn(*a, None)
    N = len(a[0]); idx = np.arange(N); bs = []
    for _ in range(n):
        b = rng.choice(idx, N, replace=True)
        try:
            v = fn(*a, b)
            if v == v:
                bs.append(v)
        except Exception:
            pass
    return base, np.percentile(bs, 2.5), np.percentile(bs, 97.5)

def auc_bin(y, p, b):
    if b is not None:
        y, p = y[b], p[b]
    return roc_auc_score(y, p)

def auc_macro(Y, P, b):
    if b is not None:
        Y, P = Y[b], P[b]
    if (Y.sum(0) == 0).any():
        return np.nan
    return roc_auc_score(Y, P, average="macro")

rows = []
HEAD = [("dx3", "clinical", "logistic_l2"), ("dx3", "multimodal", "hist_gb"),
        ("dx3", "freesurfer", "logistic_l2"),
        ("adcn", "clinical", "logistic_l2"), ("adcn", "freesurfer", "logistic_l2"),
        ("conv36", "clinical", "logistic_l2"), ("conv36", "multimodal", "hist_gb"),
        ("conv36", "freesurfer", "hist_gb"),
        ("conv24", "multimodal", "logistic_l2"), ("conv48", "multimodal", "hist_gb")]
PCOLS = {"dx3": ["p_CN", "p_MCI", "p_AD"], "adcn": ["p_CN", "p_AD"],
         "conv36": ["p_sMCI", "p_pMCI"], "conv24": ["p_sMCI", "p_pMCI"],
         "conv48": ["p_sMCI", "p_pMCI"]}
POS = {"adcn": "AD", "conv36": "pMCI", "conv24": "pMCI", "conv48": "pMCI"}
for task, fs, mdl in HEAD:
    o = oof(task); s = o[(o.featureset == fs) & (o.model == mdl)]
    if not len(s):
        continue
    r = dict(task=task, featureset=fs, model=mdl, n=len(s))
    if task == "dx3":
        Y = label_binarize(s["y_true"], classes=["CN", "MCI", "AD"])
        P = s[PCOLS[task]].values
        a, lo, hi = boot_ci(auc_macro, Y, P)
        r.update(AUC=a, AUC_lo=lo, AUC_hi=hi)
    else:
        pos = POS[task]; yb = (s["y_true"].values == pos).astype(int)
        p = s["p_" + pos].values
        a, lo, hi = boot_ci(auc_bin, yb, p)
        fpr, tpr, thr = roc_curve(yb, p); t = thr[np.argmax(tpr - fpr)]
        yhat = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(yb, yhat).ravel()
        r.update(AUC=a, AUC_lo=lo, AUC_hi=hi,
                 sensitivity=tp/(tp+fn), specificity=tn/(tn+fp),
                 ppv=tp/(tp+fp), npv=tn/(tn+fn), threshold=t)
    rows.append(r)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "headline_results.csv"), index=False)
pd.set_option("display.width", 200)
print(df.round(3).to_string(index=False))
