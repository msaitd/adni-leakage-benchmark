"""Table 1 (cohort characteristics + group tests),
Table 2 (model performance: fold bAcc/F1 + bootstrap-CI OOF AUC),
plus binary operating points (Youden sens/spec/PPV/NPV) from OOF."""
import os, json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import label_binarize

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJ, "data"); OUT = os.path.join(PROJ, "outputs")
m = pd.read_csv(os.path.join(DATA, "master_features.csv"), low_memory=False)
m = m[m["MMSE"].notna() & m["fs_ICV_mm3"].notna()].copy()   # multimodal cohort

# FreeSurfer summary ROIs (sum L+R / mean L+R)
def cols_like(sub, kind=None, exclude=None):
    out = []
    for c in m.columns:
        if not c.startswith("fs_"):
            continue
        if sub.lower() not in c.lower():
            continue
        if kind and not c.startswith("fs_" + kind):
            continue
        if exclude and exclude.lower() in c.lower():
            continue
        out.append(c)
    return out

m["HIPP"] = m[cols_like("Hippocampus", "vol", exclude="Inferior")].sum(axis=1, min_count=1)
m["ENTORH_thk"] = m[cols_like("Entorhinal", "thk")].mean(axis=1)
m["VENT"] = m[cols_like("LateralVentricle", "vol", exclude="Inferior")].sum(axis=1, min_count=1)

CONT = [("age", "Age, y"), ("PTEDUCAT", "Education, y"), ("MMSE", "MMSE"),
        ("CDRSB", "CDR-SB"), ("ADAS13", "ADAS-Cog13"), ("FAQ", "FAQ"),
        ("HIPP", "Hippocampus/ICV"), ("ENTORH_thk", "Entorhinal thickness"),
        ("VENT", "Lateral ventricle/ICV")]


def desc_table(df, group_col, groups, fname, title):
    rows = []
    rows.append([title] + groups + ["p"])
    rows.append(["N"] + [str((df[group_col] == g).sum()) for g in groups] + [""])
    # sex
    fem = [100 * (df[df[group_col] == g]["SEX"] == "Female").mean() for g in groups]
    ct = pd.crosstab(df[group_col], df["SEX"])
    try:
        p = stats.chi2_contingency(ct)[1]
    except Exception:
        p = np.nan
    rows.append(["Female, %"] + ["%.1f" % f for f in fem] + ["%.3g" % p])
    # APOE4 carrier
    df = df.assign(_apoe=(df["APOE4"] >= 1).astype(float))
    car = [100 * df[df[group_col] == g]["_apoe"].mean() for g in groups]
    ct = pd.crosstab(df[group_col], (df["APOE4"] >= 1))
    try:
        p = stats.chi2_contingency(ct)[1]
    except Exception:
        p = np.nan
    rows.append(["APOE e4 carrier, %"] + ["%.1f" % c for c in car] + ["%.3g" % p])
    for col, lab in CONT:
        vals = [df[df[group_col] == g][col].dropna() for g in groups]
        cell = ["%.2f (%.2f)" % (v.mean(), v.std()) for v in vals]
        try:
            if len(groups) > 2:
                p = stats.f_oneway(*vals)[1]
            else:
                p = stats.ttest_ind(*vals, equal_var=False)[1]
        except Exception:
            p = np.nan
        rows.append([lab] + cell + ["%.3g" % p])
    out = pd.DataFrame(rows[1:], columns=rows[0])
    out.to_csv(os.path.join(OUT, fname), index=False)
    print("\n==", title, "==")
    print(out.to_string(index=False))
    return out


diag = m[m["baseline_dx"].isin(["CN", "MCI", "AD"])]
desc_table(diag, "baseline_dx", ["CN", "MCI", "AD"],
           "table1_diagnostic.csv", "Diagnostic cohort (baseline)")
conv = m[(m["baseline_dx"] == "MCI") & m["conv_36"].isin(["sMCI", "pMCI"])]
desc_table(conv, "conv_36", ["sMCI", "pMCI"],
           "table1_conversion.csv", "MCI conversion cohort (36 mo)")


# ---- Table 2: model performance with OOF AUC + bootstrap CI ----
def boot_auc(y, P, classes, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    if len(classes) == 2:
        yb = (np.asarray(y) == classes[1]).astype(int)
        s = P[:, 1]
        base = roc_auc_score(yb, s)
        bs = []
        idx = np.arange(len(yb))
        for _ in range(n):
            b = rng.choice(idx, len(idx), replace=True)
            if len(np.unique(yb[b])) < 2:
                continue
            bs.append(roc_auc_score(yb[b], s[b]))
    else:
        Y = label_binarize(y, classes=classes)
        base = roc_auc_score(Y, P, average="macro")
        bs = []
        idx = np.arange(len(y))
        for _ in range(n):
            b = rng.choice(idx, len(idx), replace=True)
            if (Y[b].sum(0) == 0).any():
                continue
            bs.append(roc_auc_score(Y[b], P[b], average="macro"))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return base, lo, hi


folds = pd.read_csv(os.path.join(OUT, "model_folds.csv"))
PROBA = {"dx3": (["CN", "MCI", "AD"], ["p_CN", "p_MCI", "p_AD"]),
         "adcn": (["CN", "AD"], ["p_CN", "p_AD"]),
         "conv36": (["sMCI", "pMCI"], ["p_sMCI", "p_pMCI"]),
         "conv24": (["sMCI", "pMCI"], ["p_sMCI", "p_pMCI"]),
         "conv48": (["sMCI", "pMCI"], ["p_sMCI", "p_pMCI"])}
rows = []
for task, (classes, pcols) in PROBA.items():
    oof = pd.read_csv(os.path.join(OUT, "oof_%s.csv" % task))
    for (fs, mdl), g in folds[folds.task == task].groupby(["featureset", "model"]):
        ba = g["balanced_accuracy"]; f1 = g["macro_f1"]
        sub = oof[(oof.featureset == fs) & (oof.model == mdl)]
        auc, alo, ahi = (np.nan, np.nan, np.nan)
        if len(sub):
            yv = sub["y_true"].values; P = sub[pcols].values
            try:
                if len(classes) == 2:
                    auc = roc_auc_score((yv == classes[1]).astype(int), P[:, 1])
                else:
                    auc = roc_auc_score(label_binarize(yv, classes=classes), P, average="macro")
            except Exception:
                pass
        rows.append(dict(task=task, featureset=fs, model=mdl,
                         bAcc=ba.mean(), bAcc_lo=ba.mean()-1.96*ba.std(ddof=1)/np.sqrt(len(ba)),
                         bAcc_hi=ba.mean()+1.96*ba.std(ddof=1)/np.sqrt(len(ba)),
                         macroF1=f1.mean(), AUC=auc, AUC_lo=alo, AUC_hi=ahi))
perf = pd.DataFrame(rows)
perf.to_csv(os.path.join(OUT, "table2_model_performance.csv"), index=False)
print("\n== Table 2 saved (rows=%d) ==" % len(perf))
print(perf[perf.task == "dx3"][["featureset", "model", "bAcc", "AUC", "AUC_lo", "AUC_hi"]].round(3).to_string(index=False))
print("done")
