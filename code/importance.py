"""permutation importance (leakage-aware, test split) for
best diagnostic and conversion models, readable labels, + APOE-stratified table."""
import os, json, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJ, "data"); OUT = os.path.join(PROJ, "outputs"); FIG = os.path.join(PROJ, "figures")
m = pd.read_csv(os.path.join(DATA, "master_features.csv"), low_memory=False)
m = m[m["MMSE"].notna() & m["fs_ICV_mm3"].notna()].copy()
fam = json.load(open(os.path.join(DATA, "feature_families.json")))
cols = fam["demo"] + fam["cognition"] + fam["freesurfer"]

NICE = {"age": "Age", "sex_female": "Female sex", "PTEDUCAT": "Education",
        "APOE4": "APOE e4 alleles", "MMSE": "MMSE", "CDRSB": "CDR-SB",
        "CDGLOBAL": "CDR global", "ADAS11": "ADAS-Cog11", "ADAS13": "ADAS-Cog13",
        "FAQ": "FAQ", "fs_ICV_mm3": "ICV"}
def pretty(c):
    if c in NICE:
        return NICE[c]
    if c.startswith("fs_"):
        p = c.split("_", 2)
        kind = p[1]; rest = p[2] if len(p) > 2 else ""
        mm = re.search(r"of(.+)$", rest)
        region = mm.group(1) if mm else rest
        region = re.sub(r"(?<!^)(?=[A-Z])", " ", region)
        km = {"vol": "vol", "thk": "thick", "area": "area", "thksd": "thick-sd", "hsv": "subfield"}
        return "%s [%s]" % (region.strip(), km.get(kind, kind))
    return c

def imp(task, ycol, classes, fname, title):
    d = m[m[ycol].isin(classes)] if ycol != "baseline_dx" else m[m["baseline_dx"].isin(classes)]
    y = d[ycol if ycol != "baseline_dx" else "baseline_dx"].astype(str).values
    X = d[cols]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("clf", HistGradientBoostingClassifier(max_iter=80, learning_rate=0.12,
                             l2_regularization=1.0, random_state=0))])
    pipe.fit(Xtr, ytr)
    r = permutation_importance(pipe, Xte, yte, n_repeats=5, random_state=0,
                               scoring="balanced_accuracy", n_jobs=2)
    imp = pd.DataFrame({"feature": cols, "label": [pretty(c) for c in cols],
                        "importance": r.importances_mean, "sd": r.importances_std})
    imp = imp.sort_values("importance", ascending=False)
    imp.to_csv(os.path.join(OUT, fname + ".csv"), index=False)
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh(top["label"], top["importance"], xerr=top["sd"], color="#2c7fb8", capsize=2)
    ax.set_xlabel("Permutation importance (drop in balanced accuracy)")
    ax.set_title(title)
    fig.savefig(os.path.join(FIG, fname + ".png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(FIG, fname + ".svg"), bbox_inches="tight")
    plt.close(fig)
    print("\n", title); print(imp.head(10)[["label", "importance"]].to_string(index=False))

import sys
which = sys.argv[1] if len(sys.argv) > 1 else "both"
if which in ("dx3", "both"):
    imp("dx3", "baseline_dx", ["CN", "MCI", "AD"], "fig6_importance_dx3", "Feature importance - CN/MCI/AD (multimodal)")
if which in ("conv36", "both"):
    imp("conv36", "conv_36", ["sMCI", "pMCI"], "fig7_importance_conv36", "Feature importance - pMCI vs sMCI (multimodal)")

# APOE-stratified conversion
conv = m[(m["baseline_dx"] == "MCI") & m["conv_36"].isin(["sMCI", "pMCI"])].copy()
conv["e4"] = conv["APOE4"].map({0: "0", 1: "1", 2: "2"})
tab = conv.groupby("e4").apply(lambda g: pd.Series({
    "N": len(g), "pMCI_n": (g.conv_36 == "pMCI").sum(),
    "pMCI_pct": round(100 * (g.conv_36 == "pMCI").mean(), 1)})).reset_index()
tab.to_csv(os.path.join(OUT, "apoe_stratified_conversion.csv"), index=False)
print("\nAPOE-stratified 36-mo conversion:\n", tab.to_string(index=False))
print("done")
