"""manuscript figures from OOF predictions + fold metrics."""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJ, "outputs"); FIG = os.path.join(PROJ, "figures")
os.makedirs(FIG, exist_ok=True)
folds = pd.read_csv(os.path.join(OUT, "model_folds.csv"))
plt.rcParams.update({"font.size": 11})

def oof(task):
    return pd.read_csv(os.path.join(OUT, "oof_%s.csv" % task))

def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=150, bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".svg"), bbox_inches="tight")
    plt.close(fig)

C = {"CN": "#2c7fb8", "MCI": "#f0a500", "AD": "#d7301f",
     "clinical": "#2c7fb8", "cognition": "#41ab5d", "demo": "#bdbdbd",
     "freesurfer": "#f0a500", "fs_demo": "#fe9929", "multimodal": "#d7301f"}

# ---------- Fig: feature-set ablation (balanced accuracy) ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
order = ["demo", "cognition", "clinical", "freesurfer", "fs_demo", "multimodal"]
for ax, task, ttl, chance in [(axes[0], "dx3", "CN vs MCI vs AD (n=3151)", 1/3),
                              (axes[1], "conv36", "pMCI vs sMCI, 36 mo (n=832)", 0.5)]:
    d = folds[folds.task == task]
    fss = [f for f in order if f in d.featureset.unique()]
    best = (d.groupby("featureset")["balanced_accuracy"].mean())
    means, los, his = [], [], []
    for f in fss:
        g = d[d.featureset == f].groupby("model")["balanced_accuracy"].mean()
        bm = g.idxmax()
        v = d[(d.featureset == f) & (d.model == bm)]["balanced_accuracy"]
        means.append(v.mean())
        h = 1.96 * v.std(ddof=1) / np.sqrt(len(v))
        los.append(h); his.append(h)
    x = np.arange(len(fss))
    ax.bar(x, means, yerr=los, color=[C.get(f, "#888") for f in fss], capsize=3)
    ax.axhline(chance, ls="--", color="k", lw=1, label="chance")
    ax.set_xticks(x); ax.set_xticklabels(fss, rotation=30, ha="right")
    ax.set_ylim(0, 1.0); ax.set_ylabel("Balanced accuracy")
    ax.set_title(ttl); ax.legend(loc="lower right", fontsize=9)
fig.suptitle("Diagnostic and prognostic performance by feature family (best model per family)")
save(fig, "fig2_ablation_balanced_accuracy")

# ---------- Fig: ROC curves ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
# panel A: dx3 per-class OvR, best multimodal model (hist_gb)
o = oof("dx3"); sub = o[(o.featureset == "multimodal") & (o.model == "hist_gb")]
classes = ["CN", "MCI", "AD"]; Y = label_binarize(sub["y_true"], classes=classes)
P = sub[["p_CN", "p_MCI", "p_AD"]].values
ax = axes[0]
for i, c in enumerate(classes):
    fpr, tpr, _ = roc_curve(Y[:, i], P[:, i]); a = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=C[c], label="%s (AUC=%.3f)" % (c, a))
ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_title("A. CN/MCI/AD — multimodal (one-vs-rest)")
ax.set_xlabel("1 - specificity"); ax.set_ylabel("Sensitivity"); ax.legend(fontsize=9, loc="lower right")
# panel B: AD vs CN, clinical vs freesurfer
ax = axes[1]; o = oof("adcn")
for fs, lab, col in [("clinical", "Clinical", "#2c7fb8"), ("freesurfer", "FreeSurfer", "#f0a500")]:
    s = o[(o.featureset == fs) & (o.model == "logistic_l2")]
    yb = (s["y_true"].values == "AD").astype(int)
    fpr, tpr, _ = roc_curve(yb, s["p_AD"].values); a = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=col, label="%s (AUC=%.3f)" % (lab, a))
ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_title("B. AD vs CN")
ax.set_xlabel("1 - specificity"); ax.set_ylabel("Sensitivity"); ax.legend(fontsize=9, loc="lower right")
# panel C: conv36 clinical vs freesurfer vs multimodal
ax = axes[2]; o = oof("conv36")
for fs, lab, col in [("clinical", "Clinical", "#2c7fb8"), ("freesurfer", "FreeSurfer", "#f0a500"),
                     ("multimodal", "Multimodal", "#d7301f")]:
    s = o[(o.featureset == fs) & (o.model == "logistic_l2")]
    yb = (s["y_true"].values == "pMCI").astype(int)
    fpr, tpr, _ = roc_curve(yb, s["p_pMCI"].values); a = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=col, label="%s (AUC=%.3f)" % (lab, a))
ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_title("C. pMCI vs sMCI (36 mo)")
ax.set_xlabel("1 - specificity"); ax.set_ylabel("Sensitivity"); ax.legend(fontsize=9, loc="lower right")
save(fig, "fig3_roc_curves")

# ---------- Fig: confusion matrices ----------
def cm_panel(ax, task, fs, mdl, classes, title):
    o = oof(task); s = o[(o.featureset == fs) & (o.model == mdl)]
    cm = confusion_matrix(s["y_true"], s["y_pred"], labels=classes)
    cmn = cm / cm.sum(1, keepdims=True)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, "%d\n%.0f%%" % (cm[i, j], 100*cmn[i, j]),
                    ha="center", va="center", fontsize=9,
                    color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
cm_panel(axes[0], "dx3", "multimodal", "hist_gb", ["CN", "MCI", "AD"],
         "CN/MCI/AD — multimodal HistGB")
cm_panel(axes[1], "conv36", "clinical", "logistic_l2", ["sMCI", "pMCI"],
         "pMCI/sMCI — clinical logistic")
save(fig, "fig4_confusion_matrices")

# ---------- Fig: deep ablation ----------
dab = pd.read_csv(os.path.join(OUT, "deep_ablation_summary.csv"))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, task, ttl, chance in [(axes[0], "dx3", "CN/MCI/AD (n=60 imaged subset)", 1/3),
                              (axes[1], "adcn", "AD vs CN (n=40 imaged subset)", 0.5)]:
    d = dab[dab.task == task]
    fss = ["clinical", "freesurfer", "deep", "clinical+deep", "all"]
    means = [d[(d.featureset == f)]["balanced_accuracy_mean"].max() for f in fss]
    ax.bar(range(len(fss)), means, color=["#2c7fb8", "#f0a500", "#756bb1", "#31a354", "#d7301f"])
    ax.axhline(chance, ls="--", color="k", lw=1)
    ax.set_xticks(range(len(fss))); ax.set_xticklabels(fss, rotation=30, ha="right")
    ax.set_ylim(0, 1.05); ax.set_ylabel("Balanced accuracy"); ax.set_title(ttl)
fig.suptitle("Deep-feature ablation: frozen 3D ResNet-101 embeddings vs clinical / FreeSurfer")
save(fig, "fig5_deep_ablation")
print("figures saved to", FIG)
print(os.listdir(FIG))
