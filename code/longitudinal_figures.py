"""longitudinal change figures + CONSORT cohort-flow (Figure 1)."""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
HERE=os.path.dirname(os.path.abspath(__file__)); DATA=os.path.join(HERE,"..","data"); FIG=os.path.join(HERE,"..","figures")
C={"CN":"#2c7fb8","MCI":"#f0a500","AD":"#d7301f"}

# ---------- Fig 8: annual change rate by diagnosis ----------
d=pd.read_csv(os.path.join(DATA,"longitudinal_features.csv"),low_memory=False)
meas=[("fs_HIPP_slope","Hippocampus/yr"),("fs_VENT_slope","Ventricle/yr"),("fs_ENT_slope","Entorhinal thick/yr"),
      ("cog_MMSE_slope","MMSE/yr"),("cog_CDRSB_slope","CDR-SB/yr"),("cog_ADAS13_slope","ADAS-Cog13/yr")]
fig,axes=plt.subplots(2,3,figsize=(12,6.5))
for ax,(col,lab) in zip(axes.ravel(),meas):
    g=d[d.baseline_dx.isin(["CN","MCI","AD"])].groupby("baseline_dx")[col].mean().reindex(["CN","MCI","AD"])
    ax.bar(["CN","MCI","AD"],g.values,color=[C[x] for x in ["CN","MCI","AD"]])
    ax.axhline(0,color="k",lw=.8); ax.set_title(lab); ax.set_ylabel("mean annual change")
fig.suptitle("Annual rate of change by baseline diagnosis (FreeSurfer + cognition)")
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(os.path.join(FIG,"fig8_change_rates.png"),dpi=150,bbox_inches="tight")
fig.savefig(os.path.join(FIG,"fig8_change_rates.svg"),bbox_inches="tight"); plt.close(fig)

# ---------- Fig 9: landmark AUC by window ----------
s=pd.read_csv(os.path.join(DATA,"..","outputs","landmark_sensitivity.csv"))
lab=[f"L{r.landmark_mo}/H{r.horizon_mo}\n(n={r.n})" for r in s.itertuples()]
x=np.arange(len(s)); w=0.38
fig,ax=plt.subplots(figsize=(8.5,4.6))
ax.bar(x-w/2,s.AUC_baseline,w,label="Baseline only",color="#9ecae1")
ax.bar(x+w/2,s.AUC_plus_cogchange,w,label="Baseline + early change",color="#d7301f")
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylim(0.7,0.95); ax.set_ylabel("ROC-AUC (pMCI vs sMCI)")
ax.set_title("MCI conversion: early change improves prediction across landmark/horizon windows")
for i,(a,b) in enumerate(zip(s.AUC_baseline,s.AUC_plus_cogchange)):
    ax.text(i-w/2,a+.004,"%.3f"%a,ha="center",fontsize=8); ax.text(i+w/2,b+.004,"%.3f"%b,ha="center",fontsize=8)
ax.legend(loc="lower right")
fig.savefig(os.path.join(FIG,"fig9_landmark_auc.png"),dpi=150,bbox_inches="tight")
fig.savefig(os.path.join(FIG,"fig9_landmark_auc.svg"),bbox_inches="tight"); plt.close(fig)

# ---------- Fig 1: CONSORT-style cohort flow ----------
def box(ax,x,y,w,h,txt,fc="#eef3f8"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.01",fc=fc,ec="#356",lw=1.2))
    ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=9)
def arrow(ax,x1,y1,x2,y2):
    ax.annotate("",xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle="->",lw=1.2,color="#356"))
fig,ax=plt.subplots(figsize=(10,8)); ax.set_xlim(0,10); ax.set_ylim(0,12); ax.axis("off")
box(ax,3,11,4,0.8,"ADNI (ADNIMERGE2) subjects with\nharmonized baseline diagnosis: n = 3,762","#dfeaf5")
arrow(ax,5,11,5,10.5)
box(ax,3,9.6,4,0.9,"Baseline cognition + FreeSurfer (<=12 mo)\nMultimodal diagnostic cohort: n = 3,151\nCN 1,283 / MCI 1,379 / AD 489","#dfeaf5")
arrow(ax,4,9.6,2.3,8.7); arrow(ax,6,9.6,7.7,8.7)
box(ax,0.3,7.8,4,0.9,"Diagnostic models\nCN vs MCI vs AD (n=3,151)\nAD vs CN (n=1,772)","#eafbe7")
box(ax,5.7,7.8,4,0.9,"Baseline MCI: n = 1,379","#fff3da")
arrow(ax,7.7,7.8,7.7,7.0)
box(ax,5.7,6.0,4,0.9,"Conversion cohort (36 mo)\npMCI 287 / sMCI 545 (n=832)","#fff3da")
arrow(ax,7.7,6.0,7.7,5.2)
box(ax,5.4,4.0,4.5,1.0,"Landmark early-change cohort\n(not converted by 12 mo; outcome 12-48 mo)\npMCI 196 / sMCI 355 (n=551)","#fde3e1")
box(ax,0.3,5.0,4,1.4,"Longitudinal change arm\n(FreeSurfer + cognition slopes)\n>=2 visits: 2,076 FS / 2,241 cognitive\nmean follow-up 4.1 y","#eafbe7")
ax.set_title("Figure 1. Cohort derivation and analysis flow",fontsize=11)
fig.savefig(os.path.join(FIG,"fig1_cohort_flow.png"),dpi=150,bbox_inches="tight")
fig.savefig(os.path.join(FIG,"fig1_cohort_flow.svg"),bbox_inches="tight"); plt.close(fig)
print("Saved fig1_cohort_flow, fig8_change_rates, fig9_landmark_auc (png+svg)")
print(os.listdir(FIG))
