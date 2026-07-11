"""Build the subject-level baseline manifest and define analysis cohorts.

Outputs (expanded_study/data/):
  manifest_subject_baseline.csv  - one row/subject: baseline dx + baseline features + flags
  dx_trajectory.csv              - long: RID, month_from_bl, DX  (for survival/descriptives)
  cohort_diagnostic.csv          - baseline CN/MCI/AD cohort (label = baseline_dx)
  cohort_conversion.csv          - baseline MCI with pMCI/sMCI labels at 24/36/48 months
  cohort_flow.md                 - CONSORT-style inclusion/exclusion counts

Rigor rules:
  * Baseline = earliest dated diagnosis visit per subject.
  * Diagnostic label = baseline diagnosis (CN/MCI/AD).
  * Conversion labels use ONLY follow-up diagnosis (the outcome); predictors are
    baseline-only (assembled later). sMCI requires sufficient follow-up so that a
    non-converter is genuinely "stable", not merely under-observed.
"""
import os
import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJ, "data")


def L(name):
    return pd.read_csv(os.path.join(DATA, name), low_memory=False)


dx = L("dxsum.csv");  dx["EXAMDATE"] = pd.to_datetime(dx["EXAMDATE"])
demo = L("demographics.csv")
apoe = L("apoe.csv")
fs = L("freesurfer_fsx7.csv"); fs["EXAMDATE"] = pd.to_datetime(fs["EXAMDATE"])

# ---- baseline diagnosis = earliest dated DX visit per subject --------------
dx = dx.sort_values(["RID", "EXAMDATE"])
base = dx.groupby("RID").first().reset_index()[["RID", "EXAMDATE", "DX", "ORIGPROT", "PTID"]]
base = base.rename(columns={"EXAMDATE": "baseline_date", "DX": "baseline_dx",
                            "ORIGPROT": "phase"})

# ---- diagnosis trajectory (months from baseline) ---------------------------
dxj = dx.merge(base[["RID", "baseline_date"]], on="RID", how="left")
dxj["month"] = (dxj["EXAMDATE"] - dxj["baseline_date"]).dt.days / 30.4375
dxj = dxj[dxj["month"] >= -1]  # drop pre-baseline noise
dxj[["RID", "month", "DX", "EXAMDATE"]].to_csv(os.path.join(DATA, "dx_trajectory.csv"), index=False)

# follow-up summary per subject
fu = dxj.groupby("RID")["month"].max().rename("max_followup_m")
nvis = dxj.groupby("RID")["DX"].count().rename("n_dx_visits")
# first AD conversion month (for baseline non-AD subjects)
ad = dxj[dxj["DX"] == "AD"].groupby("RID")["month"].min().rename("first_AD_month")
base = base.merge(fu, on="RID").merge(nvis, on="RID").merge(ad, on="RID", how="left")

# ---- baseline age ----------------------------------------------------------
base = base.merge(demo[["RID", "SEX", "PTEDUCAT", "PTDOBYY"]], on="RID", how="left")
base["age"] = base["baseline_date"].dt.year + base["baseline_date"].dt.dayofyear / 365.25 - base["PTDOBYY"]
base = base.merge(apoe[["RID", "APOE4", "GENOTYPE"]], on="RID", how="left")


# ---- nearest baseline cognitive score (within +/- 365 days) ----------------
def attach_nearest(base, tbl, valcols, window=365):
    t = L(tbl); t["VISDATE"] = pd.to_datetime(t["VISDATE"])
    m = t.merge(base[["RID", "baseline_date"]], on="RID", how="inner")
    m["dd"] = (m["VISDATE"] - m["baseline_date"]).dt.days.abs()
    m = m[m["dd"] <= window].sort_values(["RID", "dd"])
    g = m.groupby("RID").first().reset_index()[["RID"] + valcols]
    return base.merge(g, on="RID", how="left")


base = attach_nearest(base, "cog_mmse.csv", ["MMSE"])
base = attach_nearest(base, "cog_cdr.csv", ["CDGLOBAL", "CDRSB"])
base = attach_nearest(base, "cog_adas.csv", ["ADAS11", "ADAS13"])
base = attach_nearest(base, "cog_faq.csv", ["FAQ"])

# ---- nearest baseline FreeSurfer scan (within +/- 365 days) ----------------
fsm = fs[["RID", "EXAMDATE", "IMAGEUID"]].merge(base[["RID", "baseline_date"]], on="RID", how="inner")
fsm["dd"] = (fsm["EXAMDATE"] - fsm["baseline_date"]).dt.days.abs()
fsm = fsm[fsm["dd"] <= 365].sort_values(["RID", "dd"]).groupby("RID").first().reset_index()
fsm = fsm.rename(columns={"EXAMDATE": "fs_date", "dd": "fs_offset_days"})
base = base.merge(fsm[["RID", "IMAGEUID", "fs_date", "fs_offset_days"]], on="RID", how="left")
base["has_fs"] = base["IMAGEUID"].notna()
base["has_cog"] = base["MMSE"].notna()
base["has_apoe"] = base["APOE4"].notna()

base.to_csv(os.path.join(DATA, "manifest_subject_baseline.csv"), index=False)

# ---- Diagnostic cohort -----------------------------------------------------
diag = base[base["baseline_dx"].isin(["CN", "MCI", "AD"])].copy()
diag.to_csv(os.path.join(DATA, "cohort_diagnostic.csv"), index=False)

# ---- MCI conversion cohort -------------------------------------------------
mci = base[base["baseline_dx"] == "MCI"].copy()


def conv_label(row, W):
    conv = row["first_AD_month"]
    if pd.notna(conv) and conv <= W + 3:      # converts within window (+3m slack)
        return "pMCI"
    # non-converter within window: stable only if followed long enough
    if row["max_followup_m"] >= W - 6:
        return "sMCI"
    return np.nan                              # insufficient follow-up -> censored


for W in (24, 36, 48):
    mci["conv_%d" % W] = mci.apply(lambda r: conv_label(r, W), axis=1)
mci["conversion_month"] = mci["first_AD_month"]
mci.to_csv(os.path.join(DATA, "cohort_conversion.csv"), index=False)

# ---- CONSORT-style flow ----------------------------------------------------
lines = []
lines.append("# Cohort flow\n")
lines.append("## Source (local ADNIMERGE2)\n")
lines.append("- Subjects with >=1 harmonized diagnosis: %d\n" % base["RID"].nunique())
lines.append("\n## Diagnostic cohort (baseline label)\n")
vc = diag["baseline_dx"].value_counts()
lines.append("- Total baseline CN/MCI/AD: %d (CN %d, MCI %d, AD %d)\n"
             % (len(diag), vc.get("CN", 0), vc.get("MCI", 0), vc.get("AD", 0)))
lines.append("- with baseline cognition (MMSE): %d\n" % int(diag["has_cog"].sum()))
lines.append("- with baseline FreeSurfer scan (<=365d): %d\n" % int(diag["has_fs"].sum()))
lines.append("- with APOE genotype: %d\n" % int(diag["has_apoe"].sum()))
both = diag[diag["has_cog"] & diag["has_fs"]]
vcb = both["baseline_dx"].value_counts()
lines.append("- with BOTH cognition + FreeSurfer (multimodal cohort): %d (CN %d, MCI %d, AD %d)\n"
             % (len(both), vcb.get("CN", 0), vcb.get("MCI", 0), vcb.get("AD", 0)))
lines.append("\n## MCI conversion cohort (baseline MCI = %d)\n" % len(mci))
for W in (24, 36, 48):
    v = mci["conv_%d" % W].value_counts()
    cens = mci["conv_%d" % W].isna().sum()
    lines.append("- %d-month window: pMCI %d, sMCI %d, censored(insufficient follow-up) %d\n"
                 % (W, v.get("pMCI", 0), v.get("sMCI", 0), cens))
mci_mm = mci[mci["has_cog"] & mci["has_fs"]]
v36 = mci_mm["conv_36"].value_counts()
lines.append("- 36-month window WITH cognition+FreeSurfer: pMCI %d, sMCI %d\n"
             % (v36.get("pMCI", 0), v36.get("sMCI", 0)))
open(os.path.join(DATA, "cohort_flow.md"), "w").write("".join(lines))
print("".join(lines))
print("median baseline age %.1f | %% female %.1f | median educ %.0f"
      % (diag["age"].median(), 100 * (diag["SEX"] == "Female").mean(), diag["PTEDUCAT"].median()))
print("Saved manifest + cohorts to", DATA)
