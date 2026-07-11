"""Assemble subject-level baseline FEATURE tables for modeling.

Feature families
  demo      : age, sex_female, education, APOE4 allele count
  cognition : MMSE, CDRSB, CDGLOBAL, ADAS11, ADAS13, FAQ
  freesurfer: UCSFFSX7 morphometry, ICV-normalized volumes/areas + raw thickness
  deep      : frozen 3D ResNet-101 pool5 embeddings from CAT12 maps (imaged subset)

Outputs (expanded_study/data/):
  master_features.csv      - large cohort: clinical + demo + FreeSurfer + labels + RID group
  feature_families.json    - column lists per family
  deep_subset_features.csv  - 60-subject imaged subset: clinical+FS+deep + labels
"""
import os, json, re
import numpy as np
import pandas as pd

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJ, "data")
RP = os.path.join(PROJ, "..", "research_pipeline", "outputs")  # existing deep features


def L(p):
    return pd.read_csv(p, low_memory=False)


man = L(os.path.join(DATA, "manifest_subject_baseline.csv"))
conv = L(os.path.join(DATA, "cohort_conversion.csv"))[["RID", "conv_24", "conv_36", "conv_48", "conversion_month"]]
fs = L(os.path.join(DATA, "freesurfer_fsx7.csv"))
fsd = L(os.path.join(DATA, "freesurfer_dictionary.csv"))

# ---- demographic encodings -------------------------------------------------
man["sex_female"] = (man["SEX"] == "Female").astype(float)
man.loc[man["SEX"].isna(), "sex_female"] = np.nan
demo_cols = ["age", "sex_female", "PTEDUCAT", "APOE4"]
cog_cols = ["MMSE", "CDRSB", "CDGLOBAL", "ADAS11", "ADAS13", "FAQ"]

# ---- FreeSurfer: ICV-normalize, readable names -----------------------------
icv = "ST10CV"
codes = [c for c in fsd["code"] if c in fs.columns and c != icv]
kindmap = dict(zip(fsd["code"], fsd["kind"]))
regmap = dict(zip(fsd["code"], fsd["region"]))


def clean(r):
    return re.sub(r"[^A-Za-z0-9]", "", str(r))


fs_feat = fs[["RID", "IMAGEUID", icv]].copy()
prefix = {"volume": "vol", "cort_volume": "vol", "surf_area": "area",
          "thick_avg": "thk", "thick_sd": "thksd", "hippo_subfield": "hsv"}
fscol_names = []
for c in codes:
    k = kindmap.get(c, "")
    name = "fs_%s_%s" % (prefix.get(k, k), clean(regmap.get(c, c)))
    vals = pd.to_numeric(fs[c], errors="coerce")
    if k in ("volume", "cort_volume", "surf_area", "hippo_subfield"):
        vals = vals / pd.to_numeric(fs[icv], errors="coerce")   # ICV-normalized
    fs_feat[name] = vals
    fscol_names.append(name)
fs_feat["fs_ICV_mm3"] = pd.to_numeric(fs[icv], errors="coerce")
fscol_names.append("fs_ICV_mm3")
fs_feat = fs_feat.drop_duplicates("IMAGEUID")

# join FS features to baseline scan by IMAGEUID
master = man.merge(conv, on="RID", how="left")
master = master.merge(fs_feat.drop(columns=[icv]), on=["RID", "IMAGEUID"], how="left")

fam = {"demo": demo_cols, "cognition": cog_cols, "freesurfer": fscol_names}
keep = (["RID", "PTID", "phase", "baseline_dx", "baseline_date", "age", "SEX",
         "max_followup_m", "conversion_month", "conv_24", "conv_36", "conv_48",
         "has_fs", "has_cog", "has_apoe", "fs_offset_days"]
        + demo_cols + cog_cols + fscol_names)
keep = list(dict.fromkeys([c for c in keep if c in master.columns]))
master[keep].to_csv(os.path.join(DATA, "master_features.csv"), index=False)
json.dump(fam, open(os.path.join(DATA, "feature_families.json"), "w"), indent=2)
print("master_features: %d subjects | demo %d, cognition %d, freesurfer %d feats"
      % (len(master), len(demo_cols), len(cog_cols), len(fscol_names)))
print("  baseline_dx:", master["baseline_dx"].value_counts().to_dict())

# ---- Deep imaged subset (60) ----------------------------------------------
try:
    deep = L(os.path.join(RP, "subject_baseline_deep_multimodal_features.csv"))
    # subject_norm == PTID. map to RID via manifest PTID.
    p2r = dict(zip(man["PTID"].astype(str), man["RID"]))
    deep["RID"] = deep["subject_norm"].astype(str).map(p2r)
    deep_cols = [c for c in deep.columns if c.startswith("resnet101_pool5")]
    dd = deep[["RID", "subject_norm"] + deep_cols].dropna(subset=["RID"])
    dd["RID"] = dd["RID"].astype(int)
    # attach clinical+FS+labels for the same subjects
    sub = master[master["RID"].isin(dd["RID"])]
    out = sub.merge(dd, on="RID", how="left")
    out.to_csv(os.path.join(DATA, "deep_subset_features.csv"), index=False)
    json.dump({"deep": deep_cols}, open(os.path.join(DATA, "deep_family.json"), "w"))
    print("deep_subset_features: %d subjects | %d deep dims | dx %s"
          % (len(out), len(deep_cols), out["baseline_dx"].value_counts().to_dict()))
except Exception as e:
    print("deep subset skipped:", e)
print("done")
