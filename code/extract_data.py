"""Extract clean analysis tables from the local ADNIMERGE2 R data package.

Reads .rda files from:  adni images/ADNIMERGE2/data/
Writes clean CSVs to:   expanded_study/data/

No raw images, MATLAB, or F:\\ADNI access required. All ADNI clinical,
cognitive, demographic, genetic and FreeSurfer (UCSFFSX7) tables come from the
locally bundled ADNIMERGE2 package.
"""
import os, re, warnings
import pyreadr
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # expanded_study/
ROOT = os.path.dirname(PROJ)
RDA  = os.path.join(ROOT, "adni images", "ADNIMERGE2", "data")
OUT  = os.path.join(PROJ, "data")
os.makedirs(OUT, exist_ok=True)


def load(name):
    r = pyreadr.read_r(os.path.join(RDA, name + ".rda"))
    return r[list(r.keys())[0]]


# 1. Diagnosis (DXSUM): harmonized DIAGNOSIS -> CN / MCI / AD
dx = load("DXSUM")[["RID", "PTID", "ORIGPROT", "COLPROT", "VISCODE",
                    "VISCODE2", "EXAMDATE", "DIAGNOSIS"]].copy()
dx["RID"] = pd.to_numeric(dx["RID"], errors="coerce")
dx["EXAMDATE"] = pd.to_datetime(dx["EXAMDATE"], errors="coerce")
dx["DX"] = dx["DIAGNOSIS"].map({"CN": "CN", "MCI": "MCI", "Dementia": "AD"})
dx = dx.dropna(subset=["RID", "DX", "EXAMDATE"])
dx["RID"] = dx["RID"].astype(int)
dx.to_csv(os.path.join(OUT, "dxsum.csv"), index=False)
print("[DXSUM]   %5d dx visits | %d subjects | %s"
      % (len(dx), dx["RID"].nunique(), dx["DX"].value_counts().to_dict()))

# 2. Registry (visit dates)
reg = load("REGISTRY")[["RID", "VISCODE", "VISCODE2", "EXAMDATE"]].copy()
reg["RID"] = pd.to_numeric(reg["RID"], errors="coerce")
reg["EXAMDATE"] = pd.to_datetime(reg["EXAMDATE"], errors="coerce")
reg = reg.dropna(subset=["RID"]); reg["RID"] = reg["RID"].astype(int)
reg.to_csv(os.path.join(OUT, "registry.csv"), index=False)
print("[REGISTRY] %5d rows | %d subjects" % (len(reg), reg["RID"].nunique()))

# 3. Demographics (PTDEMOG) -> subject level
pt = load("PTDEMOG")
keep = [c for c in ["RID", "PTGENDER", "PTDOBYY", "PTEDUCAT", "PTETHCAT",
                    "PTRACCAT", "PTMARRY"] if c in pt.columns]
pt = pt[keep].copy()
pt["RID"] = pd.to_numeric(pt["RID"], errors="coerce")
for c in ["PTDOBYY", "PTEDUCAT"]:
    if c in pt.columns:
        pt[c] = pd.to_numeric(pt[c], errors="coerce")
pt = pt.dropna(subset=["RID"]); pt["RID"] = pt["RID"].astype(int)
ptg = pt.groupby("RID").agg(
    lambda s: s.dropna().iloc[0] if s.dropna().size else np.nan).reset_index()


def sexmap(v):
    s = str(v).strip().lower()
    if s in ("1", "1.0", "male", "m"):
        return "Male"
    if s in ("2", "2.0", "female", "f"):
        return "Female"
    return np.nan


ptg["SEX"] = ptg["PTGENDER"].map(sexmap)
ptg.to_csv(os.path.join(OUT, "demographics.csv"), index=False)
print("[PTDEMOG] %5d subjects | %s | educ median %s"
      % (len(ptg), ptg["SEX"].value_counts().to_dict(), ptg["PTEDUCAT"].median()))

# 4. APOE genotype
ap = load("APOERES")[["RID", "GENOTYPE"]].copy()
ap["RID"] = pd.to_numeric(ap["RID"], errors="coerce")
ap = ap.dropna(subset=["RID"]); ap["RID"] = ap["RID"].astype(int)


def count_e4(g):
    g = str(g)
    if "/" not in g:
        return np.nan
    a, b = g.split("/")[:2]
    try:
        return int(a == "4") + int(b == "4")
    except Exception:
        return np.nan


ap["APOE4"] = ap["GENOTYPE"].map(count_e4)
ap = ap.dropna(subset=["GENOTYPE"]).groupby("RID").first().reset_index()
ap.to_csv(os.path.join(OUT, "apoe.csv"), index=False)
print("[APOE]    %5d subjects | e4 %s"
      % (len(ap), ap["APOE4"].value_counts(dropna=False).to_dict()))

# 5. Cognitive (visit level, with VISDATE): MMSE, CDR, ADAS, FAQ
def viz(df):
    df = df.copy()
    df["RID"] = pd.to_numeric(df["RID"], errors="coerce")
    df = df.dropna(subset=["RID"]); df["RID"] = df["RID"].astype(int)
    return df


mmse = viz(load("MMSE"))[["RID", "VISCODE2", "VISDATE", "MMSCORE"]].rename(
    columns={"MMSCORE": "MMSE"})
cdr = viz(load("CDR"))[["RID", "VISCODE2", "VISDATE", "CDGLOBAL", "CDRSB"]]
adas = viz(load("ADAS"))[["RID", "VISCODE2", "VISDATE", "TOTSCORE", "TOTAL13"]].rename(
    columns={"TOTSCORE": "ADAS11", "TOTAL13": "ADAS13"})
faq = viz(load("FAQ"))[["RID", "VISCODE2", "VISDATE", "FAQTOTAL"]].rename(
    columns={"FAQTOTAL": "FAQ"})
for nm, d in [("mmse", mmse), ("cdr", cdr), ("adas", adas), ("faq", faq)]:
    d["VISDATE"] = pd.to_datetime(d["VISDATE"], errors="coerce")
    for c in d.columns:
        if c not in ("RID", "VISCODE2", "VISDATE"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d.to_csv(os.path.join(OUT, "cog_%s.csv" % nm), index=False)
    print("[%s] %5d rows | %d subjects" % (nm.upper(), len(d), d["RID"].nunique()))

# 6. FreeSurfer (UCSFFSX7): single consistent FS7 processing, max coverage
fs = load("UCSFFSX7")
dd = load("DATADIC")
dd7 = dd[dd["TBLNAME"].astype(str) == "UCSFFSX7"][["FLDNAME", "TEXT"]].drop_duplicates("FLDNAME")
desc = dict(zip(dd7["FLDNAME"], dd7["TEXT"]))

morph_re = re.compile(r"^ST\d+(SV|CV|SA|TA|TS|HS)$")
morph_cols = [c for c in fs.columns if morph_re.match(c)]
meta = [c for c in ["RID", "PTID", "COLPROT", "ORIGPROT", "VISCODE", "VISCODE2",
                    "EXAMDATE", "IMAGEUID", "STATUS", "OVERALLQC", "VERSION",
                    "FLDSTRENG"] if c in fs.columns]
fs2 = fs[meta + morph_cols].copy()
fs2["RID"] = pd.to_numeric(fs2["RID"], errors="coerce")
fs2 = fs2.dropna(subset=["RID"]); fs2["RID"] = fs2["RID"].astype(int)
fs2["EXAMDATE"] = pd.to_datetime(fs2["EXAMDATE"], errors="coerce")
for c in morph_cols:
    fs2[c] = pd.to_numeric(fs2[c], errors="coerce")

# Drop only explicit FreeSurfer failures
qc = fs2["OVERALLQC"].astype(str).str.lower()
fs2 = fs2[~qc.isin(["fail"])].copy()

suffix_kind = {"SV": "volume", "CV": "cort_volume", "SA": "surf_area",
               "TA": "thick_avg", "TS": "thick_sd", "HS": "hippo_subfield"}
rows = []
for c in morph_cols:
    t = str(desc.get(c, ""))
    region = re.sub(r"^(Volume.*?of |Surface Area of |Cortical Thickness Average of "
                    r"|Cortical Thickness Standard Deviation of |Hippocampal "
                    r"Subfields Volume of )", "", t).strip()
    rows.append({"code": c, "kind": suffix_kind.get(c[-2:], c[-2:]),
                 "region": region, "text": t})
pd.DataFrame(rows).to_csv(os.path.join(OUT, "freesurfer_dictionary.csv"), index=False)
fs2.to_csv(os.path.join(OUT, "freesurfer_fsx7.csv"), index=False)
print("[FS UCSFFSX7] %5d scans | %d subjects | %d features | ICV(ST10CV)=%s"
      % (len(fs2), fs2["RID"].nunique(), len(morph_cols), "ST10CV" in fs2.columns))
print("Done ->", OUT)
