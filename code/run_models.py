"""RESUMABLE, TIME-BUDGETED model training.
Repeated subject-level CV for every (task, feature_set, model) combo.
Each call processes combos until ~BUDGET seconds, then exits; completed
combos are checkpointed and skipped next call. Run until outputs/_models_DONE.
"""
import os, sys, json, time, glob
import numpy as np
import pandas as pd
import ml_common as mlc

START = time.time()
BUDGET = 18.0
PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJ, "data")
OUT = os.path.join(PROJ, "outputs")
os.makedirs(OUT, exist_ok=True)

master = pd.read_csv(os.path.join(DATA, "master_features.csv"), low_memory=False)
fam = json.load(open(os.path.join(DATA, "feature_families.json")))
master["_has_cog"] = master["MMSE"].notna()
master["_has_fs"] = master["fs_ICV_mm3"].notna()

FEATSETS = {
    "demo": fam["demo"],
    "cognition": fam["cognition"],
    "clinical": fam["demo"] + fam["cognition"],
    "freesurfer": fam["freesurfer"],
    "fs_demo": fam["freesurfer"] + fam["demo"],
    "multimodal": fam["demo"] + fam["cognition"] + fam["freesurfer"],
}
MODELS = ["logistic_l2", "extra_trees", "hist_gb"]

FOLD_COLS = ["task", "featureset", "model", "fold", "repeat", "n_test",
             "balanced_accuracy", "macro_f1", "accuracy", "roc_auc",
             "roc_auc_ovr_macro", "roc_auc_ovr_weighted"]


def cohort(task):
    m = master
    common = m["_has_cog"] & m["_has_fs"]
    if task == "dx3":
        d = m[common & m["baseline_dx"].isin(["CN", "MCI", "AD"])]
        return d, "baseline_dx", ["CN", "MCI", "AD"]
    if task == "adcn":
        d = m[common & m["baseline_dx"].isin(["CN", "AD"])]
        return d, "baseline_dx", ["CN", "AD"]
    if task.startswith("conv"):
        col = "conv_" + task[4:]
        d = m[common & (m["baseline_dx"] == "MCI") & m[col].isin(["pMCI", "sMCI"])]
        return d, col, ["sMCI", "pMCI"]
    raise ValueError(task)


COMBOS = []
for fsname in FEATSETS:
    for mdl in MODELS:
        COMBOS.append(("dx3", fsname, mdl))
        COMBOS.append(("conv36", fsname, mdl))
for fsname in ["clinical", "freesurfer", "multimodal"]:
    for mdl in MODELS:
        COMBOS.append(("adcn", fsname, mdl))
for task in ["conv24", "conv48"]:
    for mdl in MODELS:
        COMBOS.append((task, "multimodal", mdl))

folds_path = os.path.join(OUT, "model_folds.csv")
if "--reset" in sys.argv:
    open(folds_path, "w").close()
    for p in glob.glob(os.path.join(OUT, "oof_*.csv")):
        open(p, "w").close()
    dp = os.path.join(OUT, "_models_DONE")
    if os.path.exists(dp):
        open(dp, "w").close()

done = set()
if os.path.exists(folds_path) and os.path.getsize(folds_path) > 0:
    try:
        prev = pd.read_csv(folds_path)
        done = set(zip(prev["task"], prev["featureset"], prev["model"]))
    except Exception:
        done = set()


def header_needed(path):
    return not (os.path.exists(path) and os.path.getsize(path) > 0)


n_run = 0
for task, fsname, mdl in COMBOS:
    if (task, fsname, mdl) in done:
        continue
    if time.time() - START > BUDGET:
        print("BUDGET reached; exiting (resumable)")
        break
    d, ycol, classes = cohort(task)
    cols = [c for c in FEATSETS[fsname] if c in d.columns]
    X = d[cols].copy()
    y = d[ycol].astype(str).values
    rid = d["RID"].values
    t0 = time.time()
    folds_df, oof_df = mlc.evaluate_combo(X, y, rid, classes, mdl,
                                          n_splits=5, n_repeats=2, seed=42)
    folds_df.insert(0, "task", task)
    folds_df.insert(1, "featureset", fsname)
    oof_df.insert(0, "task", task)
    oof_df.insert(1, "featureset", fsname)
    folds_df = folds_df.reindex(columns=FOLD_COLS)
    folds_df.to_csv(folds_path, mode="a", header=header_needed(folds_path), index=False)
    oofp = os.path.join(OUT, "oof_" + task + ".csv")
    oof_df.to_csv(oofp, mode="a", header=header_needed(oofp), index=False)
    bacc = round(float(folds_df["balanced_accuracy"].mean()), 3)
    dt = round(time.time() - t0, 1)
    print("[done]", task, fsname, mdl, "n=" + str(len(d)), "bAcc=" + str(bacc), "t=" + str(dt))
    done.add((task, fsname, mdl))
    n_run += 1

remaining = [c for c in COMBOS if c not in done]
if not remaining:
    folds = pd.read_csv(folds_path)
    rows = []
    for (task, fsname), g in folds.groupby(["task", "featureset"]):
        s = mlc.summarize(g, None)
        s.insert(0, "task", task)
        s.insert(1, "featureset", fsname)
        rows.append(s)
    pd.concat(rows, ignore_index=True).to_csv(os.path.join(OUT, "model_summary.csv"), index=False)
    open(os.path.join(OUT, "_models_DONE"), "w").write("ok")
    print("ALL COMBOS DONE ->", len(COMBOS), "combos, model_summary.csv written")
else:
    print("progress:", len(COMBOS) - len(remaining), "/", len(COMBOS), "combos done")
