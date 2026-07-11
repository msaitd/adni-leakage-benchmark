#!/usr/bin/env python3
# CAT12 ciktilarini denetler: eksik/yarim segmentasyonlar + IQR (kalite) raporu.
# Kullanim:  python audit_cat12.py  [CAT12_KLASORU]
# Default folder: F:\ADNI_derivatives\cat12
# Uretir (bu dosyanin yaninda): cat12_audit.csv, rerun_list.csv, low_iqr_all.csv
import os, re, glob, csv, sys, statistics as st
from collections import Counter

CAT = sys.argv[1] if len(sys.argv) > 1 else r"F:\ADNI_derivatives\cat12"
HERE = os.path.dirname(os.path.abspath(__file__))

def mark2pct(m): return round(max(0, min(100, 105 - m*10)), 1)
def grade(m):
    return ("A" if m<1.5 else "B" if m<2.5 else "C" if m<3.5 else
            "D" if m<4.5 else "E" if m<5.5 else "F")

rows = []
for pt in sorted(os.listdir(CAT)):
    ptp = os.path.join(CAT, pt)
    if not os.path.isdir(ptp) or "_S_" not in pt: continue
    for sc in sorted(os.listdir(ptp)):
        scp = os.path.join(ptp, sc)
        if not (os.path.isdir(scp) and sc.startswith("I")): continue
        mri, rep = os.path.join(scp,"mri"), os.path.join(scp,"report")
        mwp1 = glob.glob(mri+"/mwp1*.nii"); mwp2 = glob.glob(mri+"/mwp2*.nii")
        wm = glob.glob(mri+"/wm*.nii"); xmls = glob.glob(rep+"/cat_*.xml")
        complete = bool(mwp1 and mwp2 and wm and xmls)
        mark = None
        if xmls:
            m = re.search(r"<IQR>([\d.eE+-]+)</IQR>",
                          open(xmls[0], encoding="utf-8", errors="ignore").read())
            if m: mark = float(m.group(1))
        miss = ";".join(n for n,v in [("mwp1",mwp1),("mwp2",mwp2),("wm",wm),("cat_xml",xmls)] if not v)
        rows.append(dict(ptid=pt, image_id=sc, complete=complete, missing=miss,
                         iqr_mark=("" if mark is None else round(mark,3)),
                         iqr_pct=("" if mark is None else mark2pct(mark)),
                         grade=("" if mark is None else grade(mark))))

with open(os.path.join(HERE,"cat12_audit.csv"),"w",newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

comp = [r for r in rows if r["complete"]]
incomp = [r for r in rows if not r["complete"]]
dgrade = [r for r in comp if r["iqr_mark"]!="" and r["iqr_mark"]>=3.5]
cworse = [r for r in comp if r["iqr_mark"]!="" and r["iqr_mark"]>=2.5]

with open(os.path.join(HERE,"rerun_list.csv"),"w",newline="") as f:
    w = csv.writer(f); w.writerow(["ptid","image_id","reason","iqr_mark"])
    for r in incomp: w.writerow([r["ptid"], r["image_id"], "incomplete", ""])
    for r in sorted(dgrade, key=lambda r:-r["iqr_mark"]):
        w.writerow([r["ptid"], r["image_id"], "low_iqr_D", r["iqr_mark"]])

with open(os.path.join(HERE,"low_iqr_all.csv"),"w",newline="") as f:
    w = csv.writer(f); w.writerow(["ptid","image_id","iqr_mark","iqr_pct","grade"])
    for r in sorted(cworse, key=lambda r:-r["iqr_mark"]):
        w.writerow([r["ptid"], r["image_id"], r["iqr_mark"], r["iqr_pct"], r["grade"]])

print("Total scan folders :", len(rows))
print("Tam (mwp1+mwp2+wm+xml):", len(comp))
print("EKSIK / yarim         :", len(incomp), "->", dict(Counter(r["missing"] for r in incomp)))
gr = Counter(r["grade"] for r in comp if r["grade"])
print("IQR grade dagilimi    :", dict(sorted(gr.items())))
pcts = [r["iqr_pct"] for r in comp if r["iqr_pct"]!=""]
if pcts:
    print("IQR%% ort/medyan/min/max: %.1f / %.1f / %.1f / %.1f" %
          (sum(pcts)/len(pcts), st.median(pcts), min(pcts), max(pcts)))
print("rerun_list.csv        :", len(incomp), "eksik +", len(dgrade), "D-grade =", len(incomp)+len(dgrade))
print("low_iqr_all.csv (C+alt):", len(cworse))
