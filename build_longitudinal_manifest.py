"""long_manifest.csv: matches baseline + follow-up CAT12 maps; annualized change
uses dt_years, landmark conversion label (pMCI/sMCI), clinical features. CPU."""
import os, glob, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
DERIV=r"F:\ADNI_derivatives\cat12"; D=os.path.join(ROOT,"data")
pl=pd.read_csv(os.path.join(ROOT,"run_local","process_list_followup.csv"))
man=pd.read_csv(os.path.join(HERE,"fixed_deep_manifest.csv")); man["PTID"]=man.PTID.astype(str)
p2r=dict(zip(man.PTID,man.RID))
def find(ptid,iid,kind):
    g=glob.glob(os.path.join(DERIV,ptid,iid,"mri",f"{kind}*.nii")); return g[0] if g else None
rows=[]; miss=[]
for _,r in pl.iterrows():
    pt=str(r.ptid); fu=str(r.image_id); bl=str(r.baseline_image_id)
    b1=find(pt,bl,"mwp1"); b2=find(pt,bl,"mwp2"); f1=find(pt,fu,"mwp1"); f2=find(pt,fu,"mwp2")
    if not all([b1,b2,f1,f2]): miss.append(f"{pt} bl={bl} fu={fu} (follow-up not segmented?)"); continue
    rows.append(dict(RID=p2r.get(pt),PTID=pt,label=r.label,dt_years=round(float(r.followup_month)/12.0,3),
                     path_mwp1_bl=b1,path_mwp2_bl=b2,path_mwp1_fu=f1,path_mwp2_fu=f2))
out=pd.DataFrame(rows)
if len(out)==0:
    import sys; print("ERROR: no baseline/follow-up matches. Run step1d (follow-up CAT12) first."); sys.exit(1)
if miss:
    print(f"WARNING: {len(miss)} subjects have NO follow-up segmentation (run step1d first):")
    for m in miss[:10]: print("  ",m)
master=pd.read_csv(os.path.join(D,"master_features.csv"),low_memory=False)
import json; fam=json.load(open(os.path.join(D,"feature_families.json")))
clin=[c for c in fam["demo"]+fam["cognition"] if c in master.columns]
out=out.merge(master[["RID"]+clin],on="RID",how="left")
out.to_csv(os.path.join(HERE,"long_manifest.csv"),index=False)
print(f"long_manifest.csv: {len(out)} subjects | pMCI={int((out.label=='pMCI').sum())} sMCI={int((out.label=='sMCI').sum())}")
