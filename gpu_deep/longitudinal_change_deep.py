"""Leakage-safe image-based LONGITUDINAL change for MCI->AD conversion (earliest early-follow-up 6-24mo; horizon 36mo).
Mirrors the rigor of fixed_train_cnn_cv + fixed_fuse_and_report:
 * ONE row per subject (asserted unique RID) -> stratified split is subject-level;
 * a SINGLE shared fold assignment reused by the CNN and the fusion;
 * per fold the CNN is trained ONLY on outer-train (inner split for early stopping);
   test subjects never enter that fold's training -> their embeddings/probs are OOF;
 * per fold, per subject, we store split(train/test), probs, and 512 embeddings, and
   VALIDATE (512 emb, finite, probs sum to 1, no dup RID, train/test RID disjoint);
 * fold-aligned fusion: fit fusion on outer-train rows, evaluate on untouched outer-test
   rows, so every subject is predicted EXACTLY ONCE out-of-fold (asserted).
Predictors are from [0, t_fu] only (baseline + earliest 6-24mo follow-up); outcome in (t_fu, 36mo] -> no leakage.
Change map: annualized delta = (follow-up - baseline)/dt on modulated MNI maps.
RUN ON GPU after step1d + build_longitudinal_manifest.py.  Needs nibabel."""
import os, json, numpy as np, pandas as pd, torch, torch.nn.functional as F, nibabel as nib
from monai.networks.nets import resnet18
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline; from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE); D=os.path.join(ROOT,"data")
IMG=(96,96,96); EPOCHS=30; FOLDS=5; PATIENCE=8; LR=1e-4; SEED=42; CLS=["sMCI","pMCI"]
EMB=[f"emb_{j:03d}" for j in range(512)]
dev="cuda" if torch.cuda.is_available() else "cpu"

man=pd.read_csv(os.path.join(HERE,"long_manifest.csv"))
man["RID"]=pd.to_numeric(man["RID"],errors="raise").astype(int)
assert man["RID"].is_unique, "long_manifest must be one row per subject (unique RID)"
assert set(man["label"].astype(str))<=set(CLS), "unexpected labels"
for c in ["path_mwp1_bl","path_mwp2_bl","path_mwp1_fu","path_mwp2_fu"]:
    bad=[p for p in man[c] if not os.path.isfile(str(p))]
    assert not bad, f"{len(bad)} missing CAT12 maps in {c} (run step1d first): {bad[:5]}"
fam=json.load(open(os.path.join(D,"feature_families.json")))
clin=[c for c in fam["demo"]+fam["cognition"] if c in man.columns]
assert clin, "no clinical columns in long_manifest"
y=man["label"].map(CLS.index).to_numpy()
SPLITS=list(StratifiedKFold(FOLDS,shuffle=True,random_state=SEED).split(man,y))  # SHARED across CNN + fusion

def load(path):
    v=np.asarray(nib.load(str(path)).dataobj,dtype=np.float32)
    t=F.interpolate(torch.from_numpy(v)[None,None],size=IMG,mode="trilinear",align_corners=False)[0,0]
    return t
def zscore(x): m,s=x.mean(),x.std(); return (x-m)/(s+1e-6)
def make_input(r,mode,aug=False):
    b1,b2=load(r.path_mwp1_bl),load(r.path_mwp2_bl)
    if mode=="change":
        f1,f2=load(r.path_mwp1_fu),load(r.path_mwp2_fu); dt=float(r.dt_years); c1,c2=(f1-b1)/dt,(f2-b2)/dt
    else: c1,c2=b1,b2
    x=torch.stack([zscore(c1),zscore(c2)],0)
    if aug:
        if np.random.rand()<0.5: x=torch.flip(x,dims=[1])
        x=x+torch.randn_like(x)*0.02
    return x
class DS(torch.utils.data.Dataset):
    def __init__(s,df,mode,aug): s.df=df.reset_index(drop=True); s.mode=mode; s.aug=aug
    def __len__(s): return len(s.df)
    def __getitem__(s,i):
        r=s.df.iloc[i]; return {"x":make_input(r,s.mode,s.aug),"y":int(CLS.index(str(r.label))),"rid":int(r.RID)}
def ld(df,mode,aug,sh): return torch.utils.data.DataLoader(DS(df,mode,aug),batch_size=4,shuffle=sh,num_workers=0,pin_memory=(dev=="cuda"))
def infer(model,loader):
    model.eval(); P=[];Y=[];E=[];R=[];cap={}
    h=model.fc.register_forward_hook(lambda m,i,o:cap.__setitem__("e",i[0].detach().cpu()))
    try:
        with torch.no_grad():
            for b in loader:
                x=b["x"].to(dev)
                with torch.autocast(dev,enabled=(dev=="cuda")): lo=model(x)
                P.append(torch.softmax(lo.float(),1).cpu().numpy()); Y.append(b["y"].numpy())
                E.append(cap["e"].numpy()); R+=[int(v) for v in b["rid"].numpy()]
    finally: h.remove()
    return np.concatenate(Y),np.concatenate(P),np.concatenate(E),np.array(R,int)
def rows(fold,split,Y,Pp,Ee,Rr):
    d=pd.DataFrame({"fold":fold,"split":split,"RID":Rr.astype(int),"y_true":np.array(CLS)[Y]})
    d["p_sMCI"]=Pp[:,0]; d["p_pMCI"]=Pp[:,1]
    d=pd.concat([d,pd.DataFrame(Ee,columns=EMB)],axis=1)
    return d

def train_mode(mode):
    frames=[]
    for fold,(tr,te) in enumerate(SPLITS):
        tro,teo=man.iloc[tr],man.iloc[te]
        assert not (set(tro.RID)&set(teo.RID)), f"{mode} fold{fold}: train/test RID overlap"
        itr,iva=train_test_split(np.arange(len(tro)),test_size=0.15,stratify=tro.label,random_state=SEED+fold)
        torch.manual_seed(SEED+fold)
        model=resnet18(spatial_dims=3,n_input_channels=2,num_classes=2,shortcut_type="B").to(dev)
        cc=np.bincount(tro.iloc[itr].label.map(CLS.index).values,minlength=2)
        w=torch.tensor(cc.sum()/(2*np.maximum(cc,1)),dtype=torch.float32,device=dev)
        opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
        sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=EPOCHS)
        scaler=torch.amp.GradScaler("cuda",enabled=(dev=="cuda")); lf=torch.nn.CrossEntropyLoss(weight=w)
        best=-1; bs=None; wait=0
        for ep in range(EPOCHS):
            model.train()
            for b in ld(tro.iloc[itr],mode,True,True):
                x=b["x"].to(dev); yb=b["y"].to(dev); opt.zero_grad(set_to_none=True)
                with torch.autocast(dev,enabled=(dev=="cuda")): loss=lf(model(x),yb)
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            sch.step()
            yl,pp,_,_=infer(model,ld(tro.iloc[iva],mode,False,False))
            sc=balanced_accuracy_score(yl,pp.argmax(1))
            if sc>best: best=sc; bs={k:v.cpu().clone() for k,v in model.state_dict().items()}; wait=0
            else: wait+=1
            if wait>=PATIENCE: break
        model.load_state_dict(bs)
        ytr,ptr,etr,rtr=infer(model,ld(tro,mode,False,False))   # outer-train (in-sample) for fusion fit
        yte,pte,ete,rte=infer(model,ld(teo,mode,False,False))   # outer-test  (OOF)
        fr=pd.concat([rows(fold,"train",ytr,ptr,etr,rtr),rows(fold,"test",yte,pte,ete,rte)],ignore_index=True)
        # validate
        assert sum(c.startswith("emb_") for c in fr.columns)==512
        assert np.isfinite(fr[["p_sMCI","p_pMCI"]+EMB].to_numpy()).all()
        assert np.allclose(fr[["p_sMCI","p_pMCI"]].sum(1),1.0,atol=1e-4)
        assert not (set(fr[fr.split=="train"].RID)&set(fr[fr.split=="test"].RID))
        frames.append(fr); print(f"  [{mode}] fold{fold} val_bAcc={best:.3f}",flush=True)
    return frames

print(f"=== leakage-safe longitudinal image-change | n={len(man)} pMCI={int((man.label=='pMCI').sum())} sMCI={int((man.label=='sMCI').sum())} | dev={dev} ===",flush=True)
feat={m:train_mode(m) for m in ["change","baseline"]}
# OOF completeness assert (each subject once)
for m in feat:
    oof=pd.concat([f[f.split=="test"] for f in feat[m]],ignore_index=True)
    assert oof.RID.is_unique and set(oof.RID)==set(man.RID), f"{m}: OOF not exactly one prediction per subject"

def auc(yt,pp): return roc_auc_score((np.array(yt)==CLS[1]).astype(int),pp)
def pipe(): return Pipeline([("i",SimpleImputer(strategy="median")),("s",StandardScaler()),("c",LogisticRegression(max_iter=4000,class_weight="balanced",random_state=SEED))])
mrx=man.set_index("RID")
# fold-aligned fusion (fit on outer-train, eval untouched outer-test)
def fuse(colbuild):
    preds=[]
    for fold in range(FOLDS):
        fc=feat["change"][fold]; fb=feat["baseline"][fold]
        tr_c=fc[fc.split=="train"]; te_c=fc[fc.split=="test"]; tr_b=fb[fb.split=="train"]; te_b=fb[fb.split=="test"]
        assert not (set(tr_c.RID)&set(te_c.RID)), "fusion fold train/test overlap"
        Xtr,ytr,_=colbuild(tr_c,tr_b); Xte,yte,rte=colbuild(te_c,te_b)
        m=pipe().fit(Xtr,ytr); p=m.predict_proba(Xte)[:,list(m.named_steps["c"].classes_).index(CLS[1])]
        preds.append(pd.DataFrame({"RID":rte,"y":yte,"p":p}))
    o=pd.concat(preds,ignore_index=True); assert o.RID.is_unique
    return auc(o.y,o.p),len(o)
def cb_clin(tc,tb): d=mrx.loc[tc.RID]; return d[clin].to_numpy(), tc.y_true.values, tc.RID.values
def cb_clin_change(tc,tb): d=mrx.loc[tc.RID]; return np.hstack([d[clin].to_numpy(),tc[EMB].to_numpy()]), tc.y_true.values, tc.RID.values
def cb_clin_base(tc,tb): d=mrx.loc[tb.RID]; return np.hstack([d[clin].to_numpy(),tb[EMB].to_numpy()]), tb.y_true.values, tb.RID.values
# deep-only = OOF CNN probs directly (already leakage-safe, no fusion)
def deep_only(mode):
    o=pd.concat([f[f.split=="test"][["RID","y_true","p_pMCI"]] for f in feat[mode]],ignore_index=True)
    return auc(o.y_true,o.p_pMCI),len(o)
S=[]
a,n=deep_only("change");    S.append(("deep_change_only",a,n))
a,n=deep_only("baseline");  S.append(("deep_baseline_only",a,n))
a,n=fuse(cb_clin);          S.append(("clinical",a,n))
a,n=fuse(cb_clin_change);   S.append(("clinical+deep_change",a,n))
a,n=fuse(cb_clin_base);     S.append(("clinical+deep_baseline",a,n))
df=pd.DataFrame(S,columns=["featureset","AUC","n"]); df.to_csv(os.path.join(HERE,"long_change_summary.csv"),index=False)
print("\n=== OUT-OF-FOLD AUC (leakage-safe) ===")
for _,r in df.iterrows(): print(f"  {r.featureset:24} AUC={r.AUC:.3f} (n={int(r.n)})")
print("\nKEY: clinical+deep_change vs clinical  |  deep_change vs deep_baseline")
print("saved long_change_summary.csv")
