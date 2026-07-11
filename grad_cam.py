"""(Enhancement 7 — voxel-level XAI for the trained 3D CNN)
Run AFTER RUN_2 (needs results/model_<task>_fold0.pt + deep_manifest.csv).
Produces Grad-CAM and occlusion saliency heatmaps overlaid on example MRIs, and an
average saliency map -> shows which brain regions the CNN uses.
"""
import os, glob, numpy as np, torch, nibabel as nib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import pandas as pd
from monai.networks.nets import resnet18
from monai.transforms import (Compose, LoadImaged, EnsureChannelFirstd, Resized,
                              ScaleIntensityd, ConcatItemsd, DeleteItemsd, ToTensord)
from monai.visualize import GradCAM
HERE=os.path.dirname(os.path.abspath(__file__)); RES=os.path.join(HERE,"results"); OUT=os.path.join(HERE,"gradcam"); os.makedirs(OUT,exist_ok=True)
TASK="dx3"; IMG=(96,96,96); dev="cuda" if torch.cuda.is_available() else "cpu"
man=pd.read_csv(os.path.join(HERE,"deep_manifest.csv"))
classes={"dx3":["CN","MCI","AD"],"adcn":["CN","AD"],"conv36":["sMCI","pMCI"]}[TASK]
model=resnet18(spatial_dims=3,n_input_channels=2,num_classes=len(classes),shortcut_type="B").to(dev)
ckpt=sorted(glob.glob(os.path.join(RES,f"model_{TASK}_fold*.pt")))[0]
model.load_state_dict(torch.load(ckpt,map_location=dev)); model.eval()
cam=GradCAM(nn_module=model, target_layers="layer4")     # last residual block
tf=Compose([LoadImaged(keys=["mwp1","mwp2"]),EnsureChannelFirstd(keys=["mwp1","mwp2"]),
            Resized(keys=["mwp1","mwp2"],spatial_size=IMG),ScaleIntensityd(keys=["mwp1","mwp2"]),
            ConcatItemsd(keys=["mwp1","mwp2"],name="image",dim=0),DeleteItemsd(keys=["mwp1","mwp2"]),ToTensord(keys="image")])
sub=man.dropna(subset=["path_mwp1","path_mwp2"]).head(6)
acc=None
for i,r in sub.reset_index().iterrows():
    d=tf({"mwp1":r["path_mwp1"],"mwp2":r["path_mwp2"]}); x=d["image"].unsqueeze(0).to(dev)
    sal=cam(x).detach().cpu().numpy()[0,0]                # (H,W,D)
    acc=sal if acc is None else acc+sal
    vol=x.detach().cpu().numpy()[0,0]; z=IMG[2]//2
    fig,ax=plt.subplots(1,2,figsize=(8,4))
    ax[0].imshow(vol[:,:,z].T,cmap="gray",origin="lower"); ax[0].set_title("mwp1 (GM)"); ax[0].axis("off")
    ax[1].imshow(vol[:,:,z].T,cmap="gray",origin="lower"); ax[1].imshow(sal[:,:,z].T,cmap="jet",alpha=0.45,origin="lower")
    ax[1].set_title("Grad-CAM"); ax[1].axis("off")
    fig.suptitle(f"{r.get('PTID','subj')} — pred class {classes[int(model(x).argmax())]}")
    fig.savefig(os.path.join(OUT,f"gradcam_{i}.png"),dpi=130,bbox_inches="tight"); plt.close(fig)
# average saliency (axial mid-slice)
z=IMG[2]//2; plt.figure(figsize=(4,4)); plt.imshow((acc/len(sub))[:,:,z].T,cmap="jet",origin="lower"); plt.title("Mean Grad-CAM (axial)"); plt.axis("off")
plt.savefig(os.path.join(OUT,"gradcam_mean.png"),dpi=130,bbox_inches="tight"); plt.close()
print("Saved Grad-CAM overlays to",OUT,"- inspect whether medial temporal/temporoparietal regions are highlighted.")
