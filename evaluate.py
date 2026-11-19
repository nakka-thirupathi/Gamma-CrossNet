import argparse, yaml, torch, numpy as np
from torch.utils.data import DataLoader
from gammacrossnet.data import MammographyCSVDataset
from gammacrossnet.model import GAMMACrossNet
from gammacrossnet.metrics import classification_metrics, segmentation_metrics

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/default.yaml")
    ap.add_argument("--test_csv",required=True)
    ap.add_argument("--checkpoint",default="checkpoints/best.pt")
    args=ap.parse_args()
    cfg=yaml.safe_load(open(args.config))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds=MammographyCSVDataset(args.test_csv,cfg["image_size"],False)
    dl=DataLoader(ds,batch_size=cfg["batch_size"],shuffle=False)
    model=GAMMACrossNet(cfg["image_size"],cfg["num_classes"],False,
                        cfg["swin_name"],cfg["pvt_name"],cfg["gamma_eps"]).to(device)
    ckpt=torch.load(args.checkpoint,map_location=device)
    model.load_state_dict(ckpt["model"]); model.eval()
    ys=[]; probs=[]; dices=[]; ious=[]
    with torch.no_grad():
        for b in dl:
            x=b["image"].to(device); out=model(x)
            probs.extend(torch.softmax(out["cls_logits"],1).cpu().numpy())
            ys.extend(b["label"].numpy())
            for pm,tm in zip(out["mask"].cpu().numpy(),b["mask"].numpy()):
                m=segmentation_metrics(pm[0],tm[0]); dices.append(m["dice"]); ious.append(m["iou"])
    print("classification:",classification_metrics(ys,probs))
    print("segmentation:",{"dice_mean":float(np.mean(dices)),"iou_mean":float(np.mean(ious))})
if __name__=="__main__": main()
