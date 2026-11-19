import argparse, os, random, yaml, numpy as np, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from gammacrossnet.data import MammographyCSVDataset
from gammacrossnet.model import GAMMACrossNet
from gammacrossnet.losses import joint_loss

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_epoch(model, loader, optimizer, device, cfg, train=True):
    model.train(train)
    total = 0.0
    for batch in tqdm(loader, leave=False):
        x = batch["image"].to(device); m = batch["mask"].to(device); y = batch["label"].to(device)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type="cuda", enabled=cfg["amp"] and device.type=="cuda"):
                out = model(x)
                loss, parts = joint_loss(out, m, y, cfg["segmentation_loss_weight"],
                                         cfg["classification_loss_weight"])
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    args=ap.parse_args()
    cfg=yaml.safe_load(open(args.config))
    seed_everything(cfg["seed"])
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds=MammographyCSVDataset(args.train_csv,cfg["image_size"],augment=True)
    val_ds=MammographyCSVDataset(args.val_csv,cfg["image_size"],augment=False)
    train_dl=DataLoader(train_ds,batch_size=cfg["batch_size"],shuffle=True,
                        num_workers=cfg["num_workers"],pin_memory=True)
    val_dl=DataLoader(val_ds,batch_size=cfg["batch_size"],shuffle=False,
                      num_workers=cfg["num_workers"],pin_memory=True)
    model=GAMMACrossNet(cfg["image_size"],cfg["num_classes"],cfg["pretrained"],
                        cfg["swin_name"],cfg["pvt_name"],cfg["gamma_eps"]).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],
                          weight_decay=cfg["weight_decay"])
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg["epochs"])
    os.makedirs(cfg["checkpoint_dir"],exist_ok=True)
    best=float("inf")
    for epoch in range(1,cfg["epochs"]+1):
        tr=run_epoch(model,train_dl,opt,device,cfg,True)
        va=run_epoch(model,val_dl,opt,device,cfg,False)
        sched.step()
        print(f"epoch={epoch} train_loss={tr:.5f} val_loss={va:.5f}")
        if va<best:
            best=va
            torch.save({"epoch":epoch,"model":model.state_dict(),"config":cfg},
                       os.path.join(cfg["checkpoint_dir"],"best.pt"))
if __name__=="__main__": main()
