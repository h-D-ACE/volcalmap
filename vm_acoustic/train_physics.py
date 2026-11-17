"""Stage-0 physical measurement training scaffold, NOT a timbre-map model.

--smoke performs an engineering test on generated data. Formal reference
training needs --train plus a manifest with disjoint train/dev/test speakers.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

from acoustics import ROOT,np,read_audio
from scipy import signal
import torch
from torch import nn
from run_probe import save_json


def spectrum_features(y,sr):
    """512 fixed-frequency features; no lexical text or speaker ID input."""
    n=max(32,int(sr*.032))
    if len(y)<n:y=np.pad(y,(0,n-len(y)))
    f,t,z=signal.stft(y,fs=sr,nperseg=n,noverlap=n//2,nfft=max(1024,n),boundary=None)
    logp=np.log(np.maximum(abs(z)**2,1e-12))
    grid=np.linspace(80,5000,256)
    mean=np.interp(grid,f,np.mean(logp,axis=1))
    spread=np.interp(grid,f,np.std(logp,axis=1))
    mean-=mean.mean() # remove pure recording-gain offset, retain spectral slope
    return np.r_[mean,spread].astype(np.float32)


class PhysicsHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(512,128),nn.GELU(),nn.Linear(128,64),nn.GELU(),nn.Linear(64,3))
    def forward(self,x):return self.net(x)


def read_manifest(path):
    path=Path(path)
    obj=json.loads(path.read_text())
    rows=obj["rows"]
    splits={s:{r["speaker"] for r in rows if r["split"]==s} for s in ("train","dev","test")}
    if any(not s for s in splits.values()):raise ValueError("Require nonempty train/dev/test speaker groups")
    if (splits["train"]&splits["dev"] or splits["train"]&splits["test"] or splits["dev"]&splits["test"]):raise ValueError("Speaker leakage")
    X=[];Y=[]
    for r in rows:
        p=(path.parent/r["path"]).resolve()
        if hashlib.sha256(p.read_bytes()).hexdigest()!=r["sha256"]:raise ValueError("Audio hash mismatch")
        y,sr=read_audio(p)
        start=r["start"]+.2*(r["end"]-r["start"])
        end=r["end"]-.2*(r["end"]-r["start"])
        X.append(spectrum_features(y[int(start*sr):int(end*sr)],sr))
        refs=np.array(r["reference_formants"])[1:]
        targets=[]
        for j in range(3):
            valid=refs[:,j]>0
            targets.append(float(np.median(np.log(refs[valid,j]))) if valid.any() else float('nan'))
        Y.append(targets)
    return np.array(X),np.array(Y,dtype=np.float32),rows,obj


def smoke_data():
    from reliability_probe import synth
    xs=[];ys=[];rows=[]
    rng=np.random.default_rng(710)
    for i in range(32):
        f=[rng.uniform(350,750),rng.uniform(1200,2200),rng.uniform(2500,3200)]
        y,sr=synth(f,rng.uniform(100,240),40,[3800,4800],i)
        xs.append(spectrum_features(y,sr));ys.append(np.log(f));rows.append(dict(speaker=f"synthetic_{i}",split="train" if i<24 else "dev"))
    return np.array(xs),np.array(ys,dtype=np.float32),rows,dict(kind="synthetic engineering smoke; no generalization claim")


def main():
    p=argparse.ArgumentParser()
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke",action="store_true");mode.add_argument("--train",action="store_true")
    p.add_argument("--manifest");p.add_argument("--epochs",type=int,default=100)
    p.add_argument("--output",default=str(ROOT/"training_runs/physics"));p.add_argument("--device",default="cpu")
    p.add_argument("--resume");a=p.parse_args()
    if a.train and not a.manifest:p.error("--train requires --manifest")
    if a.epochs < 1:p.error("--epochs must be positive")
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if not a.resume and any((out/n).exists() for n in ("best.pt","last.pt","run.json")):
        p.error("Output already contains a run; use --resume or a new output directory")
    if a.resume and Path(a.resume).resolve().parent != out.resolve():
        p.error("Resume into the checkpoint directory to preserve the existing best checkpoint")
    torch.manual_seed(710);random.seed(710);np.random.seed(710);torch.set_num_threads(2)
    X,Y,rows,manifest=smoke_data() if a.smoke else read_manifest(a.manifest)
    train=np.array([r["split"]=="train" for r in rows]);dev=np.array([r["split"]=="dev" for r in rows])
    xm=X[train].mean(0);xs=np.maximum(X[train].std(0),.1)
    ym=np.nanmean(Y[train],0);ys=np.maximum(np.nanstd(Y[train],0),.03)
    if not np.isfinite(ym).all():raise ValueError("Missing all training labels for a target")
    provenance=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()+X.tobytes()+Y.tobytes()).hexdigest()
    device=torch.device(a.device)
    x=torch.tensor((X-xm)/xs,device=device);y=torch.tensor(np.nan_to_num((Y-ym)/ys),device=device)
    target_mask=torch.tensor(np.isfinite(Y),device=device)
    def masked_loss(pred,truth,mask):
        if not bool(mask.any()):raise ValueError("No valid targets in loss")
        return nn.functional.smooth_l1_loss(pred[mask],truth[mask])
    model=PhysicsHead().to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
    begin=0;best=float("inf")
    if a.resume:
        ck=torch.load(a.resume,map_location=device,weights_only=True)
        if ck["data_hash"]!=provenance:raise ValueError("Resume data mismatch")
        model.load_state_dict(ck["model"]);optimizer.load_state_dict(ck["optimizer"])
        begin=ck["epoch"]+1;best=ck["best_dev_loss"]
    history=[]
    if a.resume and (out/"run.json").exists():
        previous=json.loads((out/"run.json").read_text())
        if previous["data_hash"]!=provenance:raise ValueError("History data mismatch")
        history=[r for r in previous["history"] if r["epoch"]<begin]
    initial=None
    for epoch in range(begin,begin+(20 if a.smoke else a.epochs)):
        model.train();optimizer.zero_grad()
        pred=model(x[train]);loss=masked_loss(pred,y[train],target_mask[train])
        physical=pred*torch.tensor(ys,device=device)+torch.tensor(ym,device=device)
        order=torch.relu(physical[:,0]-physical[:,1]+.05).mean()+torch.relu(physical[:,1]-physical[:,2]+.05).mean()
        loss=loss+.1*order
        if not torch.isfinite(loss):raise ValueError("Nonfinite loss")
        if initial is None:initial=float(loss.detach())
        loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        model.eval()
        with torch.no_grad():dl=float(masked_loss(model(x[dev]),y[dev],target_mask[dev]))
        improved=dl<best;best=min(best,dl)
        ck=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,best_dev_loss=best,
            data_hash=provenance,normalization=dict(x_mean=xm.tolist(),x_std=xs.tolist(),y_mean=ym.tolist(),y_std=ys.tolist()),
            model_kind="physical_formant_head_v0",not_a_timbre_embedding=True,smoke_only=a.smoke,
            torch_version=str(torch.__version__),seed=710)
        torch.save(ck,out/"last.pt")
        if improved:torch.save(ck,out/"best.pt")
        history.append(dict(epoch=epoch,train_loss=float(loss.detach()),dev_loss=dl))
    # Reload roundtrip is checked rather than merely asserting a file exists.
    restored=PhysicsHead().to(device)
    restored.load_state_dict(torch.load(out/"last.pt",map_location=device,weights_only=True)["model"])
    with torch.no_grad():assert torch.allclose(model(x),restored(x),atol=1e-7)
    if a.smoke and not a.resume:assert history[-1]["train_loss"]<initial
    save_json(out/"run.json",dict(smoke_only=a.smoke,parameters=sum(p.numel() for p in model.parameters()),
        device=str(device),history=history,checkpoint_roundtrip="passed",data_hash=provenance,
        note="No timbre/prosody coordinate has been trained; no held-out test evaluation during fitting"))
    print(json.dumps(dict(parameters=sum(p.numel() for p in model.parameters()),initial_loss=initial,
        final_loss=history[-1]["train_loss"],checkpoint_roundtrip="passed",smoke_only=a.smoke)),flush=True)

if __name__=="__main__":main()
