"""Evaluate a locked measurement checkpoint on disjoint speakers vs Burg.

Runs once after development selection; not a timbre-map quality score.
"""
import argparse
import hashlib
import json
from pathlib import Path
from collections import defaultdict
from train_physics import ROOT,np,torch,PhysicsHead,read_manifest
from acoustics import read_audio
from reference_benchmark import formant_track
from run_probe import save_json,csv_write

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest",default=str(ROOT/"data_reference/training_manifest.json"))
    parser.add_argument("--checkpoint",default=str(ROOT/"training_runs/physics_pilot_v0/best.pt"))
    parser.add_argument("--output",default=str(ROOT/"training_runs/physics_pilot_v0/evaluation"))
    a=parser.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    x,y,rows,meta=read_manifest(a.manifest)
    ck=torch.load(a.checkpoint,map_location="cpu",weights_only=True)
    if ck["smoke_only"]:raise ValueError("Smoke checkpoint is not a measured model")
    provenance=hashlib.sha256(json.dumps(meta,sort_keys=True).encode()+x.tobytes()+y.tobytes()).hexdigest()
    if ck["data_hash"]!=provenance:raise ValueError("Checkpoint and evaluation manifest/features differ")
    model=PhysicsHead();model.load_state_dict(ck["model"]);model.eval()
    norm=ck["normalization"]
    with torch.no_grad():pred=model(torch.tensor((x-np.array(norm["x_mean"]))/np.array(norm["x_std"]),dtype=torch.float32)).numpy()*norm["y_std"]+norm["y_mean"]
    values=[]
    for i,r in enumerate(rows):
        if r["split"] not in ("dev","test"):continue
        audio,sr=read_audio(Path(a.manifest).parent/r["path"])
        times=r["start"]+(r["end"]-r["start"])*np.arange(2,9)/10
        for method,estimate in (("neural",np.exp(pred[i])),("burg25",np.median(formant_track(audio,sr,dict(ceiling=5500,window=.025),times),axis=0)),
            ("burg40",np.median(formant_track(audio,sr,dict(ceiling=5500,window=.04),times),axis=0))):
            target=np.exp(y[i]);valid=np.isfinite(target)
            errors=abs(estimate-target)/target
            values.append(dict(id=r["id"],speaker=r["speaker"],vowel=r["vowel"],split=r["split"],method=method,
                median_relative_error=float(np.median(errors[valid])),max_relative_error=float(np.max(errors[valid])),
                f1_error=float(errors[0]) if valid[0] else None,f2_error=float(errors[1]) if valid[1] else None,
                f3_error=float(errors[2]) if valid[2] else None,
                all_within10=bool(np.all(errors[valid]<=.1))))
    metrics={}
    for split in ("dev","test"):
        metrics[split]={}
        for method in ("neural","burg25","burg40"):
            rr=[r for r in values if r["split"]==split and r["method"]==method]
            metrics[split][method]=dict(n=len(rr),speakers=len({r["speaker"] for r in rr}),
                median_error=float(np.median([r["median_relative_error"] for r in rr])),
                p90_max_error=float(np.percentile([r["max_relative_error"] for r in rr],90)),
                all_three_within10_fraction=float(np.mean([r["all_within10"] for r in rr])))
    save_json(out/"metrics.json",dict(epoch=ck["epoch"],data_hash=ck["data_hash"],metrics=metrics))
    csv_write(out/"per_token.csv",values)
    lines=["# 小型物理预测头对照", "", "输入512维平均/标准差频谱，74,115参数；仅预测F1–F3。训练53人，开发11人早停选择，测试13人仅最终评估。",
           "标签为原始参考20–80%时刻的log频率中位数；固定Burg也使用相同时刻中位数，比较同一目标。", "", "|划分|方法|条数|相对误差中位|每条最大误差p90|三个参数均≤10%比例|", "|---|---|---:|---:|---:|---:|"]
    for split,mm in metrics.items():
        for method,m in mm.items():lines.append(f"|{split}|{method}|{m['n']}|{m['median_error']:.4f}|{m['p90_max_error']:.4f}|{m['all_three_within10_fraction']:.3f}|")
    lines += ["", "这个试验不测量跨语言/跨设备音色相似性；即使误差下降也不等于音色地图完成。测试集一经查看，不用于反复调结构后声称独立验证。"]
    (out/"report.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(metrics),flush=True)

if __name__=="__main__":main()
