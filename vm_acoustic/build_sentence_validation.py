"""New-speaker paired validation, frozen after reference development selection."""
import hashlib
import json
import re
from collections import defaultdict,Counter
from math import gcd
from pathlib import Path

from acoustics import ROOT,np,read_audio,extract,fit_calibration,coordinates
from scipy.signal import resample_poly
from pocketsphinx import Decoder
from measurement_v2 import Measurement
from reliability_probe import lag_seconds
from run_probe import save_json,csv_write,summary

DATA=ROOT/"data_vctk_validation"
OUT=ROOT/"results_sentence_validation"
VOWELS={"AE","EH","IY","AA","AH","UW"}
CAL={"p233","p236","p237","p241"}

def align(y,sr,text):
    g=gcd(sr,16000); y=resample_poly(y,16000//g,sr//g)
    pcm=(np.clip(y,-1,1)*32767).astype("<i2").tobytes()
    d=Decoder(samprate=16000,loglevel="ERROR")
    d.set_align_text(" ".join(re.findall("[a-z']+",text.lower())))
    for phonepass in (False,True):
        if phonepass:d.set_alignment()
        d.start_utt();d.process_raw(pcm,full_utt=True);d.end_utt()
    pp=[];wi=0
    for word in d.get_alignment():
        if word.name.startswith("<"):continue
        for pi,p in enumerate(word):
            label=re.sub(r"\d","",p.name)
            if label in VOWELS:pp.append(dict(word_index=wi,phone_index=pi,word=word.name,vowel=label,start=p.start/100,end=(p.start+p.duration)/100))
        wi+=1
    return pp

def group(s,u):return ("cal" if s in CAL else "eval")+("_old_text" if int(u)<=6 else "_new_text")

def main():
    OUT.mkdir(exist_ok=True);(OUT/"cache").mkdir(exist_ok=True)
    manifest=json.loads((DATA/"manifest.json").read_text())
    method=json.loads((ROOT/"results_reference/locked_method.json").read_text())
    source_hash=hashlib.sha256((ROOT/"measurement_v2.py").read_bytes()+(Path(__file__).read_bytes())).hexdigest()
    signature=hashlib.sha256((source_hash+json.dumps(method)).encode()).hexdigest()
    pairs=[];failures=[]
    for s in manifest["speakers"]:
        for u in manifest["utterances"]:
            paths=[DATA/f"{s}_{u}_{m}.flac" for m in ("mic1","mic2")]
            txt=DATA/f"{s}_{u}.txt"
            if not all(p.exists() for p in [*paths,txt]):
                failures.append(dict(speaker=s,utterance=u,reason="source_missing"));continue
            cache=OUT/"cache"/f"{s}_{u}.json"
            if cache.exists():
                saved=json.loads(cache.read_text())
                if saved.get("signature")==signature:
                    pairs.append(saved);continue
            a,sr=read_audio(paths[0]);b,srb=read_audio(paths[1]);assert sr==srb
            text=txt.read_text().strip()
            alignment_cache=OUT/"cache"/f"{s}_{u}_alignment.json"
            if alignment_cache.exists():
                pp=json.loads(alignment_cache.read_text())
            else:
                try:pp=align(a,sr,text)
                except Exception as e:
                    failures.append(dict(speaker=s,utterance=u,reason="alignment_failure",error=str(e)));continue
                save_json(alignment_cache,pp)
            lag,corr=lag_seconds(a,b,sr)
            models={}
            for m,c in (("baseline",dict(ceiling=5500,window=.025)),("reference_selected",method["config"])):
                models[m]=(Measurement(a,sr,**c),Measurement(b,sr,**c))
            tokens=[]
            for p in pp:
                token=dict(p)
                for name,(ma,mb) in models.items():
                    token[name]=dict(a=ma.token(p["start"],p["end"]),b=mb.token(p["start"]+lag,p["end"]+lag))
                tokens.append(token)
            whole={}
            for mic,y in (("mic1",a),("mic2",b)):
                try:whole[mic]=extract(y,sr)["values"]
                except Exception as e:whole[mic]=None
            saved=dict(signature=signature,speaker=s,utterance=u,group=group(s,u),text=text,
                lag=lag,correlation=corr,tokens=tokens,whole=whole,
                audio_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths])
            save_json(cache,saved);pairs.append(saved)
            print(s,u,"tokens",len(tokens),flush=True)
    save_json(OUT/"measurements.json",dict(pairs=pairs,failures=failures,method=method,signature=signature))
    report(pairs,failures,method)

def report(pairs,failures,method):
    # Fit common scale from NEW calibration speakers/old texts, never evaluation.
    cal=[(p["whole"]["mic1"],p["whole"]["mic2"]) for p in pairs if p["speaker"] in CAL and int(p["utterance"])<=6 and all(p["whole"].values())]
    model=fit_calibration(cal)
    model["fit_keys"]=[[p["speaker"],p["utterance"]] for p in pairs if p["speaker"] in CAL and int(p["utterance"])<=6 and all(p["whole"].values())]
    scale=np.array(model["scale"])
    rows=[];coverage=defaultdict(Counter)
    corrections={}
    for vowel in VOWELS:
        pp=[(t["reference_selected"]["a"]["values"],t["reference_selected"]["b"]["values"]) for p in pairs if p["speaker"] in CAL and int(p["utterance"])<=6 for t in p["tokens"] if t["vowel"]==vowel and all(t["reference_selected"]["a"]["mask"]+t["reference_selected"]["b"]["mask"])]
        if len(pp)>=4:corrections[vowel]=np.median(np.array([b for a,b in pp])-np.array([a for a,b in pp]),axis=0).tolist()
    save_json(OUT/"calibration.json",dict(scale=model,vowel_bias=corrections))
    for p in pairs:
        for i,t in enumerate(p["tokens"]):
            g=p["group"];c=coverage[g];c["all_aligned_candidates"]+=1
            long=t["end"]-t["start"]>=.18
            c["long_candidates"]+=int(long)
            r=dict(speaker=p["speaker"],utterance=p["utterance"],group=g,index=i,vowel=t["vowel"],word=t["word"],duration=t["end"]-t["start"],long=long)
            complete=True
            for name in ("baseline","reference_selected"):
                a,b=t[name]["a"],t[name]["b"]
                valid=np.array(a["mask"])&np.array(b["mask"])
                c[name+"_complete"]+=int(valid.all())
                if long:c[name+"_long_complete"]+=int(valid.all())
                for j in range(6):c[name+f"_feature{j}_available"]+=int(valid[j])
                if not valid.all():complete=False;r[name]=None;continue
                delta=np.array(b["values"])-a["values"]
                r[name]=float(np.linalg.norm(delta/scale)/np.sqrt(6))
                if name=="reference_selected" and t["vowel"] in corrections:
                    r["bias"]=float(np.linalg.norm((delta-corrections[t["vowel"]])/scale)/np.sqrt(6))
            if complete:
                c["common_complete"]+=1
                if long:c["long_common_complete"]+=1
                a,b=p["whole"]["mic1"],p["whole"]["mic2"]
                r["whole_raw"]=float(np.linalg.norm((np.array(b)-a)/scale)/np.sqrt(6)) if a and b else None
                rows.append(r)
    # one sentence is one replication; report all candidates + long stratum
    metrics={}
    for stratum in ("all","long"):
        metrics[stratum]={}
        for g in sorted(coverage):
            buckets=defaultdict(list)
            for r in rows:
                if r["group"]==g and (stratum=="all" or r["long"]):buckets[r["speaker"],r["utterance"]].append(r)
            metrics[stratum][g]={}
            for k in ("baseline","reference_selected","bias","whole_raw"):
                # Keep bias denominator explicit.
                vals=[float(np.median([r[k] for r in rr])) for rr in buckets.values() if all(r.get(k) is not None for r in rr)]
                metrics[stratum][g][k]=summary(vals)
    save_json(OUT/"metrics.json",dict(coverage={g:dict(c) for g,c in coverage.items()},metrics=metrics,failures=failures))
    # CSV requires union of keys when bias missing.
    for r in rows:r.setdefault("bias",None)
    csv_write(OUT/"distances.csv",rows)
    lines=["# 新说话者自然载句双麦验证", "", "8位预选新说话者，每人预定001–012；4人标定/4人验证、各2F/2M。001–006标定句，007–012新句。",
        "这些是自然朗读句内的元音载体，非专门录制的持续元音载句。只对mic1强制对齐，通过波形时差投影到mic2；不是两套独立人工边界。",
        f"共完成{len(pairs)}句对；缺失/对齐失败{len(failures)}，详情见metrics.json。",
        f"新配置由独立Hillenbrand开发集选定：{method['chosen']}。双麦验证结果不用于选窗长/上限。",
        "不使用边界扰动阈值、轨迹IQR阈值或双麦距离作剔除条件；各参数独立缺失，完整距离只用于六参数都可用的共同子集。",
        "", "|条件|全部候选|完整六参数共同覆盖|长元音候选≥180ms|长元音完整覆盖|", "|---|---:|---:|---:|---:|"]
    for g,c in sorted(coverage.items()):lines.append(f"|{g}|{c['all_aligned_candidates']}|{c['common_complete']}|{c['long_candidates']}|{c['long_common_complete']}|")
    for stratum,groups in metrics.items():
        lines += ["",f"## {stratum}：先每句token中位，再跨句中位，公共尺度", "", "|条件|句对|25ms基线|参考集选定40ms|逐元音bias（句对）|整句raw|", "|---|---:|---:|---:|---:|---:|"]
        for g,mm in groups.items():
            fmt=lambda k:"NA" if mm[k]["median"] is None else f"{mm[k]['median']:.4f}"
            lines.append(f"|{g}|{mm['baseline']['n']}|{fmt('baseline')}|{fmt('reference_selected')}|{fmt('bias')} ({mm['bias']['n']})|{fmt('whole_raw')}|")
    lines += ["", "覆盖率提升不等于准确性；Hillenbrand参考误差和此处通道重复性必须同时阅读。",
              "不同方法共同完整token集合一致，但bias或整句缺失时分母可能不同。没有独立录音日/未知设备/中日文/语气真值，因此尚非通用音色地图验收。"]
    (OUT/"report.md").write_text("\n".join(lines),encoding="utf-8")

if __name__=="__main__":main()
