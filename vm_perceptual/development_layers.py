"""Layer-wise frozen speech features on the twelve development-protocol texts."""
from runtime import ROOT
from extract_features import Frontend, resample_poly
from transformers import WavLMModel
import numpy as np
import soundfile as sf
import torch,json
from math import gcd


def main():
    manifest=json.loads((ROOT/'data_jvs/manifest.json').read_text())
    names=sorted(s for s,g in manifest['split'].items() if g in ('train','dev'))
    front=Frontend(ROOT.parent/'claude_baselines/models','cuda',False,True)
    model=WavLMModel.from_pretrained(str(ROOT.parent/'claude_baselines/models/wavlm'),local_files_only=True).eval().cuda()
    grouped=[]; acoustic=[]; ecapa=[]
    with torch.inference_mode():
        for s in names:
            per=[];ac=[];ec=[]
            files=sorted((ROOT/'data_jvs'/s/'parallel100/wav24kHz16bit').glob('*.wav'))[:12]
            if len(files)!=12:raise ValueError(s)
            for path in files:
                x,sr=sf.read(path,dtype='float32')
                g=gcd(sr,16000)
                x=resample_poly(x,16000//g,sr//g) if sr!=16000 else x
                layers=model(torch.from_numpy(x).cuda()[None],output_hidden_states=True).hidden_states
                per.append(torch.stack([h[0].mean(0) for h in layers]).cpu().numpy())
                out=front.extract(path)
                ac.append(out['acoustic']);ec.append(out['ecapa'])
            grouped.append(per);acoustic.append(ac);ecapa.append(ec)
            print(s,flush=True)
    np.savez_compressed(ROOT/'features/development_layers.npz',speakers=names,layers=np.array(grouped),acoustic=np.array(acoustic),ecapa=np.array(ecapa))


if __name__=='__main__':
    main()
