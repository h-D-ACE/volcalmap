"""Cache aligned shallow acoustic and deep contextual WavLM frames, without transcripts."""
import math
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import WavLMModel
from runtime import ROOT
from dataset import load_data


@torch.inference_mode()
def main():
    data=load_data()
    net=WavLMModel.from_pretrained(str(ROOT.parent/'claude_baselines/models/wavlm'),local_files_only=True).cuda().eval()
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    # Shared data-independent orthogonal sketches preserve a common frame coordinate system.
    rng=np.random.default_rng(40821)
    sketch=[torch.from_numpy(np.linalg.qr(rng.standard_normal((768,128)))[0].astype('float32')).cuda() for _ in range(2)]
    out=ROOT/'features/content_frames';out.mkdir(exist_ok=True)
    for name in data['speakers']:
        dest=out/f'{name}.npz'
        if dest.exists():continue
        files=sorted((ROOT/'data_jvs'/name/'parallel100/wav24kHz16bit').glob('*.wav'))
        lengths=[];shallow=[];deep=[]
        for file in files:
            x,sr=sf.read(file,dtype='float32');g=math.gcd(sr,16000)
            x=resample_poly(x,16000//g,sr//g).astype('float32')
            inp=torch.from_numpy(x).cuda()[None]
            h=net(inp,output_hidden_states=True).hidden_states
            low=(h[1][0]@sketch[0]).cpu().numpy()
            high=(h[9][0]@sketch[1]).cpu().numpy()
            if len(low)!=len(high):raise ValueError('Unaligned WavLM layer frames')
            shallow.append(low.astype(np.float16));deep.append(high.astype(np.float16));lengths.append(len(low))
        stage=dest.with_suffix('.partial.npz')
        np.savez_compressed(stage,names=[p.name for p in files],lengths=lengths,shallow=np.concatenate(shallow),deep=np.concatenate(deep))
        stage.replace(dest);print(name,flush=True)


if __name__=='__main__':main()
