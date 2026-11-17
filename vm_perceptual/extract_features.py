"""Frozen audio frontends. No identity or perceptual labels enter extraction."""
import argparse,json,math,time
from pathlib import Path
from runtime import ROOT,PROJECT,write_json,file_hash
import numpy as np
import torch
import soundfile as sf
import parselmouth as pm
from scipy.signal import resample_poly

REVISIONS={'ecapa':dict(repo='speechbrain/spkrec-ecapa-voxceleb',revision='0f99f2d0ebe89ac095bcc5903c4dd8f72b367286'),
           'wavlm':dict(repo='microsoft/wavlm-base-plus',revision='4c66d4806a428f2e922ccfa1a962776e232d487b')}

class Frontend:
    def __init__(self,models=None,device='cpu',use_wavlm=True,use_ecapa=True):
        import torchaudio
        models=Path(models or PROJECT/'claude_baselines/models').resolve()
        self.device=device
        torch.set_num_threads(2)
        local=str(models/'ecapa')
        self.ecapa=None
        if use_ecapa:
            from speechbrain.inference.speaker import EncoderClassifier
            from speechbrain.utils.fetching import LocalStrategy
            self.ecapa=EncoderClassifier.from_hparams(source=local,savedir=local,run_opts={'device':device},
                overrides={'pretrained_path':local},local_strategy=LocalStrategy.NO_LINK)
            self.ecapa.eval()
        self.wavlm=None
        if use_wavlm:
            from transformers import WavLMModel
            self.wavlm=WavLMModel.from_pretrained(str(models/'wavlm'),local_files_only=True).to(device).eval()
        self.mfcc=torchaudio.transforms.MFCC(sample_rate=16000,n_mfcc=20,
            melkwargs=dict(n_fft=512,win_length=400,hop_length=160,n_mels=40)).to(device)

    @torch.inference_mode()
    def extract(self,path):
        y,sr=sf.read(path,dtype='float32',always_2d=True)
        if y.shape[1]!=1:raise ValueError('Expected mono; choose a channel explicitly')
        y=y[:,0]
        if not np.isfinite(y).all() or len(y)/sr<.25:raise ValueError('Invalid or too short audio')
        if np.sqrt(np.mean(y*y))<1e-6:raise ValueError('Silent audio')
        if sr!=16000:
            g=math.gcd(sr,16000);y=resample_poly(y,16000//g,sr//g).astype('float32')
        x=torch.from_numpy(y.copy()).unsqueeze(0).to(self.device)
        m=self.mfcc(x)[0];e=m[0]
        keep=e>e.max()-.5*(e.max()-e.median())
        if int(keep.sum())<10:keep=torch.ones_like(e,dtype=torch.bool)
        mfcc=torch.cat([m[1:,keep].mean(1),m[1:,keep].std(1,unbiased=False)]).cpu().numpy()
        sound=pm.Sound(y,sampling_frequency=16000)
        pitch=sound.to_pitch_ac(time_step=.01,pitch_floor=60,pitch_ceiling=800)
        f=pitch.selected_array['frequency'];good=f>0
        h=sound.to_harmonicity_cc(time_step=.01,minimum_pitch=60).values.reshape(-1)
        h=h[np.isfinite(h)&(h>-100)]
        if good.sum()<3 or len(h)<3:raise ValueError('Insufficient periodic evidence')
        logf=np.log(f[good]);f0=float(np.median(logf));hnr=float(np.median(h))
        # Descriptive measurements, NOT a validated shared prosody embedding.
        frames=np.array([np.sqrt(np.mean(y[i:i+400]**2)+1e-12) for i in range(0,max(1,len(y)-400),160)])
        db=20*np.log10(frames+1e-12)
        prosody=[float((np.percentile(logf,90)-np.percentile(logf,10))*12/np.log(2)),
                 float(np.std(db)),float(good.mean()),float(np.mean(np.abs(np.diff(logf))))]
        out=dict(mfcc=mfcc.tolist(),acoustic=np.r_[mfcc,f0,hnr].tolist(),
            prosody_descriptors=prosody,duration=len(y)/16000,voiced_frames=int(good.sum()))
        if self.ecapa is not None:out['ecapa']=self.ecapa.encode_batch(x).reshape(-1).cpu().numpy().tolist()
        if self.wavlm is not None:
            out['wavlm3']=self.wavlm(x,output_hidden_states=True).hidden_states[3][0].mean(0).cpu().numpy().tolist()
        if any(not np.isfinite(out[k]).all() for k in ['mfcc','acoustic','prosody_descriptors']+(['ecapa'] if self.ecapa is not None else [])):raise ValueError('Nonfinite features')
        return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--models',default=str(PROJECT/'claude_baselines/models'))
    p.add_argument('--device',default='cpu');p.add_argument('--no-wavlm',action='store_true');a=p.parse_args()
    data=ROOT/'data_jvs';manifest=json.loads((data/'manifest.json').read_text())
    out=ROOT/'features/jvs_features.json';cache=json.loads(out.read_text()) if out.exists() else {}
    signature=file_hash(__file__);frontend=None;t0=time.time()
    rows=[r for r in manifest['files'] if r['path'].endswith('.wav')]
    for i,r in enumerate(rows):
        path=r['path'];existing=cache.get(path,{})
        if existing.get('sha256')==r['sha256'] and existing.get('extractor_hash')==signature and ('wavlm3' in existing or a.no_wavlm):continue
        if file_hash(data/path)!=r['sha256']:raise ValueError('Audio provenance mismatch')
        if frontend is None:frontend=Frontend(a.models,a.device,not a.no_wavlm)
        values=frontend.extract(data/path)
        cache[path]=dict(**values,sha256=r['sha256'],extractor_hash=signature,model_revisions=REVISIONS)
        if i%10==0:
            write_json(out,cache);print('features',i+1,'/',len(rows),'seconds',round(time.time()-t0),flush=True)
    write_json(out,cache);print('DONE features',len(cache),flush=True)

if __name__=='__main__':main()
