"""Research waveform-student map. Build a separate index or locate one mono recording."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
from map_index import search
from runtime import ROOT,file_hash,write_json
from wave_student import WaveMap

POLICY='mono-float32-polyphase16k-whole-singleton-v1'


class WaveEncoder:
    def __init__(self, checkpoint, device, variant='student'):
        self.device=torch.device(device)
        state=torch.load(checkpoint,weights_only=True,map_location='cpu')
        components = [file_hash(checkpoint), POLICY]
        if variant == 'student':
            self.model = WaveMap()
            self.model.load_state_dict(state['state'], strict=True)
        elif variant in ('robust_student', 'unaugmented_student'):
            from ssl_rank_student import SSLMap
            self.model = SSLMap('frozen')
            learned = {k: p for k, p in self.model.named_parameters() if p.requires_grad}
            recipe = 'consistency' if variant == 'robust_student' else 'unaugmented'
            if set(learned) != set(state['state']) or state.get('recipe') != recipe:
                raise ValueError(f'Expected a {recipe} head for the frozen WavLM backbone')
            with torch.no_grad():
                for k, p in learned.items():
                    p.copy_(state['state'][k])
            model_dir = ROOT.parent / 'claude_baselines/models/wavlm'
            weights = model_dir / 'model.safetensors'
            if not weights.exists():
                weights = model_dir / 'pytorch_model.bin'
            components.extend([variant, 'fp32', file_hash(model_dir/'config.json'), file_hash(weights)])
        else:
            raise ValueError(f'Unknown waveform encoder: {variant}')
        self.model.eval().to(device)
        if self.device.type=='cuda':
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        self.version=hashlib.sha256('/'.join(components).encode()).hexdigest()

    @torch.inference_mode()
    def vector(self,path):
        wave,sr=sf.read(path,always_2d=True,dtype='float32')
        if wave.shape[1]!=1:raise ValueError('Select a single voice channel before encoding')
        x=wave[:,0]
        if len(x)<sr or not np.isfinite(x).all():raise ValueError('Expected at least one second of finite mono audio')
        if np.sqrt(np.mean(x.astype('float64')**2))<1e-6:raise ValueError('Silent audio')
        g=math.gcd(sr,16000)
        x=resample_poly(x,16000//g,sr//g).astype('float32')
        t=torch.from_numpy(np.ascontiguousarray(x)).to(self.device)[None]
        n=torch.tensor([len(x)],device=self.device)
        return self.model(t,n)[0].cpu().numpy()


def build(encoder,root):
    man=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    records=[];training=[]
    for row in man['files']:
        path=row['path']
        if not path.endswith('.wav'):continue
        wave=root/path
        if file_hash(wave)!=row['sha256']:raise ValueError(f'Unexpected recording contents: {path}')
        z=encoder.vector(wave)
        records.append(dict(record_id=path,sample_id=row['sha256'],vector=z.tolist()))
        if man['split'][path.split('/')[0]]=='train' and '/parallel100/' in path:training.append(z)
    if len(training)<4:raise ValueError('Insufficient training reference clips for display axes')
    training=np.asarray(training,dtype='float64');mu=training.mean(0)
    _,_,vt=np.linalg.svd(training-mu,full_matrices=False)
    axes=vt[:3].copy()
    for axis in axes:
        if axis[np.argmax(abs(axis))]<0:axis*=-1
    view=dict(mean=mu.tolist(),axes=axes.tolist())
    digest=hashlib.sha256(json.dumps(view,sort_keys=True).encode()).hexdigest()
    for r in records:r['display']=((np.asarray(r['vector'])-mu)@axes.T).tolist()
    return dict(model_version=encoder.version,projection_version=digest,dimensions=16,
        display=view,records=records,research_only=True,preprocessing=POLICY)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--variant', choices=('student','robust_student'), default='student')
    ap.add_argument('--checkpoint', type=Path)
    ap.add_argument('--device',default='cpu')
    sub=ap.add_subparsers(dest='command',required=True)
    b=sub.add_parser('build');b.add_argument('--data',type=Path,default=ROOT/'data_jvs')
    b.add_argument('--output',type=Path)
    q=sub.add_parser('query');q.add_argument('audio',type=Path)
    q.add_argument('--index',type=Path)
    q.add_argument('--display-dimensions',type=int,choices=(1,2,3),default=2)
    q.add_argument('--neighbors',type=int,default=5);q.add_argument('--output',type=Path)
    a=ap.parse_args()
    robust = a.variant == 'robust_student'
    prefix = 'robust_student' if robust else 'wave'
    run = 'robust_ssl_consistency_177_1800' if robust else 'wave_431_noise'
    encoder = WaveEncoder(a.checkpoint or ROOT/'runs'/run/'best.pt', a.device, a.variant)
    default_index = ROOT/'results'/f'{prefix}_index.json'
    if a.command == 'build':
        a.output = a.output or default_index
    else:
        a.index = a.index or default_index
        a.output = a.output or ROOT/'results'/f'{prefix}_query.json'
    if a.command=='build':
        result=build(encoder,a.data)
        print(f"Built {len(result['records'])} recording coordinates")
    else:
        index=json.loads(a.index.read_text(encoding='utf-8'))
        if index['model_version']!=encoder.version:raise ValueError('Index and query use different map versions')
        z=encoder.vector(a.audio)
        axes=np.asarray(index['display']['axes'])[:a.display_dimensions]
        coords=(z-index['display']['mean'])@axes.T
        result=dict(sample_id=file_hash(a.audio),model_version=encoder.version,timbre={'vector':z.tolist()},
            display={'coordinates':coords.tolist(),'version':index['projection_version']},research_only=True)
        result['neighbors']=search(index,result,k=a.neighbors)['matches']
        print(result['display'],result['neighbors'][0]['record_id'] if result['neighbors'] else 'empty gallery')
    write_json(a.output,result)


if __name__=='__main__':main()
