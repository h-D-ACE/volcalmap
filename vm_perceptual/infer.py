"""Audio -> research perceptual coordinates. Does NOT invent tone normalization."""
import argparse,json
from pathlib import Path
from runtime import ROOT,PROJECT,write_json,file_hash
from extract_features import Frontend
from dataset import transform,unit
from train_map import MetricHead
import numpy as np
import torch

def encode(features,ck):
    if ck['kind']=='baseline':
        x=transform(features,ck['preprocessing'],pca=ck['pca_dim'] is not None)
        return unit(x if ck['pca_dim'] is None else x[:ck['pca_dim']])
    model=MetricHead(ck['input_dim'],ck['output_dim']);model.load_state_dict(ck['model']);model.eval()
    with torch.no_grad():return model(torch.tensor(transform(features,ck['preprocessing'],pca=True),dtype=torch.float32)).numpy()

def main():
    p=argparse.ArgumentParser();p.add_argument('audio');p.add_argument('--checkpoint',default=str(ROOT/'runs/selected.pt'))
    p.add_argument('--models',default=str(PROJECT/'claude_baselines/models'));p.add_argument('--device',default='cpu')
    p.add_argument('--projection',default=str(ROOT/'results/projection.json'));p.add_argument('--display-dimensions',type=int,choices=[1,2,3],default=2)
    p.add_argument('--output',default=str(ROOT/'results/query.json'));a=p.parse_args()
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    frontend=Frontend(a.models,a.device,ck['representation']=='wavlm3',ck['representation'] in ('ecapa','fusion'),edge_policy=ck.get('edge_policy','preserve'))
    feature=frontend.extract(a.audio);z=encode(feature,ck)
    view=json.loads(Path(a.projection).read_text())
    if view.get('checkpoint_sha256')!=file_hash(a.checkpoint):raise ValueError('Projection belongs to a different checkpoint')
    dim=min(a.display_dimensions,len(view['axes']))
    display=((z-view['mean'])@np.array(view['axes'][:dim]).T).tolist()
    result=dict(sample_id=file_hash(a.audio),model_version=file_hash(a.checkpoint),research_only=True,
        timbre=dict(vector=z.tolist(),mode='observed',status='experimental',uncertainty=None,
            limitation='No calibrated confidence interval for an arbitrary query; validated only on JVS normal-voice pair ratings'),
        prosody=dict(vector=feature['prosody_descriptors'],kind='untrained acoustic descriptors',
            axes=['pitch_90_10_semitones','energy_std_db','voiced_fraction','mean_voiced_log_pitch_step'],status='descriptive_only'),
        display=dict(dimensions=dim,coordinates=display,projection_version=file_hash(a.projection),
            reference_distortion=view['evaluation']['views'].get(str(dim)),retrieval_uses_full_vector=True),
        tone_normalization=dict(status='not_implemented',reason='requires independently validated same-voice conditional model'),
        identity_links=dict(status='not_implemented'),duration=feature['duration'])
    write_json(a.output,result);print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
