"""Build or query an experimental ordinal 16-D timbre map without mixing model versions."""
import argparse
import json
from pathlib import Path
from runtime import ROOT,file_hash,write_json
from rank_map_candidate import Frontends,encode
from map_index import search
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',choices=('own','combined'),default='own')
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--artifact',type=Path)
    sub=p.add_subparsers(dest='mode',required=True)
    b=sub.add_parser('build');b.add_argument('--output',type=Path);b.add_argument('--data',type=Path,default=ROOT/'data_jvs')
    q=sub.add_parser('query');q.add_argument('audio',type=Path);q.add_argument('--index',type=Path)
    q.add_argument('--output',type=Path);q.add_argument('--display-dimensions',choices=(1,2,3),type=int,default=2)
    args=p.parse_args()
    art=args.artifact or ROOT/'runs'/f'rank_map_{args.variant}.npz'
    version=file_hash(art)
    if args.variant=='combined' and args.device!='cuda':raise ValueError('The optional VoxSim implementation uses CUDA')
    default_index=ROOT/'results'/f'rank_{args.variant}_index.json'
    encoder=Frontends(args.variant=='combined',args.device)
    with np.load(art) as params:
        expected=['acoustic','identity','wavlm1']+(['voxsim'] if args.variant=='combined' else [])
        if params['keys'].tolist()!=expected:raise ValueError('Artifact frontend configuration mismatch')
        def point(audio):
            z=encode(encoder.extract(audio),params)
            return z,((z-params['view_mean'])@params['view_axes'][:3].T)
        if args.mode=='build':
            manifest=json.loads((args.data/'manifest.json').read_text(encoding='utf-8'))
            rows=[]
            for item in manifest['files']:
                rel=item['path']
                if not rel.endswith('.wav'):continue
                path=args.data/rel
                if file_hash(path)!=item['sha256']:raise ValueError(f'Audio hash mismatch: {rel}')
                z,display=point(path)
                rows.append(dict(record_id=rel,sample_id=item['sha256'],vector=z.tolist(),display=display.tolist()))
            result=dict(model_version=version,projection_version=version,dimensions=16,research_only=True,records=rows)
            output=args.output or default_index
        else:
            index=json.loads((args.index or default_index).read_text(encoding='utf-8'))
            if index['model_version']!=version:raise ValueError('Map version differs from the query encoder')
            z,xyz=point(args.audio)
            result=dict(model_version=version,sample_id=file_hash(args.audio),timbre=dict(vector=z.tolist()),
                display=dict(coordinates=xyz[:args.display_dimensions].tolist(),projection_version=version),research_only=True)
            result['neighbors']=search(index,result)['matches']
            output=args.output or ROOT/'results'/f'rank_{args.variant}_query.json'
        write_json(output,result)
        print(output,'records',len(result.get('records',[])),'neighbors',len(result.get('neighbors',[])))


if __name__=='__main__':main()
