"""Fetch pinned frozen frontends to a user-selected local directory."""
import argparse
from runtime import ROOT,write_json
from extract_features import REVISIONS

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'models'));a=p.parse_args()
    from pathlib import Path
    from huggingface_hub import snapshot_download
    for key,info in REVISIONS.items():
        patterns=['*.json','*.ckpt','*.yaml','*.txt','README.md'] if key=='ecapa' else ['config.json','preprocessor_config.json','pytorch_model.bin']
        snapshot_download(repo_id=info['repo'],revision=info['revision'],local_dir=str(Path(a.output)/key),allow_patterns=patterns)
    write_json(Path(a.output)/'model_revisions.json',REVISIONS)

if __name__=='__main__':main()
