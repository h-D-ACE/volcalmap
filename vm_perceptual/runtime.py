"""Optional local dependency paths; ordinary site-packages work on cloud."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
for p in (PROJECT/'claude_baselines/.deps',PROJECT/'vm_acoustic/.deps'):
    if p.is_dir():sys.path.insert(0,str(p))

def write_json(path,obj):
    import json
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def file_hash(path):
    import hashlib
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
