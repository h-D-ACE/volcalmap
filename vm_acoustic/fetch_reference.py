"""Bounded downloads for an independently annotated vowel validation subset."""
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"data_reference"
REPO="santiagobarreda/hillenbrand_et_al_1995"
COMMIT="7fee5b548ad3898b1d0be3755253f1cb33ad083d"
ARCHIVE_SHA256="2560548591e3a726c88549b6dc9d226995616c26874d601c69fee8b9eee9d730"

def get(url,limit):
    with urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"VM-Acoustic-research"}),timeout=60) as r:
        b=r.read(limit+1)
    if len(b)>limit: raise ValueError("Download budget exceeded")
    return b

def main():
    OUT.mkdir(exist_ok=True)
    commit=COMMIT
    prefix=f"https://raw.githubusercontent.com/{REPO}/{commit}/"
    for name in ("README.md","LICENSE"):
        (OUT/name).write_bytes(get(prefix+name,100000))
    path=OUT/"h95-alldata.zip"
    if not path.exists(): path.write_bytes(get(prefix+path.name,40000000))
    if hashlib.sha256(path.read_bytes()).hexdigest()!=ARCHIVE_SHA256:raise ValueError("Pinned archive checksum mismatch")
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        print(names,flush=True)
        for name in names:
            if name.lower().endswith(("timedata.dat","vowdata.dat","bigdata.dat","readme.txt")):
                (OUT/Path(name).name.lower()).write_bytes(z.read(name))
    manifest=dict(repository=f"https://github.com/{REPO}",commit=commit,
        archive_url=prefix+path.name,archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        citation="Hillenbrand, Getty, Clark & Wheeler (1995), doi:10.1121/1.411872",
        usage="Local annotated measurement validation; not a paired-microphone corpus")
    (OUT/"source.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")

if __name__=="__main__": main()
