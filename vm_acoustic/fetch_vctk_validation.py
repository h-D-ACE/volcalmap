"""Fresh fixed VCTK speakers/texts; no acoustic-quality-based selection."""
import hashlib
import json
import zipfile
import argparse
from fetch_samples import ROOT, RangeFile,API,ITEM,get

SPEAKERS=["p233","p236","p239","p240","p237","p241","p243","p245"]
UTTS=[f"{i:03d}" for i in range(1,13)]

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--confirmation",action="store_true");args=parser.parse_args()
    speakers=["p244","p250","p252","p254"] if args.confirmation else SPEAKERS
    utterances=["001","003","005","008","011","012"] if args.confirmation else UTTS
    out=ROOT/("data_vctk_confirmation" if args.confirmation else "data_vctk_validation")
    out.mkdir(exist_ok=True)
    if (ROOT/"data/manifest.json").exists():
        archive=json.loads((ROOT/"data/manifest.json").read_text())["archive"]
    else:
        bundles=json.loads(get(f"{API}/items/{ITEM}/bundles"))["_embedded"]["bundles"]
        bundle=next(b for b in bundles if b["name"]=="ORIGINAL")
        streams=json.loads(get(bundle["_links"]["bitstreams"]["href"]))["_embedded"]["bitstreams"]
        archive=next(s for s in streams if s["name"].endswith(".zip"))
        for s in streams:
            if s["name"] in ("README.txt","license_text.txt"):
                (out/s["name"]).write_bytes(get(s["_links"]["content"]["href"]))
    remote=RangeFile(archive["_links"]["content"]["href"],archive["sizeBytes"])
    wanted={f"{s}_{u}{suffix}" for s in speakers for u in utterances for suffix in (".txt","_mic1.flac","_mic2.flac")}
    rows=[]
    with zipfile.ZipFile(remote) as z:
        selected=[n for n in z.namelist() if n.rsplit("/",1)[-1] in wanted]
        print("selected",len(selected),flush=True)
        for i,n in enumerate(selected):
            info=z.getinfo(n)
            target=out/n.rsplit("/",1)[-1]
            if target.exists():
                import zlib
                b=target.read_bytes()
                if len(b)!=info.file_size or zlib.crc32(b)&0xffffffff!=info.CRC:
                    raise ValueError("Existing file checksum failed")
            else:
                if info.file_size>5000000: raise ValueError("Unexpected member size")
                b=z.read(n)
                target.write_bytes(b)
            rows.append(dict(file=target.name,archive_path=n,sha256=hashlib.sha256(b).hexdigest(),zip_crc=info.CRC))
            if i%12==0: print(i+1,"/",len(selected),"MB",round(remote.bytes_received/1e6,2),flush=True)
    manifest=dict(source="VCTK 0.92",source_url="https://datashare.ed.ac.uk/handle/10283/3443",
        archive_id=archive["id"],license="CC BY 4.0 (official README)",
        speakers=speakers,utterances=utterances,files=rows,range_bytes=remote.bytes_received,
        split={s:("calibration" if s in ("p233","p236","p237","p241") else "evaluation") for s in speakers},
        note="All preselected utterances, no outcome-based quality selection; evaluation 2F/2M")
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print("DONE",len(rows),flush=True)

if __name__=="__main__": main()
