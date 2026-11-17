"""Bounded exact-Range extraction of a fixed JVS research subset, CRC checked."""
import json,urllib.request,struct,zlib,hashlib,random,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parent
URL='https://drive.usercontent.google.com/download?id=19oAw8wWn3Y7z6CKChRdAyGOB9yupL_Xt&export=download&confirm=t'

def read_range(start,end):
    for attempt in range(3):
        try:
            req=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity','User-Agent':'VocalMap-research'})
            with urllib.request.urlopen(req,timeout=40) as r:
                if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/3536595425':raise RuntimeError('Range not honored')
                b=r.read(end-start+2)
            if len(b)!=end-start+1:raise RuntimeError('Bad range size')
            return b
        except OSError:
            if attempt==2:raise
            time.sleep(1)

def extract(info):
    dest=ROOT/'data_jvs'/Path(info['name']).relative_to('jvs_ver1')
    if dest.exists() and zlib.crc32(dest.read_bytes())&0xffffffff==info['crc']:return dest,0
    # ZIP local header extra is small; bounded 1024-byte margin lets one request fetch each member.
    n=30+len(info['name'].encode())+1024+info['compressed']
    b=read_range(info['offset'],info['offset']+n-1)
    if b[:4]!=b'PK\x03\x04':raise ValueError('Invalid member header')
    fn,ex=struct.unpack_from('<HH',b,26);start=30+fn+ex
    if start+info['compressed']>len(b):raise ValueError('Unexpected large extra header')
    raw=b[start:start+info['compressed']]
    data=zlib.decompress(raw,-15) if info['method']==8 else raw
    if len(data)!=info['size'] or zlib.crc32(data)&0xffffffff!=info['crc']:raise ValueError('CRC mismatch')
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    return dest,len(b)

def main():
    source=ROOT/'data_sources';index=json.loads((source/'jvs_zip_index.json').read_text())
    genders={};rng=random.Random(20261117);split={}
    for gender in ('male','female'):
        ratings=json.loads((source/f'jvs_{gender}_raw_similarity_data.json').read_text())
        people=sorted({s for k in ratings for s in k.split('-')});rng.shuffle(people)
        for i,s in enumerate(people):genders[s]=gender;split[s]='test' if i<10 else 'dev' if i<20 else 'train'
    selected=[i for i in index if i['name'] in ['jvs_ver1/README.txt','jvs_ver1/gender_f0range.txt','jvs_ver1/speaker_similarity_male.csv','jvs_ver1/speaker_similarity_female.csv']]
    for s in sorted(split):
        for style,n in [('parallel100',4),('falset10',2 if split[s]=='test' else 0)]:
            matches=sorted([i for i in index if i['name'].startswith(f'jvs_ver1/{s}/{style}/wav24kHz16bit/') and i['name'].endswith('.wav')],key=lambda i:i['name'])
            if len(matches)<n:raise ValueError(f'Missing predetermined sample {s} {style}')
            selected.extend(matches[:n])
    budget=sum(i['compressed']+len(i['name'].encode())+1054 for i in selected)
    if budget>250_000_000:raise ValueError('Subset exceeds 250MB budget')
    (source/'selection.json').write_text(json.dumps(dict(split=split,gender=genders,files=selected,budget=budget),indent=2))
    print('fixed files',len(selected),'expected MB',round(budget/1e6,2),flush=True)
    records=[];received=0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,(p,n) in enumerate(pool.map(extract,selected)):
            received+=n;records.append(dict(path=p.relative_to(ROOT/'data_jvs').as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
            if i%25==0:print(i+1,'/',len(selected),'MB',round(received/1e6,2),flush=True)
    (ROOT/'data_jvs/manifest.json').write_text(json.dumps(dict(source=URL,split=split,gender=genders,files=records,received=received),indent=2))
    print('DONE',len(records),received,flush=True)

if __name__=='__main__':main()
