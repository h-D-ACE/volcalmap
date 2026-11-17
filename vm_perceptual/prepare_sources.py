"""Reconstruct source metadata without shipping corpus files in the code archive."""
import json,hashlib,io,zipfile,urllib.request
from runtime import ROOT,write_json
from fetch_jvs import read_range

COMMIT='d709591d262c8c80fe4079ebf87f49c32ce61b10'
EXPECTED={'README.md':'7a9a1c32eb0adcdcc362305c607905f5be7fc439b2d8dc7a752d54e82f0f853b',
    'jvs_female_raw_similarity_data.json':'7dd81b62afb679e2def6f34d3efe3a8169ec91f11228675f1e446235cf230740',
    'jvs_male_raw_similarity_data.json':'42fe86717721d03a869b7b5ae29aa0fe2a1dd64842eca2f24243114cfdb6e430'}

class RemoteZip(io.RawIOBase):
    def __init__(self):self.pos=0;self.size=3536595425;self.received=0
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,n,whence=0):
        self.pos=n+(0 if whence==0 else self.pos if whence==1 else self.size)
        if self.pos<0:raise ValueError('Negative seek')
        return self.pos
    def read(self,n=-1):
        n=min(self.size-self.pos,n if n>=0 else self.size-self.pos)
        if n<=0:return b''
        if n>10_000_000 or self.received+n>12_000_000:raise ValueError('Metadata budget exceeded')
        data=read_range(self.pos,self.pos+n-1);self.pos+=n;self.received+=n;return data

def main():
    out=ROOT/'data_sources';out.mkdir(exist_ok=True)
    for name,expected in EXPECTED.items():
        url=f'https://raw.githubusercontent.com/sarulab-speech/jvs_similarity_score/{COMMIT}/{name}'
        data=urllib.request.urlopen(url,timeout=40).read(2_000_001)
        if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Pinned label hash mismatch')
        (out/name).write_bytes(data)
    r=RemoteZip()
    with zipfile.ZipFile(r) as z:
        index=[dict(name=i.filename,size=i.file_size,compressed=i.compress_size,offset=i.header_offset,crc=i.CRC,method=i.compress_type) for i in z.infolist()]
    write_json(out/'jvs_zip_index.json',index)
    write_json(out/'source.json',dict(repository='https://github.com/sarulab-speech/jvs_similarity_score',commit=COMMIT,hashes=EXPECTED,
        corpus_page='https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus',metadata_download_bytes=r.received))
    print('Pinned label hashes checked; archive entries',len(index))

if __name__=='__main__':main()
