"""Reserve eight additional JVS sentences per development voice for text transfer."""
import argparse
from runtime import ROOT, write_json
from fetch_jvs import extract
from concurrent.futures import ThreadPoolExecutor
import json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--split',choices=['dev','train','test'],default='dev')
    split_name=parser.parse_args().split
    index=json.loads((ROOT/'data_sources/jvs_zip_index.json').read_text())
    split=json.loads((ROOT/'data_jvs/manifest.json').read_text())['split']
    selected=[]
    for speaker,group in sorted(split.items()):
        if group==split_name:
            prefix=f'jvs_ver1/{speaker}/parallel100/wav24kHz16bit/'
            rows=sorted((r for r in index if r['name'].startswith(prefix) and r['name'].endswith('.wav')),key=lambda r:r['name'])
            selected.extend(rows[4:12])
    assert len(selected)==8*sum(s==split_name for s in split.values())
    write_json(ROOT/'data_sources'/('extra_text_selection.json' if split_name=='dev' else f'{split_name}_extra_text_selection.json'),[r['name'] for r in selected])
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(extract,selected))
    print('CRC-checked extra texts:',len(results),flush=True)


if __name__=='__main__':
    main()
