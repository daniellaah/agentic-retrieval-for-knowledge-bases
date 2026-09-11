"""Download hash-locked public evaluation inputs; never execute remote code."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen


def fetch(url, path, expected):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.partial')
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={'User-Agent':'ARKB-evaluation/1.0'}), timeout=60) as r, temp.open('wb') as w:
                while chunk := r.read(1024*1024): w.write(chunk)
            with temp.open('rb') as f: checksum=hashlib.file_digest(f,'sha256').hexdigest()
            if checksum!=expected['sha256'] or temp.stat().st_size!=expected['bytes']:
                raise ValueError('Downloaded bytes differ from the registered P4 input.')
            temp.replace(path)
            return dict(expected)
        except Exception:
            if attempt == 2: raise
            time.sleep(2*(attempt+1))


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--lock',type=Path,default=Path(__file__).with_name('p4-downloads.lock.json'))
    a=p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    manifest_path=a.output/'downloads.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {'schema':'arkb-public-downloads-v1','files':{},'errors':{}}
    lock=json.loads(a.lock.read_text())
    if lock['schema']!='arkb-p4-download-lock-v1':raise ValueError('Unsupported download lock.')
    for name,expected in lock['files'].items():
        url=expected['url']
        path=a.output/name
        if path.resolve().is_relative_to(a.output.resolve()) is False:raise ValueError('Invalid download path.')
        if path.exists():
            with path.open('rb') as f:checksum=hashlib.file_digest(f,'sha256').hexdigest()
            if checksum==expected['sha256'] and path.stat().st_size==expected['bytes']:
                manifest['files'][name]=dict(expected);manifest['errors'].pop(name,None)
                continue
        try:
            manifest['files'][name]=fetch(url,path,expected)
            manifest['errors'].pop(name,None)
            print(name,manifest['files'][name]['bytes'],flush=True)
        except Exception as e:
            manifest['errors'][name]={'type':type(e).__name__,'message':str(e),'url':url}
            print('FAILED',name,str(e),flush=True)
        temporary=manifest_path.with_suffix('.tmp');temporary.write_text(json.dumps(manifest,indent=2)+'\n');temporary.replace(manifest_path)
    temporary=manifest_path.with_suffix('.tmp');temporary.write_text(json.dumps(manifest,indent=2)+'\n');temporary.replace(manifest_path)
    return bool(manifest['errors'])


if __name__=='__main__': raise SystemExit(main())
