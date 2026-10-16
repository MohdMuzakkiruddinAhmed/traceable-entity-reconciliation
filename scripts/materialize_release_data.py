"""Restore bundled compressed data, checking declared decoded SHA-256 values."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]

def materialize(root=ROOT, overwrite=False):
    root=Path(root).resolve()
    manifest=json.loads((root/'data/distribution/manifest.json').read_text())
    completed=[]
    for item in manifest['files']:
        if item.get('encoding') not in {'gzip','zip'}:continue
        source=(root/item['path']).resolve();target=(root/item['materialize_to']).resolve()
        if not source.is_relative_to(root) or not target.is_relative_to(root):raise ValueError('Data path escapes project')
        if target.exists():
            with target.open('rb') as handle:actual=hashlib.file_digest(handle,'sha256').hexdigest()
            if actual==item['decoded_sha256']:
                completed.append({'path':item['materialize_to'],'status':'already_verified'});continue
            if not overwrite:raise FileExistsError('Existing bytes differ; use --overwrite explicitly: '+item['materialize_to'])
        target.parent.mkdir(parents=True,exist_ok=True)
        part=target.with_name(target.name+'.materializing')
        try:
            opener=gzip.open if item['encoding']=='gzip' else open
            with opener(source,'rb') as src,part.open('wb') as dst:shutil.copyfileobj(src,dst,1024*1024)
            with part.open('rb') as handle:actual=hashlib.file_digest(handle,'sha256').hexdigest()
            if actual!=item['decoded_sha256']:raise ValueError('Decoded checksum mismatch: '+item['path'])
            part.replace(target)
        finally:
            if part.exists():part.unlink()
        completed.append({'path':item['materialize_to'],'status':'materialized_and_verified'})
    return completed

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--overwrite',action='store_true');args=parser.parse_args()
    rows=materialize(overwrite=args.overwrite)
    print(json.dumps({'files_verified':len(rows),'DataCo':'Privacy copies are under data/public; original-data model replay requires separately acquired source data.'},indent=2))

if __name__=='__main__':main()
