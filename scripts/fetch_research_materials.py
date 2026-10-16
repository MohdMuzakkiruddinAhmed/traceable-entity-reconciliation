"""Restore declared model/data assets from a hash-verified companion archive."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def restore(archive, root=ROOT):
    root = root.resolve()
    manifest = json.loads((root / 'data/research_materials.json').read_text())
    if not manifest.get('archive_sha256') or sha(archive) != manifest['archive_sha256']:
        raise ValueError('Companion archive SHA-256 mismatch')
    restored = []
    with zipfile.ZipFile(archive) as bundle:
        if len(bundle.namelist()) != len(set(bundle.namelist())):
            raise ValueError('Duplicate archive members')
        for item in manifest['files']:
            target = (root / item['path']).resolve()
            if not target.is_relative_to(root):
                raise ValueError('Asset path escapes the repository')
            if target.exists():
                if sha(target) != item['sha256']:
                    raise FileExistsError('Existing asset differs; preserve or relocate it first: ' + item['path'])
                restored.append({'path': item['path'], 'status': 'already_verified'})
                continue
            member = manifest['archive_root'] + item['path']
            info = bundle.getinfo(member)
            if info.file_size != item['bytes']:
                raise ValueError('Asset size mismatch: ' + item['path'])
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + '.restoring')
            try:
                with bundle.open(info) as src, partial.open('wb') as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024)
                if sha(partial) != item['sha256']:
                    raise ValueError('Asset SHA-256 mismatch: ' + item['path'])
                partial.replace(target)
            finally:
                if partial.exists():
                    partial.unlink()
            restored.append({'path': item['path'], 'status': 'restored_and_verified'})
    return restored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--archive', type=Path, help='Previously downloaded companion archive')
    source.add_argument('--repository', help='Public GitHub repository in OWNER/NAME form')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'data/research_materials.json').read_text())
    if args.archive:
        archive = args.archive.resolve()
    else:
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+', args.repository):
            parser.error('Use OWNER/NAME for --repository')
        archive = ROOT / 'data/downloads' / manifest['asset_name']
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            url = 'https://github.com/' + args.repository + '/releases/download/' + manifest['release_tag'] + '/' + manifest['asset_name']
            partial = archive.with_name(archive.name + '.downloading')
            try:
                request = urllib.request.Request(url, headers={'User-Agent': 'kg-erp-material-restorer/1.1'})
                with urllib.request.urlopen(request, timeout=60) as src, partial.open('wb') as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024)
                if sha(partial) != manifest['archive_sha256']:
                    raise ValueError('Downloaded companion archive SHA-256 mismatch')
                partial.replace(archive)
            finally:
                if partial.exists():
                    partial.unlink()
    rows = restore(archive)
    print(json.dumps({'status':'passed','assets_verified':len(rows),
                      'models_fitted':False,'software_or_results_replaced':False}, indent=2))


if __name__ == '__main__':
    main()
