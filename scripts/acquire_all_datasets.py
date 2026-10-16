"""Download the expanded study inputs without replacing original acquisitions."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw'


def download(spec):
    name, url, provenance, md5 = spec
    target = RAW / name
    if not target.exists():
        part = target.with_suffix(target.suffix + '.part')
        with requests.get(url, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            with part.open('wb') as handle:
                for block in response.iter_content(1024 * 1024):
                    handle.write(block)
        part.replace(target)
    sha = hashlib.sha256()
    checksum = hashlib.md5()
    with target.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            sha.update(block)
            checksum.update(block)
    if md5 and checksum.hexdigest() != md5:
        raise RuntimeError('Publisher MD5 mismatch: ' + name)
    print(f'Download verified: {name}, {target.stat().st_size:,} bytes', flush=True)
    return {'file': name, 'url': url, 'source': provenance, 'bytes': target.stat().st_size,
            'sha256': sha.hexdigest(), 'md5': checksum.hexdigest(), 'publisher_md5_verified': bool(md5)}


def main():
    RAW.mkdir(exist_ok=True)
    bpi = requests.get('https://data.4tu.nl/v2/articles/35ed7122-966a-484e-a0e1-749b64e3366d', timeout=45).json()
    (ROOT / 'data/manifests/bpi2019_publisher.json').write_text(json.dumps(bpi, indent=2), encoding='utf-8')
    file = next(f for f in bpi['files'] if f['name'] == 'BPI_Challenge_2019.xes')
    release = requests.get('https://api.github.com/repos/trquanganhng/DataCoSupplyChain/releases/tags/Download', timeout=45)
    release.raise_for_status()
    release = release.json()
    (ROOT / 'data/manifests/dataco_mirror_release.json').write_text(json.dumps(release, indent=2), encoding='utf-8')
    asset = next(a for a in release['assets'] if a['name'] == 'DataCoSupplyChainDataset.csv')
    files = [
        ('olist-v2.zip', 'https://www.kaggle.com/api/v1/datasets/download/olistbr/brazilian-ecommerce?datasetVersionNumber=2', 'Official Olist Kaggle v2; CC BY-NC-SA 4.0', None),
        ('DataCoSupplyChainDataset.csv', asset['browser_download_url'], 'Mirror of Mendeley DataCo v5; official HTTP acquisition returned 403; mirror asset ID ' + str(asset['id']) + '; primary DOI 10.17632/8gx2fvg2k6.5; CC BY 4.0', None),
        ('BPI_Challenge_2019.xes', 'https://ndownloader.figshare.com/files/' + str(file['id']), 'Official 4TU BPI Challenge 2019 v1; legacy Figshare file ID from publisher metadata, current 4TU download returned 503; CC BY 4.0', file['computed_md5']),
    ]
    for hardness in [20, 80]:
        files.append((f'wdc-products-{hardness}pair.zip', f'https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/wdc-products/{hardness}pair.zip', 'Official WDC Products pairwise ' + str(hardness) + '% corner cases', None))
    revision = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'
    model = RAW / 'models/all-MiniLM-L6-v2'
    model.mkdir(parents=True, exist_ok=True)
    for filename in ['onnx/model_quint8_avx2.onnx', 'tokenizer.json', 'tokenizer_config.json', 'config.json', 'README.md']:
        files.append(('models/all-MiniLM-L6-v2/' + filename.replace('onnx/', ''),
                      f'https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/{revision}/{filename}',
                      'Official sentence-transformers/all-MiniLM-L6-v2; pinned revision ' + revision + '; Apache-2.0 model card', None))
    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(download, files))
    manifest = {'datasets': ['WWI', 'WDC', 'Olist', 'DataCo', 'BPI2019'], 'files': results,
                'dataco_provenance_limit': 'Mirror bytes are checksummed and schema/count verified; not claimed byte-identical to inaccessible publisher files.'}
    (ROOT / 'data/manifests/expanded_downloads.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
