"""Acquire immutable official inputs; never download optional datasets."""
import hashlib
import json
from pathlib import Path
import re
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw"
RAW.mkdir(parents=True, exist_ok=True)
manifest = {}

def download(name, url):
    target = RAW / name
    if not target.exists():
        with requests.get(url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with target.with_suffix(target.suffix + ".part").open("wb") as out:
                for chunk in response.iter_content(1024 * 1024):
                    out.write(chunk)
        target.with_suffix(target.suffix + ".part").replace(target)
    digest = hashlib.file_digest(target.open("rb"), "sha256").hexdigest()
    manifest[name] = {"url": url, "bytes": target.stat().st_size, "sha256": digest}
    print(f"Acquired {name}: {target.stat().st_size:,} bytes", flush=True)

page_url = "https://webdatacommons.org/largescaleproductcorpus/wdc-products/index.html"
page = requests.get(page_url, timeout=30)
page.raise_for_status()
links = re.findall(r'href=[\"\']([^\"\']+)[\"\'][^>]*>\s*50pair', page.text)
if len(links) != 1:
    raise RuntimeError("Cannot identify official WDC 50pair archive unambiguously")
download("wdc-products-50pair.zip", links[0])
download("WideWorldImporters-Standard.bak", "https://github.com/microsoft/sql-server-samples/releases/download/wide-world-importers-v1.0/WideWorldImporters-Standard.bak")
(ROOT / "data/manifests").mkdir(parents=True, exist_ok=True)
(ROOT / "data/manifests/downloads.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
