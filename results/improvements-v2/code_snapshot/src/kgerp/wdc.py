import gzip
import json
from pathlib import Path
import zipfile
from .io import ROOT, digest, write_json

FIELDS = {"title": "name", "description": "description", "brand": "brand", "price": "price", "priceCurrency": "currency"}

def prepare(config):
    archive = ROOT / "data/raw/wdc-products-50pair.zip"
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        write_json(ROOT / "data/manifests/wdc_archive_members.json", names)
        return names

def import_splits(config):
    """Official filenames are frozen in config after acquisition inspection."""
    archive = ROOT / "data/raw/wdc-products-50pair.zip"
    output, manifest, offer_sets = {}, {}, {}
    with zipfile.ZipFile(archive) as z:
        for split, member in config["files"].items():
            raw = z.read(member)
            if member.endswith(".gz"):
                raw = gzip.decompress(raw)
            raw_rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
            pairs = []
            offers = set()
            for row in raw_rows:
                records = []
                for side in ("left", "right"):
                    offers.add(str(row['id_' + side]))
                    rec = {"source_uid": "wdc:" + str(row['id_' + side]), "entity_type": "StockItem", "silo": side}
                    for attr, target in FIELDS.items():
                        value = row.get(attr + "_" + side)
                        rec[target] = "" if value is None else str(value)
                    records.append(rec)
                pairs.append({"pair_id": row['pair_id'], "left": records[0], "right": records[1],
                              "label": int(row['label']), "hard_negative": bool(row.get('is_hard_negative', False))})
            if len(pairs) != config['expected_' + split]:
                raise ValueError(f"Unexpected official {split} count: {len(pairs)}")
            if len({p['pair_id'] for p in pairs}) != len(pairs):
                raise ValueError(f"Duplicate pairs in {split}")
            output[split], offer_sets[split] = pairs, offers
            import hashlib
            manifest[split] = {"member": member, "sha256_decompressed": hashlib.sha256(raw).hexdigest(),
                               "pairs": len(pairs), "positives": sum(p['label'] for p in pairs), "offers": len(offers)}
            write_json(ROOT / "data/wdc" / (split + ".json"), pairs)
    overlaps = {f"{a}_{b}": len(offer_sets[a] & offer_sets[b]) for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]}
    if any(overlaps.values()):
        raise ValueError(f"Unexpected official offer overlap: {overlaps}")
    write_json(ROOT / "data/manifests/wdc_splits.json", {"config": config, "archive_sha256": digest(archive), "splits": manifest, "offer_overlaps": overlaps})
    return output
