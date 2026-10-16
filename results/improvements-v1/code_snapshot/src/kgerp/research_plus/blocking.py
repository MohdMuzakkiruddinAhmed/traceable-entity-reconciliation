"""Budgeted retrieval with measured recall; no oracle or replica-key features."""
from collections import defaultdict
import hashlib
from rapidfuzz.fuzz import token_sort_ratio
from ..blocking import block_keys, normalize

def bounded_candidates(records, config, k):
    if k <= 0:
        raise ValueError('k must be positive')
    visible = {r['source_uid']: r for r in records}
    index = defaultdict(list)
    for record in records:
        if record['silo'] == 'B':
            for key in block_keys(record, config):
                index[key].append(record['source_uid'])
    full_sizes = {key: len(ids) for key, ids in index.items()}
    cap = config['postings_per_k'] * k
    # One label-blind hashed ordering per posting list; no inference from UID
    # structure, canonical groups, copied transactions or workload suffixes.
    for key, ids in index.items():
        index[key] = sorted(ids, key=lambda u: hashlib.sha256(u.encode()).digest())[:cap]
    pairs = []
    for record in sorted((r for r in records if r['silo'] == 'A'), key=lambda r:r['source_uid']):
        keys = sorted(block_keys(record, config), key=lambda key:(full_sizes.get(key,0), key))[:config['max_keys']]
        pool = set()
        for key in keys:
            pool.update(index.get(key, []))
        name = normalize(record['name'])
        ranked = sorted(pool, key=lambda u:(-token_sort_ratio(name, normalize(visible[u]['name'])), u))
        pairs.extend((record['source_uid'], u) for u in ranked[:k])
        if len(pairs) > config['max_candidates']:
            raise RuntimeError('Total bounded candidate budget exceeded')
    left = sum(r['silo']=='A' for r in records)
    return pairs, {'candidate_count': len(pairs), 'source_records':len(records), 'left_records':left,
                   'per_left_limit':k, 'cardinality_bound':left*k,
                   'max_full_posting':max(full_sizes.values(),default=0),
                   'postings_cap':cap, 'max_pool_bound':config['max_keys']*cap,
                   'cardinality_bound_respected':len(pairs)<=left*k}
