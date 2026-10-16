from collections import defaultdict
import hashlib
import re
import unicodedata

def normalize(value):
    value = unicodedata.normalize('NFKD', str(value or '')).casefold()
    return ' '.join(re.findall(r'[a-z0-9]+', value))

def pair_id(left, right):
    return hashlib.sha256((left + '|' + right).encode()).hexdigest()[:24]

def block_keys(record, config):
    prefix = config['token_prefix']
    tokens = normalize(record['name']).split()
    keys = {('token', t[:prefix]) for t in tokens if len(t) >= 2}
    if config.get('include_category') and record.get('category'):
        keys.add(('category', normalize(record['category'])))
    return {(record['entity_type'],) + k for k in keys}

def candidates(records, config):
    index = defaultdict(set)
    left = [r for r in records if r['silo'] == 'A']
    right = [r for r in records if r['silo'] == 'B']
    for record in right:
        for key in block_keys(record, config):
            index[key].add(record['source_uid'])
    pairs = []
    for record in sorted(left, key=lambda r:r['source_uid']):
        matches = set()
        for key in block_keys(record, config):
            matches.update(index[key])
        for target in sorted(matches):
            pairs.append((record['source_uid'], target))
            if len(pairs) > config.get('max_candidates', 2_000_000):
                raise RuntimeError('Candidate budget exceeded; refine blocker on development data')
    universe = sum(sum(r['entity_type']==kind for r in left)*sum(r['entity_type']==kind for r in right)
                   for kind in {r['entity_type'] for r in records})
    return pairs, {'candidate_count':len(pairs), 'possible_pairs':universe,
                   'reduction_ratio':1-len(pairs)/max(1,universe), 'max_index_block':max(map(len,index.values()),default=0)}
