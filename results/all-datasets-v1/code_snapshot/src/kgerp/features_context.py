"""Label-blind, single-pass independent anchors; graph and SQL extractors."""
from collections import Counter, defaultdict
import sqlite3
import numpy as np
from .blocking import normalize

NAMES = ['anchor_jaccard','anchor_count','degree_similarity','category_cosine','support_fraction','has_supported_context']
VISIBLE_FIELDS = {'source_uid','silo','entity_type','split','name','category','brand','description',
                  'price','currency','source_table','source_batch','mapper_version'}

def assert_visible(records):
    for r in records:
        extra = set(r) - VISIBLE_FIELDS
        if extra:
            raise ValueError(f'Model input contains forbidden fields: {sorted(extra)}')
        if not r['source_uid'].startswith(('A:','B:')):
            raise ValueError('Original source identity supplied to matcher')

def infer_anchors(records):
    assert_visible(records)
    index = defaultdict(lambda: defaultdict(list))
    for r in records:
        name = normalize(r['name'])
        if name:
            index[(r['entity_type'],name)][r['silo']].append(r)
    anchors = []
    for sides in index.values():
        if len(sides['A']) == len(sides['B']) == 1:
            a,b = sides['A'][0],sides['B'][0]
            if a.get('category') and b.get('category') and a['category'] != b['category']:
                continue
            anchors.append((a['source_uid'],b['source_uid']))
    return sorted(anchors)

def graph_neighbors(graph, records):
    allowed = {r['source_uid'] for r in records}
    outgoing, incoming = defaultdict(set),defaultdict(set)
    for a,b,key in graph.edges(keys=True):
        if key == 'FROM_SUPPLIER' and a in allowed and b in allowed:
            outgoing[a].add(b)
            incoming[b].add(a)
    neighbors = {}
    for r in records:
        source_uid = r['source_uid']
        if r['entity_type'] == 'Supplier':
            neighbors[source_uid] = incoming[source_uid]
        else:
            # 2-hop co-supplied items plus typed supplier; no PO identifiers.
            values = set(outgoing[source_uid])
            for supplier in outgoing[source_uid]:
                values.update(incoming[supplier])
            neighbors[source_uid] = values - {source_uid}
    return neighbors

def flat_neighbors(graph, records):
    """Independent relational adjacency joins on the same permitted evidence."""
    allowed = {r['source_uid'] for r in records}
    with sqlite3.connect(':memory:') as sql:
        sql.execute('CREATE TABLE links(item TEXT,supplier TEXT)')
        sql.executemany('INSERT INTO links VALUES(?,?)', [(a,b) for a,b,k in graph.edges(keys=True) if k=='FROM_SUPPLIER' and a in allowed and b in allowed])
        sql.execute('CREATE INDEX item_idx ON links(item)')
        sql.execute('CREATE INDEX supplier_idx ON links(supplier)')
        output = defaultdict(set)
        for a,b in sql.execute('SELECT item,supplier FROM links UNION SELECT supplier,item FROM links UNION SELECT a.item,b.item FROM links a JOIN links b ON a.supplier=b.supplier WHERE a.item<>b.item'):
            output[a].add(b)
    return {r['source_uid']:output[r['source_uid']] for r in records}

def context_features(pairs, records, neighbors, anchors, min_anchors=2):
    output,evidence = [],[]
    anchor_map = dict(anchors)
    for left,right in pairs:
        a,b = neighbors.get(left,set()),neighbors.get(right,set())
        supported = [(u,anchor_map[u]) for u in sorted(a) if u in anchor_map and anchor_map[u] in b and u != left and anchor_map[u] != right]
        count = len(supported)
        def distribution(nodes):
            return Counter((records[n]['entity_type'],records[n].get('category','')) for n in nodes if records[n].get('category'))
        x,y = distribution(a),distribution(b)
        dot = sum(v*y[k] for k,v in x.items())
        cosine = dot / max(1,(sum(v*v for v in x.values())*sum(v*v for v in y.values()))**.5)
        available = count >= min_anchors
        output.append([count/max(1,len(a)+len(b)-count),count,min(len(a),len(b))/max(1,len(a),len(b)),
                       cosine,2*count/max(1,len(a)+len(b)),float(available)])
        evidence.append({'independent_anchor_count':count,'has_supported_context':available,
                         'anchor_source_pairs':supported,'neighbor_source_uids':sorted(a|b)})
    return np.asarray(output,dtype=float).reshape(-1,len(NAMES)),evidence
