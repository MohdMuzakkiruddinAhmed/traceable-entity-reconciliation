"""Source-visible product variants and weighted typed-neighbor evidence."""
from collections import Counter, defaultdict
import math
import re
import sqlite3
import numpy as np
from ..blocking import normalize
from ..features_attribute import AttributeFeatures, NAMES as BASE_NAMES
from ..features_context import infer_anchors

EXTRA_NAMES = ['number_jaccard', 'alphanumeric_jaccard', 'size_agreement', 'size_conflict',
               'size_missing', 'name_length_ratio', 'digit_missing']
ATTRIBUTE_NAMES = BASE_NAMES + EXTRA_NAMES
CONTEXT_NAMES = ['weighted_anchor_jaccard', 'weighted_support', 'degree_similarity',
                 'weighted_category_cosine', 'supported_fraction', 'context_available']

def explicit_size(name):
    tokens = normalize(name).split()
    alternatives = {'xxs': '2xs', 'xxxs': '3xs', 'xxl': '2xl', 'xxxl': '3xl'}
    # A bounded, explicit clothing-size grammar; arbitrary one-letter tokens
    # such as "s" are excluded because they often occur outside size fields.
    for token in reversed(tokens):
        if re.fullmatch(r'(?:[2-5]x[sl]|x{1,3}[sl])', token):
            return alternatives.get(token, token)
    return ''

def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))

class EnhancedAttributes:
    def fit(self, records):
        self.base = AttributeFeatures().fit(records)
        return self

    def transform(self, pairs, records):
        base = self.base.transform(pairs, records)
        extra = []
        for a, b in pairs:
            x, y = normalize(records[a].get('name')), normalize(records[b].get('name'))
            nx, ny = set(re.findall(r'\d+(?:\.\d+)?', x)), set(re.findall(r'\d+(?:\.\d+)?', y))
            ax = {t for t in x.split() if any(c.isdigit() for c in t)}
            ay = {t for t in y.split() if any(c.isdigit() for c in t)}
            sx, sy = explicit_size(x), explicit_size(y)
            both = bool(sx and sy)
            extra.append([jaccard(nx, ny), jaccard(ax, ay), float(both and sx == sy),
                          float(both and sx != sy), float(not both), min(len(x), len(y)) / max(1, len(x), len(y)),
                          float(not nx or not ny)])
        return np.column_stack([base, np.asarray(extra).reshape(-1, len(EXTRA_NAMES))])

def lineage_quality(record):
    return sum(bool(record.get(k)) for k in ['source_table', 'source_batch', 'mapper_version']) / 3

def weighted_graph_neighbors(graph, records):
    """Reliability is inferred from visible provenance completeness, not noise labels."""
    visible = {r['source_uid']: r for r in records}
    out, reverse = defaultdict(dict), defaultdict(dict)
    for a, b, relation in graph.edges(keys=True):
        if relation != 'FROM_SUPPLIER' or a not in visible or b not in visible:
            continue
        if visible[a]['silo'] != visible[b]['silo']:
            raise ValueError('Cross-silo source key in typed context')
        if visible[a]['entity_type'] != 'StockItem' or visible[b]['entity_type'] != 'Supplier':
            raise ValueError('Invalid typed context edge')
        quality = min(lineage_quality(visible[a]), lineage_quality(visible[b]))
        out[a][b] = quality; reverse[b][a] = quality
    result = {}
    for a, r in visible.items():
        if r['entity_type'] == 'Supplier':
            result[a] = dict(reverse[a])
        else:
            neighbors = dict(out[a])
            for supplier, q in out[a].items():
                for item, other_q in reverse[supplier].items():
                    if item != a:
                        neighbors[item] = max(neighbors.get(item, 0), min(q, other_q))
            result[a] = neighbors
    return result

def weighted_flat_neighbors(graph, records):
    visible = {r['source_uid']: r for r in records}
    links = []
    for a, b, relation in graph.edges(keys=True):
        if relation == 'FROM_SUPPLIER' and a in visible and b in visible:
            if visible[a]['silo'] != visible[b]['silo']:
                raise ValueError('Cross-silo context edge')
            if visible[a]['entity_type'] != 'StockItem' or visible[b]['entity_type'] != 'Supplier':
                raise ValueError('Invalid typed context edge')
            links.append((a, b, min(lineage_quality(visible[a]), lineage_quality(visible[b]))))
    with sqlite3.connect(':memory:') as sql:
        sql.execute('CREATE TABLE links(item TEXT,supplier TEXT,quality REAL)')
        sql.executemany('INSERT INTO links VALUES(?,?,?)', links)
        sql.executescript('CREATE INDEX item_idx ON links(item); CREATE INDEX supplier_idx ON links(supplier);')
        joined = sql.execute('''SELECT a,b,MAX(quality) FROM (
          SELECT item AS a,supplier AS b,quality FROM links
          UNION ALL SELECT supplier,item,quality FROM links
          UNION ALL SELECT x.item,y.item,MIN(x.quality,y.quality)
            FROM links x JOIN links y ON x.supplier=y.supplier WHERE x.item<>y.item
        ) GROUP BY a,b''').fetchall()
    output = {a: {} for a in visible}
    for a, b, q in joined:
        output[a][b] = q
    return output

def weighted_context(pairs, records, neighborhoods, anchors, min_anchors=2, min_weighted_support=1., weighted=True, support_features=True):
    anchor_map = dict(anchors)
    result, evidence = [], []
    for left, right in pairs:
        a, b = neighborhoods.get(left, {}), neighborhoods.get(right, {})
        matched = [(u, anchor_map[u]) for u in sorted(a) if u in anchor_map and anchor_map[u] in b and u != left and anchor_map[u] != right]
        weights = []
        for u, v in matched:
            completeness = min(sum(bool(records[k].get(f)) for f in ['name', 'category', 'brand', 'description']) / 4 for k in [u, v])
            weights.append(min(a[u], b[v]) * completeness if weighted else 1.)
        support = sum(weights)
        def distribution(nodes):
            counts = Counter()
            for u, q in nodes.items():
                if records[u].get('category'):
                    counts[(records[u]['entity_type'], records[u]['category'])] += q if weighted else 1.
            return counts
        x, y = distribution(a), distribution(b)
        cosine = sum(v * y[k] for k, v in x.items()) / max(1., math.sqrt(sum(v*v for v in x.values()) * sum(v*v for v in y.values())))
        available = len(matched) >= min_anchors and support >= min_weighted_support
        values = [support / max(1., len(a) + len(b) - support), support, min(len(a),len(b)) / max(1,len(a),len(b)),
                  cosine, 2*support / max(1, len(a)+len(b)), float(available)]
        if not support_features:
            values[0] = values[1] = values[4] = values[5] = 0.
        result.append(values)
        evidence.append({'neighbor_source_uids': sorted(set(a) | set(b)), 'anchor_source_pairs': matched,
                         'anchor_weights': weights, 'independent_anchor_count': len(matched),
                         'weighted_support': support, 'has_supported_context': available})
    return np.asarray(result, dtype=float).reshape(-1, 6), evidence
