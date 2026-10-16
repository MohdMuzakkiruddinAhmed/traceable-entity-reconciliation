"""Evaluator-only partitioning and perturbation with explicit provenance logs."""
from collections import defaultdict
import copy
import random
import numpy as np
from ..blocking import pair_id

ROLES = ['selection', 'calibration', 'policy']

def partition_validation(records, gold, seed):
    """Assign whole canonical groups to one development role before any fitting."""
    rng = random.Random(seed)
    kinds = defaultdict(set)
    for r in records:
        kinds[r['entity_type']].add(gold[r['source_uid']]['canonical'])
    groups = {}
    for kind in sorted(kinds):
        ids = sorted(kinds[kind]); rng.shuffle(ids)
        for i, group in enumerate(ids):
            groups[group] = ROLES[i % len(ROLES)]
    assignment = {r['source_uid']: groups[gold[r['source_uid']]['canonical']] for r in records}
    return assignment, {role: sorted(g for g, r in groups.items() if r == role) for role in ROLES}

def role_masks(pairs, assignment):
    return {role: np.array([assignment[a] == role and assignment[b] == role for a,b in pairs]) for role in ROLES}

def scenario(records, graph, seed, attribute_noise=0., missingness=0., edge_drop=0., rewire=0., provenance_missingness=0.):
    """Perturb source-visible data without consulting canonical truth or labels."""
    rng = random.Random(seed)
    records = copy.deepcopy(records)
    allowed = {r['source_uid']:r for r in records}
    # Only the permitted master subgraph is needed for context; source
    # transaction rows remain untouched and are never copied into this view.
    view = graph.subgraph(allowed).copy()
    log = {'attributes_changed':0, 'optional_fields_removed':0, 'provenance_fields_removed':0,
           'edges_removed':0, 'edges_rewired':0, 'injected_cross_silo_keys':0}
    for r in records:
        if rng.random() < attribute_noise:
            r['name'] = ' '.join(w[:max(1,len(w)//2)] for w in r['name'].split())
            log['attributes_changed'] += 1
        for field in ['brand','category','description','price']:
            if r.get(field) and rng.random() < missingness:
                r[field] = ''; log['optional_fields_removed'] += 1
        if rng.random() < provenance_missingness:
            for field in ['source_batch','mapper_version','source_table']:
                if r.get(field):
                    r[field]='';log['provenance_fields_removed']+=1
        view.nodes[r['source_uid']].update(r)
    for a,b,k in sorted(view.edges(keys=True)):
        if k == 'FROM_SUPPLIER' and rng.random() < edge_drop:
            view.remove_edge(a,b,k);log['edges_removed'] += 1
    suppliers = defaultdict(list)
    for r in records:
        if r['entity_type'] == 'Supplier':
            suppliers[r['silo']].append(r['source_uid'])
    for a,b,k in sorted(view.edges(keys=True)):
        if k != 'FROM_SUPPLIER' or rng.random() >= rewire:
            continue
        alternatives = sorted(u for u in suppliers[allowed[a]['silo']] if u != b and not view.has_edge(a,u,k))
        if alternatives:
            target=rng.choice(alternatives)
            view.remove_edge(a,b,k);view.add_edge(a,target,key=k,relation=k)
            log['edges_rewired'] += 1
    # The log is evaluator-only. No corruption flags or known correct/incorrect
    # labels enter the source-visible records or inference graph.
    return records, view, log
