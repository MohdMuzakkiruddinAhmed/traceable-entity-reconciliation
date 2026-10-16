"""Evaluator-only gold access. Never import this module from feature/scoring code."""
from collections import defaultdict
import numpy as np

def labels(pairs,gold):
    return np.array([int(gold[a]['canonical']==gold[b]['canonical']) for a,b in pairs],dtype=int)

def true_pair_count(records,gold):
    sides = defaultdict(lambda:defaultdict(int))
    for r in records:
        sides[gold[r['source_uid']]['canonical']][r['silo']]+=1
    return sum(x['A']*x['B'] for x in sides.values())

def oracle_mapping(records,gold):
    return {r['source_uid']:gold[r['source_uid']]['canonical'] for r in records}
