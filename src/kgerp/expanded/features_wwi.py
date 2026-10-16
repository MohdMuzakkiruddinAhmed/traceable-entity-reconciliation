"""WWI context combines source-safe master evidence and typed transactions."""
from collections import defaultdict
import numpy as np
from ..features_context import infer_anchors
from ..research_plus.features import weighted_graph_neighbors,weighted_flat_neighbors,weighted_context,CONTEXT_NAMES
from .features import context,NAMES


def one_hop(neighbors,records):
    visible={r['source_uid']:r for r in records}
    return {a:{b:q for b,q in adjacent.items() if visible[a]['entity_type']!=visible[b]['entity_type']} for a,adjacent in neighbors.items()}


def combined(pairs,records,graph,profiles,flat_profiles_,unit_profiles,one_profiles,config,no_anchors=False):
    # The preserved anchor extractor has an explicit original field allowlist.
    # Packaging is a visible extra attribute, but is not anchor evidence.
    anchor_records=[{k:v for k,v in r.items() if k!='quantity_per_outer'} for r in records]
    by_uid={r['source_uid']:r for r in records};anchors=[] if no_anchors else infer_anchors(anchor_records)
    gn=weighted_graph_neighbors(graph,records);fn=weighted_flat_neighbors(graph,records)
    if gn!=fn:raise ValueError('Native WWI graph/flat master context differs')
    kwargs={'min_anchors':config['models']['min_anchors'],'min_weighted_support':config['models']['min_weighted_support']}
    master,evidence=weighted_context(pairs,by_uid,gn,anchors,**kwargs)
    fm,_=weighted_context(pairs,by_uid,fn,anchors,**kwargs)
    um,_=weighted_context(pairs,by_uid,gn,anchors,weighted=False,**kwargs)
    sm,_=weighted_context(pairs,by_uid,gn,anchors,support_features=False,**kwargs)
    om,_=weighted_context(pairs,by_uid,one_hop(gn,records),anchors,**kwargs)
    tx,_=context(pairs,profiles);ftx,_=context(pairs,flat_profiles_);utx,_=context(pairs,unit_profiles)
    stx,_=context(pairs,profiles,support_features=False);otx,_=context(pairs,one_profiles,one_hop=True)
    def join(base,transaction,gate):return np.column_stack([base[:,:-1],transaction[:,:-1],gate])
    output={'weighted':join(master,tx,master[:,-1]),'flat':join(fm,ftx,fm[:,-1]),'unit':join(um,utx,um[:,-1]),
            'no_support':join(sm,stx,np.zeros(len(pairs))),'no_transactions':join(master,np.zeros_like(tx),master[:,-1]),
            'one_hop':join(om,otx,om[:,-1]),'evidence':evidence,
            'context_names':CONTEXT_NAMES[:-1]+NAMES[:-1]+['native_independent_anchor_gate']}
    return output
