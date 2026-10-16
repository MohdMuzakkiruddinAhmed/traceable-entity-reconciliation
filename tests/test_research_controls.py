import ast
from pathlib import Path
import random
import networkx as nx
import numpy as np
import pytest
from kgerp.silo_generator import generate,split_groups
from kgerp.blocking import candidates
from kgerp.features_context import assert_visible,graph_neighbors,flat_neighbors,context_features,infer_anchors
from kgerp.identity import resolve
from kgerp.metrics import binary,set_metrics,paired_bootstrap
from kgerp.graph_loader import build_graph
from kgerp.queries import graph_answers,sql_database,sql_answers

@pytest.fixture
def tables():
    return {
      'Purchasing.Suppliers':[{'SupplierID':i,'SupplierName':f'Acme {i}','SupplierCategoryID':1} for i in range(1,11)],
      'Warehouse.StockItems':[{'StockItemID':i,'StockItemName':f'Tool {i}','SupplierID':(i-1)%10+1,'UnitPackageID':1,'UnitPrice':2} for i in range(1,31)],
      'Purchasing.PurchaseOrders':[{'PurchaseOrderID':i,'SupplierID':i,'OrderDate':'2020-01-01'} for i in range(1,11)],
      'Purchasing.PurchaseOrderLines':[{'PurchaseOrderLineID':i,'PurchaseOrderID':(i-1)%10+1,'StockItemID':i,'OrderedOuters':10,'ReceivedOuters':8,'ExpectedUnitPricePerOuter':2} for i in range(1,31)],
      'Warehouse.StockItemTransactions':[{'StockItemTransactionID':i,'StockItemID':i,'SupplierID':(i-1)%10+1,'PurchaseOrderID':None,'Quantity':5,'TransactionOccurredWhen':'2020-01-01'} for i in range(1,31)],
      'Purchasing.SupplierTransactions':[{'SupplierTransactionID':i,'SupplierID':i,'TransactionAmount':10,'TransactionDate':'2020-01-01'} for i in range(1,11)],
    }

CONFIG={'typo_probability':.25,'abbreviation_probability':.35,'optional_missingness':.2,'edge_drop':.15}

def test_generator_reproducible_and_truth_separate(tables):
    a,g,gold=generate(tables,17,CONFIG)
    b,h,truth=generate(tables,17,CONFIG)
    assert a==b and gold==truth
    assert list(g.edges(keys=True))==list(h.edges(keys=True))
    assert_visible(a)
    for node,attrs in g.nodes(data=True):
        assert not {'canonical','original_row','SupplierID','StockItemID','PurchaseOrderID'}&set(attrs)
        assert node.split(':')[-1].isalnum() and len(node.split(':')[-1])==32

def test_group_disjoint_before_aliasing(tables):
    records,_,gold=generate(tables,17,CONFIG)
    groups={}
    for r in records:
        groups.setdefault(gold[r['source_uid']]['canonical'],set()).add(r['split'])
    assert all(len(splits)==1 for splits in groups.values())
    assert set.union(*groups.values())=={'train','validation','test'}

def test_supplier_split_includes_active_test_entities(tables):
    tables['Purchasing.Suppliers'].extend([{'SupplierID':i,'SupplierName':f'Inactive {i}','SupplierCategoryID':2} for i in range(11,17)])
    groups=split_groups(tables)
    active={r['SupplierID'] for r in tables['Purchasing.PurchaseOrders']}
    assert any(split=='test' and key[0]=='Purchasing.Suppliers' and key[1] in active for key,split in groups.items())

def test_no_gold_imports_in_inference():
    folder=Path(__file__).resolve().parents[1]/'src/kgerp'
    for name in ['blocking','features_attribute','features_context','matchers','identity','queries']:
        text=(folder/(name+'.py')).read_text(encoding='utf-8')
        tree=ast.parse(text)
        assert 'data/gold' not in text
        assert not any(isinstance(n,ast.ImportFrom) and n.module and 'oracle' in n.module for n in ast.walk(tree))

def test_reject_original_identity():
    with pytest.raises(ValueError):assert_visible([{'source_uid':'A:Supplier:opaque','SupplierID':7}])

def test_fixed_blocking_and_type_constraints(tables):
    records,_,_=generate(tables,17,CONFIG)
    config={'token_prefix':4,'include_category':True,'max_candidates':10000}
    pairs,stats=candidates(records,config);same,_=candidates(records,config)
    by_uid={r['source_uid']:r for r in records}
    assert pairs==same and len(pairs)==len(set(pairs))
    assert all(by_uid[a]['entity_type']==by_uid[b]['entity_type'] for a,b in pairs)
    with pytest.raises(RuntimeError):candidates(records,{**config,'max_candidates':1})

def test_raw_graph_idempotence_and_no_invented_relations(tables):
    g=build_graph(tables)
    assert g.has_edge('wwi:Purchasing.PurchaseOrderLines:1','wwi:Warehouse.StockItems:1','FOR_STOCK_ITEM')
    assert not any(k=='FOR_PO' and g.nodes[a]['entity_type']=='SupplierTransaction' for a,b,k in g.edges(keys=True))
    n,e=len(g),g.number_of_edges();build_graph(tables,g)
    assert (len(g),g.number_of_edges())==(n,e)
    tables['Warehouse.StockItems'][0]['SupplierID']=999
    with pytest.raises(ValueError):build_graph(tables)

def test_sql_context_equivalence_and_no_self_anchor(tables):
    records,g,gold=generate(tables,17,{**CONFIG,'typo_probability':0,'abbreviation_probability':0,'edge_drop':0})
    by_uid={r['source_uid']:r for r in records}
    pairs,_=candidates(records,{'token_prefix':4,'include_category':True})
    gn,fn=graph_neighbors(g,records),flat_neighbors(g,records)
    assert gn==fn
    anchors=infer_anchors(records)
    x,e=context_features(pairs,by_uid,gn,anchors)
    y,_=context_features(pairs,by_uid,fn,anchors)
    assert np.array_equal(x,y)
    for (a,b),trace in zip(pairs,e):assert (a,b) not in trace['anchor_source_pairs']
    no,_=context_features(pairs,by_uid,gn,[])
    assert not no[:,-1].any()

def test_constrained_identity_prevents_conflict_and_type_merge():
    records={k:{'entity_type':kind} for k,kind in [('a','Supplier'),('b','Supplier'),('c','Supplier'),('d','StockItem')]}
    evidence=[{'neighbor_source_uids':[],'anchor_source_pairs':[]}]*3
    mapping,decisions=resolve([('a','b'),('a','c'),('b','d')],[.99,.98,.97],records,evidence,.9,.1,'test','v1')
    assert mapping['a']==mapping['b'] and mapping['c']!=mapping['a']
    assert mapping['d']!=mapping['b']
    assert {d['status'] for d in decisions}=={'accepted','one_to_one_conflict','type_conflict'}

def test_six_query_oracles_detect_removed_identity_link(tables):
    records,g,gold=generate(tables,17,CONFIG)
    truth={r['source_uid']:gold[r['source_uid']]['canonical'] for r in records}
    with sql_database(g,truth) as sql:answers=sql_answers(sql)
    assert graph_answers(g,truth)==answers
    assert len(answers)==6 and answers['Q6']
    raw={r['source_uid']:r['source_uid'] for r in records}
    with sql_database(g,raw) as sql:unresolved=sql_answers(sql)
    assert unresolved['Q6']!=answers['Q6']
    assert unresolved['Q2']==answers['Q2'] and unresolved['Q4']==answers['Q4']

def test_metrics_include_blocking_misses_and_empty_sets():
    out=binary([1,1,0],[1,0,1],total_positives=4)
    assert out['f1']==.5 and out['end_to_end_recall']==.25 and out['blocking_recall']==.5
    assert set_metrics(set(),set())['f1']==1
    assert set_metrics({('x',)},set())['f1']==0
    pairs=[('a','b'),('a','c')];gold={'a':{'canonical':'1'},'b':{'canonical':'1'},'c':{'canonical':'2'}}
    assert paired_bootstrap(np.array([1,0]),np.array([True,False]),np.array([True,False]),pairs,gold,17,20)==(0.,0.)
