import ast
from pathlib import Path
import networkx as nx
import numpy as np
import pytest
from kgerp.research_plus.features import EnhancedAttributes, explicit_size, weighted_graph_neighbors, weighted_flat_neighbors, weighted_context
from kgerp.research_plus.blocking import bounded_candidates
from kgerp.research_plus.protocol import partition_validation, role_masks, scenario
from kgerp.research_plus.learning import SigmoidCalibration, reliability, coverage_risk, prior_weights
from kgerp.research_plus.queries import PreparedGraphQueries
from kgerp.queries import graph_answers, sql_database, sql_answers

def record(silo,kind,number,name):
    return {'source_uid':f'{silo}:{kind}:{number:032x}','entity_type':kind,'silo':silo,'split':'test',
            'name':name,'category':'package','brand':'','description':'','price':'10','currency':'USD',
            'source_table':'fixture','source_batch':'batch','mapper_version':'1'}

@pytest.fixture
def context_fixture():
    records=[];graph=nx.MultiDiGraph()
    for silo,offset in [('A',0),('B',100)]:
        supplier=record(silo,'Supplier',offset+1,'Example supplier')
        records.append(supplier);graph.add_node(supplier['source_uid'],**supplier)
        for i,name in [(2,'Shirt XS'),(3,'Shirt XL')]:
            stock=record(silo,'StockItem',offset+i,name)
            records.append(stock);graph.add_node(stock['source_uid'],**stock)
            graph.add_edge(stock['source_uid'],supplier['source_uid'],key='FROM_SUPPLIER',relation='FROM_SUPPLIER')
    return records,graph

def test_size_features_distinguish_variants_and_equivalent_notation():
    assert explicit_size('Shirt XXS')==explicit_size('Shirt 2XS')
    assert explicit_size('Company S Ltd')==''
    a=record('A','StockItem',1,'Shirt 3XS');b=record('B','StockItem',2,'Shirt XXS')
    features=EnhancedAttributes().fit([a,b])
    x=features.transform([(a['source_uid'],b['source_uid'])],{a['source_uid']:a,b['source_uid']:b})
    assert x.shape==(1,20) and x[0,16]==1
    a['name']='Shirt 2XS';x=features.transform([(a['source_uid'],b['source_uid'])],{a['source_uid']:a,b['source_uid']:b})
    assert x[0,15]==1 and x[0,16]==0

def test_provenance_weights_and_sql_equivalence(context_fixture):
    records,graph=context_fixture
    a=weighted_graph_neighbors(graph,records);b=weighted_flat_neighbors(graph,records)
    assert a==b
    from kgerp.features_context import infer_anchors
    anchors=infer_anchors(records);by_uid={r['source_uid']:r for r in records}
    pairs=[(records[0]['source_uid'],records[3]['source_uid'])]
    x,e=weighted_context(pairs,by_uid,a,anchors)
    assert x[0,-1]==1 and e[0]['independent_anchor_count']==2
    changed,view,log=scenario(records,graph,9,provenance_missingness=1.)
    gn=weighted_graph_neighbors(view,changed);fn=weighted_flat_neighbors(view,changed)
    assert gn==fn
    y,_=weighted_context(pairs,{r['source_uid']:r for r in changed},gn,infer_anchors(changed))
    assert y[0,-1]==0 and log['provenance_fields_removed']>0
    z,_=weighted_context(pairs,{r['source_uid']:r for r in changed},gn,infer_anchors(changed),weighted=False)
    assert z[0,-1]==1  # explicit unit-weight ablation ignores missing lineage

def test_context_rejects_cross_silo_foreign_key(context_fixture):
    records,graph=context_fixture
    graph.add_edge(records[1]['source_uid'],records[3]['source_uid'],key='FROM_SUPPLIER')
    with pytest.raises(ValueError,match='Cross-silo'):weighted_graph_neighbors(graph,records)

def test_budgeted_candidates_are_deterministic_type_safe_and_bounded():
    records=[record(silo,'StockItem',offset+i,'Shared widget') for silo,offset in [('A',0),('B',100)] for i in range(10)]
    config={'token_prefix':4,'include_category':True,'postings_per_k':2,'max_keys':3,'max_candidates':1000}
    pairs,info=bounded_candidates(records,config,3)
    reversed_pairs,_=bounded_candidates(list(reversed(records)),config,3)
    assert pairs==reversed_pairs and len(pairs)==30 and info['cardinality_bound_respected']
    assert all(a.startswith('A:') and b.startswith('B:') for a,b in pairs)
    with pytest.raises(RuntimeError):bounded_candidates(records,{**config,'max_candidates':1},3)

def test_development_roles_keep_whole_groups_and_pair_endpoints_together():
    records=[record(silo,'StockItem',i+offset,str(i)) for silo,offset in [('A',0),('B',100)] for i in range(9)]
    gold={r['source_uid']:{'canonical':str(int(r['source_uid'].split(':')[-1],16)%100)} for r in records}
    assignment,groups=partition_validation(records,gold,17)
    assert all(len(values)==3 for values in groups.values())
    assert not any(set(a)&set(b) for i,a in enumerate(groups.values()) for j,b in enumerate(groups.values()) if i!=j)
    for i in range(9):assert assignment[records[i]['source_uid']]==assignment[records[i+9]['source_uid']]
    pairs=[(records[i]['source_uid'],records[j+9]['source_uid']) for i in range(9) for j in range(9)]
    masks=role_masks(pairs,assignment)
    assert all(not np.any(a&b) for i,a in enumerate(masks.values()) for j,b in enumerate(masks.values()) if i!=j)
    assert sum(mask.sum() for mask in masks.values())<len(pairs)

def test_probability_audit_calibration_and_abstention_edges():
    score=np.array([.05,.10,.80,.95]);y=np.array([0,0,1,1])
    calibration=SigmoidCalibration().fit(score,y)
    assert np.isfinite(calibration.transform(score)).all()
    metric,bins=reliability(y,score)
    assert sum(r['count'] for r in bins)==4 and metric['n_pairs']==4
    assert coverage_risk(y,score,.9,.1)['abstained_count']==1
    assert coverage_risk(y,score,.9,.1)['selective_risk']==0
    with pytest.raises(ValueError):SigmoidCalibration().fit(score,[0,0,0,0])
    with pytest.raises(ValueError):reliability(y,[0,.1,float('nan'),1])

def test_perturbations_are_deterministic_and_do_not_mutate_sources(context_fixture):
    records,graph=context_fixture
    records.append(record('A','Supplier',90,'Other supplier'));graph.add_node(records[-1]['source_uid'],**records[-1])
    records.append(record('B','Supplier',190,'Other supplier'));graph.add_node(records[-1]['source_uid'],**records[-1])
    edges=set(graph.edges(keys=True));names=[r['name'] for r in records]
    first,view,log=scenario(records,graph,17,attribute_noise=1.,rewire=1.)
    second,other,again=scenario(records,graph,17,attribute_noise=1.,rewire=1.)
    assert first==second and set(view.edges(keys=True))==set(other.edges(keys=True)) and log==again
    assert set(graph.edges(keys=True))==edges and [r['name'] for r in records]==names
    assert log['edges_rewired']==4 and not log['injected_cross_silo_keys']
    assert all(view.nodes[a]['silo']==view.nodes[b]['silo'] for a,b,k in view.edges(keys=True))

def test_prepared_queries_match_independent_sql_and_preserve_snapshot(context_fixture):
    records,graph=context_fixture
    mapping={r['source_uid']:r['name'] for r in records}
    stock=records[1]['source_uid'];supplier=records[0]['source_uid']
    graph.add_node('po',entity_type='PurchaseOrder',silo='A');graph.add_node('line',entity_type='PurchaseOrderLine',silo='A',ordered=10,received=5)
    graph.add_edge('po',supplier,key='ORDERED_FROM');graph.add_edge('line','po',key='PART_OF_PO');graph.add_edge('line',stock,key='FOR_STOCK_ITEM')
    prepared=PreparedGraphQueries(graph,mapping)
    with sql_database(graph,mapping) as sql:expected=sql_answers(sql)
    assert prepared.execute()==expected==graph_answers(graph,mapping)
    assert prepared.query('Q2')=={('line',)} and prepared.query('Q3')
    graph.nodes['line']['ordered']=100
    assert prepared.execute()==expected  # preparation owns the snapshot
    with pytest.raises(ValueError):prepared.query('unknown')

def test_extended_inference_modules_never_import_evaluator_truth():
    directory=Path(__file__).resolve().parents[1]/'src/kgerp/research_plus'
    for name in ['features','blocking','learning','queries']:
        source=(directory/(name+'.py')).read_text(encoding='utf-8');tree=ast.parse(source)
        assert 'data/gold' not in source
        assert not any(isinstance(n,ast.ImportFrom) and n.module and 'oracle' in n.module for n in ast.walk(tree))

def test_prior_correction_restores_development_population_prevalence():
    labels=np.array([1,1,0,0])
    weights=prior_weights(labels,.1)
    assert np.average(labels,weights=weights)==pytest.approx(.1)
    with pytest.raises(ValueError):prior_weights([0,0,0],.1)

def test_new_gated_scorer_preserves_attribute_fallback_probability():
    from kgerp.research_plus.runner import method_scores
    class Estimator:
        def __init__(self,p):self.p=p
        def predict_proba(self,x):return np.tile([1-self.p,self.p],(len(x),1))
    class Calibration:
        def transform(self,p):return p
    models={'attr':Estimator(.2),'weighted':Estimator(.9),'unweighted':Estimator(.8),'no_support':Estimator(.7),
            'calibrators':{k:Calibration() for k in ['attr','weighted','unweighted','no_support']}}
    c=np.zeros((2,6));c[1,-1]=1
    matrix={'ax':np.zeros((2,20)),'weighted':c,'flat_weighted':c.copy(),'unweighted':c,'no_support':np.zeros((2,6))}
    p,_=method_scores(models,matrix)
    assert p['GraphWeighted'].tolist()==[.2,.9]
    assert p['GraphUngated'].tolist()==[.9,.9]
    assert np.array_equal(p['GraphNoContext'],p['AttrPlus'])
    assert np.array_equal(p['GraphWeighted'],p['FlatWeighted'])
