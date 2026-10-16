import ast
import copy
from pathlib import Path
import networkx as nx
import numpy as np
import pytest
from kgerp.expanded.attributes import FastAttributes
from kgerp.research_plus.features import EnhancedAttributes
from kgerp.expanded.features import graph_profiles,flat_profiles,context,day
from kgerp.expanded.protocol import perturb
from kgerp.expanded.queries import PreparedQueries,SQLQueries


def fixture():
    graph=nx.MultiDiGraph();records=[]
    for side in ['A','B']:
        for i in range(2):
            uid=f'{side}:Customer:{i}'
            r={'source_uid':uid,'silo':side,'split':'test','entity_type':'Customer','name':f'Jane shirt {i} XL','category':'Retail','brand':'State',
               'description':'public example','price':'10','currency':'','source_table':'source','source_batch':'batch','mapper_version':'v4'}
            graph.add_node(uid,**r);records.append(r)
            tx=f'{side}:tx:{i}'
            graph.add_node(tx,entity_type='Transaction',silo=side,date='2020-01-01',amount=12.34,quantity=2,
                amount_unit='dataset_amount',quantity_unit='items',activity='received',source_table='tx',source_batch='batch',mapper_version='v4')
            graph.add_edge(uid,tx,key='HAS_TRANSACTION')
            for label in ['category:sports','department:clothes']:
                tag=side+':'+label;graph.add_node(tag,entity_type='ContextCategory',label=label,silo=side)
                graph.add_edge(tx,tag,key='HAS_CATEGORY')
    return records,graph


def test_fast_attributes_preserve_original_features():
    records,_=fixture();records[1]['name']='Jane shirt 12 XXL';records[1]['currency']='EUR'
    records[2]['price']='';records[2]['description']='';records[3]['name']=''
    by_uid={r['source_uid']:r for r in records};pairs=[(a['source_uid'],b['source_uid']) for a in records[:2] for b in records[2:]]
    original=EnhancedAttributes().fit(records);fast=FastAttributes().fit(records)
    assert np.allclose(original.transform(pairs,by_uid),fast.transform(pairs,by_uid),atol=1e-14,rtol=0)


def test_graph_sql_context_and_unit_compatibility():
    records,graph=fixture();pairs=[('A:Customer:0','B:Customer:0')]
    gp=graph_profiles(graph,records);fp=flat_profiles(graph,records)
    a,e=context(pairs,gp);b,_=context(pairs,fp)
    assert np.array_equal(a,b) and e[0]['context_available']
    graph.nodes['B:tx:0']['quantity_unit']='purchase_outers'
    changed,e=context(pairs,graph_profiles(graph,records))
    assert not e[0]['quantity_units_compatible'] and changed[0,7]==0


def test_rewiring_preserves_sources_and_silos():
    records,graph=fixture();original_nodes=copy.deepcopy(dict(graph.nodes(data=True)));original_edges=set(graph.edges(keys=True))
    _,changed,logs=perturb(records,graph,3,misleading_edge_rewire=1.)
    assert dict(graph.nodes(data=True))==original_nodes and set(graph.edges(keys=True))==original_edges
    assert logs['edges_rewired']==4
    for a,b,k in changed.edges(keys=True):
        if k=='HAS_TRANSACTION':assert changed.nodes[a]['silo']==changed.nodes[b]['silo']


def test_eight_queries_are_nonempty_and_equivalent():
    records,graph=fixture();identity={r['source_uid']:'canonical:'+r['source_uid'].rsplit(':',1)[-1] for r in records}
    graph_answers=PreparedQueries(graph,identity).execute();sql=SQLQueries(graph,identity)
    assert graph_answers==sql.execute() and len(graph_answers)==8 and all(graph_answers.values())
    assert graph_answers['Q3']=={('A:Customer:0',1234,1),('A:Customer:1',1234,1)}
    sql.close()


def test_category_query_deduplicates_repeated_transaction_evidence():
    records,graph=fixture()
    for side in ['A','B']:
        original=side+':tx:0';additional=side+':additional_tx'
        graph.add_node(additional,**dict(graph.nodes[original]))
        graph.add_edge(side+':Customer:0',additional,key='HAS_TRANSACTION')
        for _,tag,key in list(graph.out_edges(original,keys=True)):
            graph.add_edge(additional,tag,key=key)
    identity={r['source_uid']:'canonical:'+r['source_uid'].rsplit(':',1)[-1] for r in records}
    expected=PreparedQueries(graph,identity).execute();sql=SQLQueries(graph,identity)
    assert sql.execute()==expected
    assert len(expected['Q1'])==4
    assert ('A:Customer:0',2468,2) in expected['Q3']
    sql.close()


def test_wwi_owner_projections_keep_packaging_units_and_native_rows(monkeypatch,tmp_path):
    import json
    import pickle
    from kgerp.expanded import wwi
    supplier='A:Supplier:example';item='A:StockItem:example'
    provenance={'silo':'A','source_batch':'fixture','mapper_version':'fixture-v1'}
    records=[{'source_uid':supplier,'entity_type':'Supplier','category':'Wholesale','brand':'',
              'source_table':'Purchasing.Suppliers',**provenance},
             {'source_uid':item,'entity_type':'StockItem','category':'Packaging','brand':'Example',
              'source_table':'Warehouse.StockItems',**provenance}]
    native=nx.MultiDiGraph()
    for record in records:native.add_node(record['source_uid'],**record)
    native.add_node('order',entity_type='PurchaseOrder',date='2020-01-01',source_table='Purchasing.PurchaseOrders',**provenance)
    native.add_node('line',entity_type='PurchaseOrderLine',ordered=3,unit_price=7.5,source_table='Purchasing.PurchaseOrderLines',**provenance)
    native.add_node('movement',entity_type='StockMovement',date='2020-01-02',quantity=-2,source_table='Warehouse.StockItemTransactions',**provenance)
    native.add_node('finance',entity_type='SupplierTransaction',date='2020-01-03',amount=22.5,source_table='Purchasing.SupplierTransactions',**provenance)
    for a,b,key in [('line',item,'FOR_STOCK_ITEM'),('line','order','PART_OF_PO'),
                    ('order',supplier,'ORDERED_FROM'),('movement',item,'MOVES_STOCK_ITEM'),
                    ('finance',supplier,'RELATES_TO_SUPPLIER')]:native.add_edge(a,b,key=key)
    generated=tmp_path/'data/generated/seed_43';generated.mkdir(parents=True)
    gold=tmp_path/'data/gold/seed_43';gold.mkdir(parents=True)
    staged=tmp_path/'data/staged';staged.mkdir(parents=True)
    (generated/'records.json').write_text(json.dumps(records))
    with (generated/'graph.pkl').open('wb') as stream:pickle.dump(native,stream)
    (gold/'truth.json').write_text(json.dumps({supplier:{'canonical':'supplier'},item:{'canonical':'item','original_row':7}}))
    (staged/'Warehouse.StockItems.json').write_text(json.dumps([{'StockItemID':7,'QuantityPerOuter':10}]))
    monkeypatch.setattr(wwi,'ROOT',tmp_path)
    records,graph,_,_=wwi.build(43)
    by_uid={r['source_uid']:r for r in records}
    lines=[(u,a) for u,a in graph.nodes(data=True) if a['entity_type']=='PurchaseOrderLine']
    assert lines
    for uid,line in lines:
        item=next(b for _,b,k in graph.out_edges(uid,keys=True) if k=='FOR_STOCK_ITEM')
        supplier_projection=graph.nodes[uid+':transaction_context']
        stock_projection=graph.nodes[uid+':transaction_context:stock_units']
        assert supplier_projection['quantity_unit']=='purchase_outers'
        assert supplier_projection['quantity']==line['ordered']
        assert stock_projection['quantity_unit']=='stock_units'
        assert stock_projection['quantity']==line['ordered']*by_uid[item]['quantity_per_outer']
        assert supplier_projection['amount']==stock_projection['amount']==line['ordered']*line['unit_price']
        assert graph.has_edge(item,uid+':transaction_context:stock_units','HAS_TRANSACTION')
        assert not graph.has_edge(item,uid+':transaction_context','HAS_TRANSACTION')
        assert (line['ordered'],line['unit_price'])==(3,7.5)
    assert graph.nodes['movement:transaction_context']['quantity']==-2
    assert graph.nodes['finance:transaction_context']['quantity'] is None


def test_olist_projected_transaction_origin_is_its_item_file(monkeypatch,tmp_path):
    import json
    from kgerp.expanded import protocol
    folder=tmp_path/'data/expanded/Olist';folder.mkdir(parents=True)
    views=[]
    for side in ['A','B']:
        views.append({'name':'city','source_table':'olist_customers_dataset.csv',
            'transactions':[{'source_row':'Olist:items:'+side,'date':'2017-01-01','amount':10,'quantity':1,
                'tags':['product_category:example','seller_state:SP'],'activity':'delivered'}]})
    (folder/'base_views.json').write_text(json.dumps([{'canonical':'example','kind':'Customer','views':views}]))
    monkeypatch.setattr(protocol,'ROOT',tmp_path)
    records,graph,_,_=protocol.build_views('Olist',17,{'split_seed':1,'max_groups':1})
    assert all(r['source_table']=='olist_customers_dataset.csv' for r in records)
    transactions=[a for _,a in graph.nodes(data=True) if a['entity_type']=='Transaction']
    assert len(transactions)==2 and all(a['source_table']=='olist_order_items_dataset.csv' for a in transactions)


def test_no_anchors_ablation_retains_transaction_rows():
    records,graph=fixture();changed,view,_=perturb(records,graph,17,no_anchors=True)
    x,e=context([('A:Customer:0','B:Customer:0')],graph_profiles(view,changed))
    assert not e[0]['context_available'] and x[0,5]==1. and x[0,7]==1.


def test_context_rejects_cross_silo_transaction_keys():
    records,graph=fixture();graph.add_edge('A:Customer:0','B:tx:0',key='HAS_TRANSACTION')
    with pytest.raises(ValueError):graph_profiles(graph,records)
    with pytest.raises(ValueError):flat_profiles(graph,records)


def test_dates_keep_unknown_values_missing():
    assert day('10/2/2017 0:00')==day('2017-10-02T00:00:00')
    assert day('unknown') is None and day('') is None


def test_feature_modules_are_label_blind():
    root=Path(__file__).resolve().parents[1]/'src/kgerp/expanded'
    for name in ['attributes.py','features.py','features_wwi.py','embeddings.py']:
        tree=ast.parse((root/name).read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):assert not any(x in (node.module or '').split('.') for x in ['oracle','protocol','sources','wwi'])
            if isinstance(node,ast.Name):assert node.id not in {'gold','canonical_id','cluster_id','injection_log'}
