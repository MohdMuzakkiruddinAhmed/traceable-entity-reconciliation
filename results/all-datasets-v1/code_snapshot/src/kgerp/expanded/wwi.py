"""Evaluator/source preparation for richer WWI transaction context."""
from collections import defaultdict
import json
import pickle
from ..io import ROOT
from .sources import stable


def build(seed):
    records=json.loads((ROOT/f'data/generated/seed_{seed}/records.json').read_text())
    gold=json.loads((ROOT/f'data/gold/seed_{seed}/truth.json').read_text())
    graph=pickle.load((ROOT/f'data/generated/seed_{seed}/graph.pkl').open('rb'))
    stocks=json.loads((ROOT/'data/staged/Warehouse.StockItems.json').read_text())
    by_pk={r['StockItemID']:r for r in stocks}
    # Source mapping is used only here, just as in the original alias generator,
    # to expose a business attribute. No original PK/canonical ID enters records.
    for r in records:
        if r['entity_type']=='StockItem':r['quantity_per_outer']=by_pk[gold[r['source_uid']]['original_row']]['QuantityPerOuter']
        graph.nodes[r['source_uid']].update(r)
    outgoing=defaultdict(dict)
    for a,b,k in graph.edges(keys=True):outgoing[a][k]=b
    by_uid={r['source_uid']:r for r in records}
    tag_nodes={}
    def add(owner,original,date,amount,quantity,q_unit,activity,category_record):
        source=graph.nodes[original];uid=original+':transaction_context'
        attrs={'source_uid':uid,'entity_type':'Transaction','silo':source['silo'],'date':date or '', 'amount':amount,'quantity':quantity,
            'amount_unit':'WWI_source_monetary_units','quantity_unit':q_unit,'activity':activity,
            'source_table':source['source_table'],'source_batch':source['source_batch'],'mapper_version':source['mapper_version'],
            'source_row_digest':stable(original)}
        graph.add_node(uid,**attrs);graph.add_edge(owner,uid,key='HAS_TRANSACTION',relation='HAS_TRANSACTION')
        for field in ['category','brand']:
            label=str(category_record.get(field) or '')
            if not label:continue
            text=field+':'+label;key=(source['silo'],text)
            if key not in tag_nodes:
                tid=source['silo']+':context_category:'+stable(text)[:24];tag_nodes[key]=tid
                graph.add_node(tid,entity_type='ContextCategory',label=text,silo=source['silo'],source_table='visible_master_category',source_batch=source['source_batch'],mapper_version='expanded-v4')
            graph.add_edge(uid,tag_nodes[key],key='HAS_CATEGORY',relation='HAS_CATEGORY')
    for uid,attrs in list(graph.nodes(data=True)):
        kind=attrs['entity_type']
        if kind=='PurchaseOrderLine':
            item=outgoing[uid]['FOR_STOCK_ITEM'];po=outgoing[uid]['PART_OF_PO'];supplier=outgoing[po]['ORDERED_FROM']
            date=graph.nodes[po]['date'];amount=attrs['ordered']*attrs['unit_price']
            add(supplier,uid,date,amount,attrs['ordered'],'purchase_outers','purchase_line',by_uid[item])
            # Valid package conversion; not a comparison of outers to stock units.
            add(item,uid,date,amount,attrs['ordered']*by_uid[item]['quantity_per_outer'],'stock_units','purchase_line',by_uid[item])
            # The same local line can describe its supplier and stock item.
            # Use separate projections because their quantity units differ.
            # Reassign the item edge to an item-specific projection.
            projection=uid+':transaction_context';item_projection=projection+':stock_units'
            graph.add_node(item_projection,**{**graph.nodes[projection],'source_uid':item_projection})
            graph.remove_edge(item,projection,'HAS_TRANSACTION');graph.add_edge(item,item_projection,key='HAS_TRANSACTION',relation='HAS_TRANSACTION')
            for _,tag,k in list(graph.out_edges(projection,keys=True)):
                if k=='HAS_CATEGORY':graph.add_edge(item_projection,tag,key=k,relation=k)
            graph.nodes[projection]['quantity']=attrs['ordered'];graph.nodes[projection]['quantity_unit']='purchase_outers'
        elif kind=='StockMovement':
            item=outgoing[uid]['MOVES_STOCK_ITEM'];add(item,uid,attrs['date'],None,attrs['quantity'],'stock_units','stock_movement',by_uid[item])
        elif kind=='SupplierTransaction':
            supplier=outgoing[uid]['RELATES_TO_SUPPLIER'];add(supplier,uid,attrs['date'],attrs['amount'],None,'','supplier_transaction',by_uid[supplier])
    manifest={'dataset':'WWI','selected_groups':240,'eligible_source_groups':240,'group_counts':{'train':143,'validation':48,'test':49},
        'source_run':'initial-core-v3','original_splits_aliases_and_native_transactions_unchanged':True,
        'quantity_conversion':'source QuantityPerOuter converts purchased stock-item outers to stock units; supplier-level outers remain distinct',
        'quantity_per_outer_baseline':'exposed as additional attribute compatibility features for every learned attribute/context comparator',
        'context_gate':'original independently inferred master anchors; category summaries from candidate endpoints do not establish independent anchor support',
        'lineage':'native keys, transactions and relationships preserved; packaging attribute exposed on source master views; separate transaction projections added'}
    return records,graph,gold,manifest
