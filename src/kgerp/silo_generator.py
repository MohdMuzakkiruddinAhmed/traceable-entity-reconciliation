"""Controlled source-silo emulation; original keys leave this module only as gold.

Master views are perturbed independently. Transactions are never cloned between
silos. The raw reference graph is not passed to any feature extractor.
"""
from collections import defaultdict
import random
import math
import uuid
import networkx as nx
from .io import ROOT, write_json

def opaque(rng):
    return uuid.UUID(int=rng.getrandbits(128)).hex

def split_groups(tables, seed=20261011):
    rng = random.Random(seed)
    groups = {}
    for table, pk in [('Purchasing.Suppliers', 'SupplierID'), ('Warehouse.StockItems', 'StockItemID')]:
        ids = sorted(row[pk] for row in tables[table])
        if table == 'Purchasing.Suppliers':
            # Source-visible activity stratification avoids empty query tasks.
            # This is fixed before aliases, training or evaluation.
            active_ids = {r['SupplierID'] for r in tables['Purchasing.PurchaseOrders']} & {r['SupplierID'] for r in tables['Warehouse.StockItems']}
            active = [i for i in ids if i in active_ids]
            inactive = [i for i in ids if i not in active_ids]
            rng.shuffle(active);rng.shuffle(inactive)
            train_count=int(len(ids)*.6);validation_count=int(len(ids)*.8)-train_count;test_count=len(ids)-train_count-validation_count
            active_test=min(test_count,math.ceil(len(active)*.2))
            active_validation=min(validation_count,math.floor(len(active)*.2))
            active_train=len(active)-active_test-active_validation
            if active_train>train_count:
                overflow=active_train-train_count;active_validation+=overflow;active_train-=overflow
            partitions=[('train',active_train,train_count),('validation',active_validation,validation_count),('test',active_test,test_count)]
            start_active=start_inactive=0
            for split,n_active,n_total in partitions:
                chosen=active[start_active:start_active+n_active]+inactive[start_inactive:start_inactive+n_total-n_active]
                for value in chosen:groups[(table,value)]=split
                start_active+=n_active;start_inactive+=n_total-n_active
            if len([g for g in groups if g[0]==table])!=len(ids):raise ValueError('Invalid activity-stratified supplier allocation')
            continue
        rng.shuffle(ids)
        cut1, cut2 = int(len(ids) * .6), int(len(ids) * .8)
        for i, value in enumerate(ids):
            groups[(table, value)] = 'train' if i < cut1 else 'validation' if i < cut2 else 'test'
    return groups

def corrupt_name(name, rng, config):
    words = name.split()
    if rng.random() < config['abbreviation_probability'] and len(words) > 1:
        words[0] = words[0][:3]
    value = ' '.join(words)
    if rng.random() < config['typo_probability'] and len(value) > 3:
        i = rng.randrange(1, len(value) - 1)
        value = value[:i] + value[i + 1] + value[i] + value[i + 2:]
    return value

def generate(tables, seed, config, split_seed=20261011):
    rng = random.Random(seed)
    groups = split_groups(tables, split_seed)
    graph = nx.MultiDiGraph()
    records, gold, lookup = [], {}, {}
    supplier_rows = {r['SupplierID']: r for r in tables['Purchasing.Suppliers']}
    for silo in ('A', 'B'):
        for table, kind, pk, name in [('Purchasing.Suppliers','Supplier','SupplierID','SupplierName'),
                                       ('Warehouse.StockItems','StockItem','StockItemID','StockItemName')]:
            for row in tables[table]:
                source_uid = f'{silo}:{kind}:{opaque(rng)}'
                category = str(row.get('SupplierCategoryID', '')) if kind == 'Supplier' else str(row.get('UnitPackageID', ''))
                record = {'source_uid': source_uid, 'silo': silo, 'entity_type': kind,
                          'split': groups[(table, row[pk])], 'name': corrupt_name(row[name], rng, config),
                          'category': '' if rng.random() < config['optional_missingness'] else category,
                          'brand': '' if rng.random() < config['optional_missingness'] else str(row.get('Brand') or ''),
                          'description': str(row.get('MarketingComments') or ''),
                          'price': str(row.get('UnitPrice') or ''), 'currency': '',
                          'source_table': table, 'source_batch': f'silo-emulation-seed-{seed}', 'mapper_version': '1'}
                records.append(record)
                graph.add_node(source_uid, **record)
                lookup[(silo, table, row[pk])] = source_uid
                gold[source_uid] = {'canonical': f'{kind}:{row[pk]}', 'split': record['split'],
                                    'original_table': table, 'original_row': row[pk]}
        for row in tables['Warehouse.StockItems']:
            if rng.random() >= config['edge_drop']:
                graph.add_edge(lookup[(silo,'Warehouse.StockItems',row['StockItemID'])],
                               lookup[(silo,'Purchasing.Suppliers',row['SupplierID'])], key='FROM_SUPPLIER', relation='FROM_SUPPLIER')

    def transaction(row, table, kind, pk, silo, attrs):
        source_uid = f'{silo}:{kind}:{opaque(rng)}'
        graph.add_node(source_uid, source_uid=source_uid, entity_type=kind, silo=silo,
                       source_table=table, source_batch=f'silo-emulation-seed-{seed}', mapper_version='1', **attrs)
        gold[source_uid] = {'canonical': f'{kind}:{row[pk]}', 'original_table': table, 'original_row': row[pk]}
        lookup[(silo, table, row[pk])] = source_uid
        return source_uid

    for row in tables['Purchasing.PurchaseOrders']:
        po = transaction(row,'Purchasing.PurchaseOrders','PurchaseOrder','PurchaseOrderID','A', {'date':row['OrderDate']})
        graph.add_edge(po, lookup[('A','Purchasing.Suppliers',row['SupplierID'])], key='ORDERED_FROM', relation='ORDERED_FROM')
    for row in tables['Purchasing.PurchaseOrderLines']:
        line = transaction(row,'Purchasing.PurchaseOrderLines','PurchaseOrderLine','PurchaseOrderLineID','A',
                           {'ordered':row['OrderedOuters'], 'received':row['ReceivedOuters'], 'unit_price':row['ExpectedUnitPricePerOuter']})
        graph.add_edge(line, lookup[('A','Purchasing.PurchaseOrders',row['PurchaseOrderID'])], key='PART_OF_PO', relation='PART_OF_PO')
        graph.add_edge(line, lookup[('A','Warehouse.StockItems',row['StockItemID'])], key='FOR_STOCK_ITEM', relation='FOR_STOCK_ITEM')
    # Source-specific transaction subsets are bounded independently of labels.
    for row in tables['Warehouse.StockItemTransactions'][:5000]:
        move = transaction(row,'Warehouse.StockItemTransactions','StockMovement','StockItemTransactionID','B',
                           {'quantity':row['Quantity'], 'date':row['TransactionOccurredWhen']})
        graph.add_edge(move, lookup[('B','Warehouse.StockItems',row['StockItemID'])], key='MOVES_STOCK_ITEM', relation='MOVES_STOCK_ITEM')
    for row in tables['Purchasing.SupplierTransactions']:
        tx = transaction(row,'Purchasing.SupplierTransactions','SupplierTransaction','SupplierTransactionID','B',
                         {'amount':row['TransactionAmount'], 'date':row['TransactionDate']})
        graph.add_edge(tx, lookup[('B','Purchasing.Suppliers',row['SupplierID'])], key='RELATES_TO_SUPPLIER', relation='RELATES_TO_SUPPLIER')
    return records, graph, gold

def save(records, graph, gold, seed):
    import pickle
    write_json(ROOT / f'data/generated/seed_{seed}/records.json', records)
    write_json(ROOT / f'data/gold/seed_{seed}/truth.json', gold)
    path = ROOT / f'data/generated/seed_{seed}/graph.pkl'
    with path.open('wb') as handle:
        pickle.dump(graph, handle)
    summary = []
    for split in ('train','validation','test'):
        for kind in ('Supplier','StockItem'):
            selected = [r for r in records if r['split'] == split and r['entity_type'] == kind]
            summary.append({'split':split, 'entity_type':kind, 'source_records':len(selected),
                            'independent_entities':len({gold[r['source_uid']]['canonical'] for r in selected})})
    write_json(ROOT / f'data/manifests/silos_{seed}.json', {'seed':seed,'groups':summary,
               'stock_movement_scope':'first 5000 original PK-sorted rows; E1 covers all source rows',
               'finance_scope':'all supplier transactions in source B; no copied PO keys',
               'context_scope':'same-split master nodes and FROM_SUPPLIER edges only'})
