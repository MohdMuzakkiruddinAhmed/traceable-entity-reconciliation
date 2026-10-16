"""Evaluator-side full-source graph ingestion and source-safe view preparation."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import xml.etree.ElementTree as ET
import zipfile
import pandas as pd
from ..io import ROOT, write_json, write_rows


def clean(value):
    if value is None or pd.isna(value):
        return None
    return value.item() if hasattr(value, 'item') else value


def stable(text):
    return hashlib.sha256(str(text).encode()).hexdigest()


class SourceStore:
    """A typed property-graph store plus independent native relational tables."""
    def __init__(self, dataset):
        self.dataset = dataset
        self.folder = ROOT / 'data/expanded' / dataset
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / 'source.sqlite'
        if self.path.exists():
            raise FileExistsError('Source snapshot already exists: ' + str(self.path))
        self.db = sqlite3.connect(self.path)
        self.db.executescript('PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL; CREATE TABLE nodes(uid TEXT PRIMARY KEY,kind TEXT,source_table TEXT,source_row TEXT,attrs TEXT); CREATE TABLE edges(a TEXT,b TEXT,relation TEXT,PRIMARY KEY(a,b,relation));')
        self.profiles = []
        self.references = []

    def table(self, name, kind, frame, keys):
        frame = frame.copy()
        keys = [keys] if isinstance(keys, str) else keys
        rowkeys = frame[keys].astype(str).agg(':'.join, axis=1)
        frame['__uid'] = self.dataset + ':' + name + ':' + rowkeys
        if frame['__uid'].duplicated().any():
            raise ValueError('Duplicate source PK: ' + name)
        frame.to_sql('raw_' + name, self.db, index=False)
        self.db.execute(f'CREATE UNIQUE INDEX idx_{name}_uid ON raw_{name}(__uid)')
        self.nodes(name, kind, frame, rowkeys.tolist())
        self.profiles.append({'dataset': self.dataset, 'table': name, 'rows': len(frame), 'columns': len(frame.columns)-1, 'null_cells': int(frame.drop(columns='__uid').isna().sum().sum())})
        return frame

    def nodes(self, name, kind, frame, rowkeys):
        batch = []
        for row, key in zip(frame.to_dict('records'), rowkeys):
            uid = row.pop('__uid')
            # Original fields live only in the source snapshot. Inference views
            # whitelist descriptive properties and independently remap keys.
            attrs = {k: clean(v) for k, v in row.items() if k not in {'Customer Password', 'Customer Email'}}
            batch.append((uid, kind, name, str(key), json.dumps(attrs, ensure_ascii=False, allow_nan=False)))
            if len(batch) == 10000:
                self.db.executemany('INSERT INTO nodes VALUES (?,?,?,?,?)', batch)
                batch = []
        self.db.executemany('INSERT INTO nodes VALUES (?,?,?,?,?)', batch)
        self.db.commit()

    def relation(self, left, right, left_keys, right_keys, relation):
        left_keys = [left_keys] if isinstance(left_keys, str) else left_keys
        right_keys = [right_keys] if isinstance(right_keys, str) else right_keys
        lookup = {tuple(clean(row[k]) for k in right_keys): row['__uid'] for row in right.to_dict('records')}
        links = []
        for row in left.to_dict('records'):
            key = tuple(clean(row[k]) for k in left_keys)
            if key in lookup:
                links.append((row['__uid'], lookup[key], relation))
        self.db.executemany('INSERT OR IGNORE INTO edges VALUES (?,?,?)', links)
        lt, rt = left['__uid'].iloc[0].split(':')[1], right['__uid'].iloc[0].split(':')[1]
        condition = ' AND '.join(f'a."{a}"=b."{b}"' for a, b in zip(left_keys, right_keys))
        self.references.append((relation, f'SELECT a.__uid,b.__uid FROM raw_{lt} a JOIN raw_{rt} b ON {condition}'))
        self.db.commit()

    def finish(self, out, seconds, extra=None):
        self.db.executescript('CREATE INDEX edge_a ON edges(a,relation); CREATE INDEX edge_b ON edges(b,relation); CREATE INDEX node_kind ON nodes(kind);')
        fidelity = []
        for relation, query in self.references:
            expected = f"SELECT DISTINCT a,b,'{relation}' AS relation FROM ({query.replace('a.__uid,b.__uid', 'a.__uid AS a,b.__uid AS b')})"
            count = self.db.execute('SELECT COUNT(*) FROM (' + expected + ')').fetchone()[0]
            missing = self.db.execute('SELECT COUNT(*) FROM (' + expected + ' EXCEPT SELECT a,b,relation FROM edges)').fetchone()[0]
            extra_edges = self.db.execute(f"SELECT COUNT(*) FROM (SELECT a,b,relation FROM edges WHERE relation='{relation}' EXCEPT {expected})").fetchone()[0]
            fidelity.append({'dataset': self.dataset, 'relation': relation, 'expected_edges': count, 'missing_edges': missing, 'extra_edges': extra_edges, 'recall': (count-missing)/max(1,count)})
        dangling = self.db.execute('SELECT COUNT(*) FROM edges e LEFT JOIN nodes a ON e.a=a.uid LEFT JOIN nodes b ON e.b=b.uid WHERE a.uid IS NULL OR b.uid IS NULL').fetchone()[0]
        lineage = self.db.execute("SELECT COUNT(*) FROM nodes WHERE source_table='' OR source_row='' OR attrs IS NULL").fetchone()[0]
        before = self.db.total_changes
        self.db.execute('INSERT OR IGNORE INTO edges SELECT * FROM edges')
        self.db.commit()
        idempotent = self.db.total_changes == before
        summary = {'dataset': self.dataset, 'nodes': self.db.execute('SELECT COUNT(*) FROM nodes').fetchone()[0],
                   'edges': self.db.execute('SELECT COUNT(*) FROM edges').fetchone()[0], 'dangling_edges': dangling,
                   'lineage_missing': lineage, 'lineage_coverage': 1.0 if not lineage else None,
                   'all_reference_sets_exact': all(not r['missing_edges'] and not r['extra_edges'] for r in fidelity),
                   'edge_import_idempotent': idempotent, 'build_seconds': seconds,
                   'representation': 'disk SQLite typed property graph; independent native relational FK joins', **(extra or {})}
        if dangling or lineage or not summary['all_reference_sets_exact'] or not idempotent:
            raise ValueError('Expanded E1 source integrity failure')
        out.mkdir(parents=True, exist_ok=True)
        write_json(out/'e1.json', summary)
        write_rows(out/'e1_relationships.csv', fidelity)
        write_rows(out/'source_profile.csv', self.profiles)
        write_json(self.folder/'source_manifest.json', summary)
        self.db.close()


def tx(date, amount, quantity, tags, activity, row):
    return {'date': str(date or ''), 'amount': clean(amount), 'quantity': clean(quantity),
            'tags': sorted(set(str(x) for x in tags if x)), 'activity': str(activity or ''), 'source_row': str(row)}


def olist(out):
    start = time.perf_counter()
    store = SourceStore('Olist')
    with zipfile.ZipFile(ROOT/'data/raw/olist-v2.zip') as archive:
        raw = {name: pd.read_csv(archive.open(name), low_memory=False) for name in archive.namelist() if name.endswith('.csv')}
    specs = [('customers','Customer','olist_customers_dataset.csv','customer_id'),('orders','Order','olist_orders_dataset.csv','order_id'),
             ('items','OrderItem','olist_order_items_dataset.csv',['order_id','order_item_id']),('payments','Payment','olist_order_payments_dataset.csv',['order_id','payment_sequential']),
             ('products','Product','olist_products_dataset.csv','product_id'),('sellers','Seller','olist_sellers_dataset.csv','seller_id'),
             ('reviews','Review','olist_order_reviews_dataset.csv','__row'),('geolocation','GeoObservation','olist_geolocation_dataset.csv','__row'),
             ('categories','Category','product_category_name_translation.csv','product_category_name')]
    tables = {}
    for name, kind, filename, keys in specs:
        frame = raw[filename]
        if keys == '__row':
            frame = frame.copy();frame['__row'] = range(len(frame))
        tables[name] = store.table(name, kind, frame, keys)
        print('Olist imported ' + name, flush=True)
    for a,b,ak,bk,rel in [('orders','customers','customer_id','customer_id','ORDER_CUSTOMER'),('items','orders','order_id','order_id','ITEM_ORDER'),
                          ('items','products','product_id','product_id','ITEM_PRODUCT'),('items','sellers','seller_id','seller_id','ITEM_SELLER'),
                          ('payments','orders','order_id','order_id','PAYMENT_ORDER'),('reviews','orders','order_id','order_id','REVIEW_ORDER'),
                          ('products','categories','product_category_name','product_category_name','PRODUCT_CATEGORY')]:
        store.relation(tables[a], tables[b], ak, bk, rel)
    # Zip prefixes are a grouping relation, not an invented single geolocation FK.
    zips = pd.DataFrame({'zip': sorted(set(tables['geolocation']['geolocation_zip_code_prefix']))})
    tables['zips'] = store.table('zips','PostalPrefix',zips,'zip')
    for a,key,rel in [('geolocation','geolocation_zip_code_prefix','GEO_POSTAL'),('customers','customer_zip_code_prefix','CUSTOMER_POSTAL'),('sellers','seller_zip_code_prefix','SELLER_POSTAL')]:
        store.relation(tables[a], tables['zips'], key, 'zip', rel)
    customer_rows = tables['customers'].set_index('customer_id').to_dict('index')
    product_categories = tables['products'].set_index('product_id')['product_category_name'].fillna('').to_dict()
    seller_states = tables['sellers'].set_index('seller_id')['seller_state'].fillna('').to_dict()
    orders = tables['orders'].set_index('order_id').to_dict('index')
    transactions = defaultdict(list)
    for row in tables['items'].to_dict('records'):
        order = orders[row['order_id']]
        transactions[order['customer_id']].append(tx(order['order_purchase_timestamp'],row['price'],1,
            ['product_category:' + str(product_categories.get(row['product_id']) or ''),'seller_state:'+str(seller_states.get(row['seller_id']) or '')],order['order_status'],row['__uid']))
    bases = []
    for canonical, group in tables['customers'].groupby('customer_unique_id', sort=True):
        if len(group) < 2:
            continue
        eligible = [r for r in group.to_dict('records') if transactions[r['customer_id']]]
        if len(eligible) < 2:
            continue
        eligible.sort(key=lambda r: stable(r['customer_id']))
        views = []
        for row in eligible[:2]:
            views.append({'name': str(row['customer_city']), 'category': str(row['customer_zip_code_prefix']), 'brand': str(row['customer_state']),
                          'description': str(row['customer_city'])+' '+str(row['customer_state']), 'price':'','currency':'',
                          'transactions': transactions[row['customer_id']], 'source_table':'olist_customers_dataset.csv', 'original_row':row['customer_id']})
        bases.append({'canonical':canonical,'kind':'Customer','views':views})
    write_json(store.folder/'base_views.json', bases)
    store.finish(out,time.perf_counter()-start,{'identity_truth':'natural customer_unique_id, evaluator-only; two distinct observed accounts per eligible repeat customer',
               'eligible_repeat_groups':len(bases),'source_tables':9,'geolocation_semantics':'separate observations and postal-prefix grouping nodes; no unique-coordinate assumption'})


def dataco(out):
    start = time.perf_counter();store = SourceStore('DataCo')
    raw = pd.read_csv(ROOT/'data/raw/DataCoSupplyChainDataset.csv',encoding='latin-1',low_memory=False)
    if raw.shape != (180519,53):
        raise ValueError('Unexpected DataCo mirror schema/count')
    raw['__row'] = range(len(raw))
    tables = {}
    specs = [
        ('customers','Customer','Customer Id',['Customer Id','Customer Fname','Customer Lname','Customer City','Customer State','Customer Country','Customer Segment']),
        ('orders','Order','Order Id',['Order Id','Order Customer Id','order date (DateOrders)','shipping date (DateOrders)','Order Status','Shipping Mode']),
        ('products','Product','Product Card Id',['Product Card Id','Product Name','Product Category Id','Product Price']),
        ('categories','Category','Category Id',['Category Id','Category Name','Department Id']),
        ('departments','Department','Department Id',['Department Id','Department Name'])]
    conflicts = []
    for name,kind,key,columns in specs:
        distinct = raw[columns].drop_duplicates()
        conflicts.append({'table':name,'identity_key':key,'groups_with_attribute_conflicts':int((distinct.groupby(key).size()>1).sum())})
        tables[name] = store.table(name,kind,raw[columns].drop_duplicates(key,keep='first'),key)
    # Every source row remains an observation node, preserving conflicting facts.
    tables['items'] = store.table('items','OrderItem',raw,'Order Item Id')
    for a,b,ak,bk,rel in [('items','orders','Order Id','Order Id','ITEM_ORDER'),('items','products','Order Item Cardprod Id','Product Card Id','ITEM_PRODUCT'),
                          ('items','customers','Customer Id','Customer Id','ITEM_CUSTOMER'),('orders','customers','Order Customer Id','Customer Id','ORDER_CUSTOMER'),
                          ('products','categories','Product Category Id','Category Id','PRODUCT_CATEGORY'),('categories','departments','Department Id','Department Id','CATEGORY_DEPARTMENT')]:
        store.relation(tables[a],tables[b],ak,bk,rel)
    per_customer = defaultdict(lambda: [[],[]])
    for row in raw.to_dict('records'):
        side = int(stable(row['Order Id'])[:8],16)%2
        per_customer[row['Customer Id']][side].append(tx(row['order date (DateOrders)'],row['Order Item Total'],row['Order Item Quantity'],
            ['product_category:'+str(row['Category Name']), 'department:'+str(row['Department Name'])],row['Delivery Status'],row['Order Item Id']))
    bases = []
    for row in tables['customers'].to_dict('records'):
        canonical = str(row['Customer Id'])
        histories = per_customer[row['Customer Id']]
        if not all(histories):
            continue
        attrs = {'name':str(row['Customer Fname'] or '')+' '+str(clean(row['Customer Lname']) or ''),'category':str(row['Customer Segment']),
                 'brand':str(row['Customer Country']),'description':str(row['Customer City'])+' '+str(row['Customer State']),'price':'','currency':'',
                 'source_table':'DataCoSupplyChainDataset.csv','original_row':canonical}
        bases.append({'canonical':canonical,'kind':'Customer','views':[{**attrs,'transactions':h} for h in histories]})
    write_json(store.folder/'base_views.json',bases)
    write_rows(out/'normalization_conflicts.csv',conflicts)
    store.finish(out,time.perf_counter()-start,{'identity_truth':'controlled aliases grouped by Customer Id; no independently annotated dirty-source matching labels',
               'eligible_groups':len(bases),'normalization':'first source occurrence represents master attributes; all row observations retained; conflicts audited',
               'download_limit':'documented third-party mirror; publisher acquisition 403; publisher byte equality not established'})


def xes_attrs(element):
    return {child.attrib['key']:child.attrib.get('value','') for child in element if child.tag.rsplit('}',1)[-1] != 'event' and 'key' in child.attrib}


def bpi(out):
    start=time.perf_counter();store=SourceStore('BPI2019')
    db=store.db
    db.executescript('CREATE TABLE raw_cases(__uid TEXT PRIMARY KEY,case_id TEXT,po TEXT,vendor TEXT,company TEXT,category TEXT,attrs TEXT); CREATE TABLE raw_events(__uid TEXT PRIMARY KEY,case_uid TEXT,ordinal INTEGER,activity TEXT,resource TEXT,date TEXT,amount REAL,attrs TEXT);')
    node_batch=[];edge_batch=[];case_batch=[];event_batch=[]
    distinct=defaultdict(set);histories=defaultdict(lambda:[[],[]]);attributes=defaultdict(lambda:[Counter(),Counter()])
    cases=events=0;root=None
    case_columns=set();event_columns=set();case_nulls=event_nulls=0
    def flush():
        if node_batch:
            db.executemany('INSERT INTO nodes VALUES (?,?,?,?,?)',node_batch)
            db.executemany('INSERT INTO raw_cases VALUES (?,?,?,?,?,?,?)',case_batch)
            db.executemany('INSERT INTO raw_events VALUES (?,?,?,?,?,?,?,?)',event_batch)
            db.executemany('INSERT OR IGNORE INTO edges VALUES (?,?,?)',edge_batch)
            db.commit();node_batch.clear();edge_batch.clear();case_batch.clear();event_batch.clear()
    for event,element in ET.iterparse(ROOT/'data/raw/BPI_Challenge_2019.xes',events=['start','end']):
        if root is None:root=element
        if event!='end' or element.tag.rsplit('}',1)[-1]!='trace':continue
        attrs=xes_attrs(element);case_id=attrs['concept:name'];case_uid='BPI2019:cases:'+case_id
        case_columns.update(attrs);case_nulls+=sum(v in ('','NONE') for v in attrs.values())
        po=attrs.get('Purchasing Document','');vendor=attrs.get('Vendor','');company=attrs.get('Company','');category=attrs.get('Item Category','')
        node_batch.append((case_uid,'PurchaseItem','cases',case_id,json.dumps(attrs)))
        case_batch.append((case_uid,case_id,po,vendor,company,category,json.dumps(attrs)))
        for relation,table,key in [('CASE_PO','purchase_orders',po),('CASE_VENDOR','vendors',vendor),('CASE_COMPANY','companies',company),('CASE_CATEGORY','categories',category)]:
            if key:distinct[table].add(key);edge_batch.append((case_uid,'BPI2019:'+table+':'+key,relation))
        evs=[child for child in element if child.tag.rsplit('}',1)[-1]=='event']
        previous=None;activity_counts=Counter();first_amount=None;first_date=''
        for ordinal,ev in enumerate(evs):
            e=xes_attrs(ev);uid=case_uid+':event:'+str(ordinal);activity=e.get('concept:name','');resource=e.get('org:resource','');date=e.get('time:timestamp','')
            event_columns.update(e);event_nulls+=sum(v in ('','NONE') for v in e.values())
            try:amount=float(e['Cumulative net worth (EUR)'])
            except (KeyError,ValueError):amount=None
            node_batch.append((uid,'Event','events',case_id+':'+str(ordinal),json.dumps(e)))
            event_batch.append((uid,case_uid,ordinal,activity,resource,date,amount,json.dumps(e)))
            edge_batch.append((uid,case_uid,'EVENT_CASE'))
            if previous:edge_batch.append((previous,uid,'DIRECTLY_FOLLOWS'))
            previous=uid
            if activity:distinct['activities'].add(activity);edge_batch.append((uid,'BPI2019:activities:'+activity,'EVENT_ACTIVITY'));activity_counts[activity]+=1
            if resource:distinct['resources'].add(resource);edge_batch.append((uid,'BPI2019:resources:'+resource,'EVENT_RESOURCE'))
            if ordinal==0:first_amount=amount;first_date=date
        side=int(stable(case_id)[:8],16)%2
        if vendor:
            tags=['spend_area:'+attrs.get('Spend area text',''),'sub_spend_area:'+attrs.get('Sub spend area text',''),'item_category:'+category]
            transaction=tx(first_date,first_amount,None,tags,category,case_id)
            transaction['activities']=dict(activity_counts)
            histories[vendor][side].append(transaction)
            attributes[vendor][side].update([(attrs.get('Spend area text',''),attrs.get('Document Type',''),attrs.get('Spend classification text',''))])
        cases+=1;events+=len(evs)
        if len(node_batch)>=10000:flush()
        element.clear();root.clear()
        if cases%50000==0:print(f'BPI streamed {cases:,} cases / {events:,} events',flush=True)
    flush()
    for table,keys in distinct.items():
        kind={'purchase_orders':'PurchaseOrder','vendors':'Vendor','companies':'Company','categories':'ItemCategory','activities':'Activity','resources':'Resource'}[table]
        frame=pd.DataFrame({'key':sorted(keys)})
        store.table(table,kind,frame,'key')
    # Independent SQL derives structural relations from native case/event fields.
    store.references=[('EVENT_CASE','SELECT a.__uid,b.__uid FROM raw_events a JOIN raw_cases b ON a.case_uid=b.__uid')]
    for rel,column,table in [('CASE_PO','po','purchase_orders'),('CASE_VENDOR','vendor','vendors'),('CASE_COMPANY','company','companies'),('CASE_CATEGORY','category','categories')]:
        store.references.append((rel,f'SELECT a.__uid,b.__uid FROM raw_cases a JOIN raw_{table} b ON a.{column}=b.key'))
    for rel,column,table in [('EVENT_ACTIVITY','activity','activities'),('EVENT_RESOURCE','resource','resources')]:
        store.references.append((rel,f'SELECT a.__uid,b.__uid FROM raw_events a JOIN raw_{table} b ON a.{column}=b.key'))
    db.executescript('CREATE INDEX event_case_ordinal ON raw_events(case_uid,ordinal); CREATE INDEX case_vendor ON raw_cases(vendor);')
    store.references.append(('DIRECTLY_FOLLOWS','SELECT a.__uid,b.__uid FROM raw_events a JOIN raw_events b ON a.case_uid=b.case_uid AND b.ordinal=a.ordinal+1'))
    bases=[]
    for vendor,views in sorted(histories.items()):
        if not all(views):continue
        attrs=[]
        for side,history in enumerate(views):
            mode=attributes[vendor][side].most_common(1)[0][0]
            attrs.append({'name':mode[0], 'category':mode[1], 'brand':mode[2], 'description':mode[0]+' '+mode[1], 'price':'','currency':'EUR',
                          'transactions':history,'source_table':'BPI_Challenge_2019.xes','original_row':vendor})
        # Vendor Name and anonymized vendor ID are deliberately never attributes.
        bases.append({'canonical':vendor,'kind':'Vendor','views':attrs})
    write_json(store.folder/'base_views.json',bases)
    store.profiles.extend([{'dataset':'BPI2019','table':'cases','rows':cases,'columns':len(case_columns),'null_cells':case_nulls}, {'dataset':'BPI2019','table':'events','rows':events,'columns':len(event_columns),'null_cells':event_nulls}])
    if (cases,events)!=(251734,1595923):raise ValueError('BPI publisher count mismatch')
    store.finish(out,time.perf_counter()-start,{'identity_truth':'controlled vendor views; separate hashed purchase cases per silo; no annotated dirty-identity labels',
        'eligible_vendor_groups':len(bases),'vendor_names_excluded':True,'amount_semantics':'first-event anonymized cumulative net worth per case, not sum of cumulative event values or actual paid spend',
        'quantity_semantics':'not provided; never fabricated','ordering':'original trace ordinal; timestamp inversions/ties retained as source observations'})


def prepare(dataset,out):
    folder=ROOT/'data/expanded'/dataset
    if (folder/'source_manifest.json').exists():
        # Reuse a completed input snapshot, not partially imported data.
        write_json(out/'e1.json',json.loads((folder/'source_manifest.json').read_text()))
        return
    {'Olist':olist,'DataCo':dataco,'BPI2019':bpi}[dataset](out)


def enrich_olist_cache():
    """Add an authentic seller-location path to the earlier prepared fixture."""
    folder=ROOT/'data/expanded/Olist';path=folder/'base_views.json'
    bases=json.loads(path.read_text(encoding='utf-8'))
    if all(any(t.startswith('seller_state:') for t in tx['tags']) for row in bases for view in row['views'] for tx in view['transactions']):return
    # Preserve the diagnostic fixture before the pre-training adapter expansion.
    old=folder/'base_views_minimal_profile_v1.json'
    if not old.exists():old.write_bytes(path.read_bytes())
    with sqlite3.connect(folder/'source.sqlite') as db:
        states=dict(db.execute('SELECT i.__uid,s.seller_state FROM raw_items i JOIN raw_sellers s ON i.seller_id=s.seller_id'))
    for row in bases:
        for view in row['views']:
            for transaction in view['transactions']:
                state=states.get(transaction['source_row'])
                if state:transaction['tags']=sorted(set(transaction['tags']+['seller_state:'+state]))
    write_json(path,bases)
    write_json(folder/'view_adapter_amendment.json',{'classification':'exploratory before first expanded model fitting, after diagnostic test-context availability review',
        'reason':'minimal Olist product-category profiles supplied at most one shared support tag; include authentic order-item-to-seller-state path',
        'changed':'source-supported seller state added to category summaries; no new labels, altered groups, thresholds or copied cross-silo keys',
        'minimal_fixture_sha256':stable(old.read_text(encoding='utf-8')),'expanded_fixture_sha256':stable(path.read_text(encoding='utf-8'))})
