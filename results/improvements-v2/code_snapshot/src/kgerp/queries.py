"""Six frozen source-UID answer templates, independent SQL/graph execution."""
from collections import defaultdict
import sqlite3

SQL = {
 'Q1':"""SELECT DISTINCT s.uid,w.uid FROM nodes p
 JOIN edges e ON e.a=p.uid AND e.relation='ORDERED_FROM' JOIN nodes s ON s.uid=e.b
 JOIN edges lp ON lp.b=p.uid AND lp.relation='PART_OF_PO'
 JOIN edges li ON li.a=lp.a AND li.relation='FOR_STOCK_ITEM'
 JOIN identities ia ON ia.uid=li.b JOIN identities iw ON iw.canonical=ia.canonical
 JOIN nodes w ON w.uid=iw.uid AND w.kind='StockItem' AND w.silo='B'
 WHERE s.split='test' AND w.split='test'""",
 'Q2':"""SELECT DISTINCT l.uid FROM nodes l JOIN edges e ON e.a=l.uid AND e.relation='FOR_STOCK_ITEM'
 JOIN nodes i ON i.uid=e.b WHERE l.ordered>l.received""",
 'Q3':"""SELECT w.uid,SUM(l.ordered),SUM(l.received) FROM nodes p
 JOIN edges e ON e.a=p.uid AND e.relation='ORDERED_FROM' JOIN nodes s ON s.uid=e.b
 JOIN edges lp ON lp.b=p.uid AND lp.relation='PART_OF_PO' JOIN nodes l ON l.uid=lp.a
 JOIN identities ia ON ia.uid=s.uid JOIN identities iw ON iw.canonical=ia.canonical
 JOIN nodes w ON w.uid=iw.uid AND w.kind='Supplier' AND w.silo='B'
 WHERE s.split='test' AND w.split='test' GROUP BY w.uid""",
 'Q4':"""SELECT DISTINCT i.uid FROM edges e JOIN nodes i ON i.uid=e.b
 WHERE e.relation='MOVES_STOCK_ITEM' AND i.split='test'""",
 'Q5':"""SELECT DISTINCT a.uid,t.uid FROM nodes t JOIN edges e ON e.a=t.uid AND e.relation='RELATES_TO_SUPPLIER'
 JOIN nodes b ON b.uid=e.b JOIN identities ib ON ib.uid=b.uid JOIN identities ia ON ia.canonical=ib.canonical
 JOIN nodes a ON a.uid=ia.uid AND a.kind='Supplier' AND a.silo='A'
 WHERE b.split='test' AND a.split='test'""",
 'Q6':"""SELECT DISTINCT a.uid,b.uid FROM nodes a JOIN identities ia ON ia.uid=a.uid
 JOIN identities ib ON ib.canonical=ia.canonical JOIN nodes b ON b.uid=ib.uid
 WHERE a.kind='Supplier' AND b.kind='Supplier' AND a.silo='A' AND b.silo='B'
 AND a.split='test' AND b.split='test'""",
}

def sql_database(graph,mapping):
    sql=sqlite3.connect(':memory:')
    sql.executescript('CREATE TABLE nodes(uid TEXT PRIMARY KEY,kind TEXT,silo TEXT,split TEXT,ordered INTEGER,received INTEGER); CREATE TABLE edges(a TEXT,b TEXT,relation TEXT); CREATE TABLE identities(uid TEXT PRIMARY KEY,canonical TEXT);')
    sql.executemany('INSERT INTO nodes VALUES(?,?,?,?,?,?)',[(u,r['entity_type'],r['silo'],r.get('split',''),r.get('ordered',0),r.get('received',0)) for u,r in graph.nodes(data=True)])
    sql.executemany('INSERT INTO edges VALUES(?,?,?)',[(a,b,k) for a,b,k in graph.edges(keys=True)])
    sql.executemany('INSERT INTO identities VALUES(?,?)',mapping.items())
    sql.executescript('CREATE INDEX edge_out ON edges(a,relation); CREATE INDEX edge_in ON edges(b,relation); CREATE INDEX canonical_idx ON identities(canonical); CREATE INDEX node_kind ON nodes(kind,silo,split);')
    return sql

def sql_answers(sql):
    return {q:set(sql.execute(statement).fetchall()) for q,statement in SQL.items()}

def graph_answers(graph,mapping):
    outgoing,incoming = defaultdict(lambda:defaultdict(set)),defaultdict(lambda:defaultdict(set))
    for a,b,k in graph.edges(keys=True):
        outgoing[a][k].add(b);incoming[b][k].add(a)
    suppliers_a,suppliers_b,items_b = [],[],[]
    for u,r in graph.nodes(data=True):
        if r.get('split') != 'test':continue
        if r['entity_type']=='Supplier':
            (suppliers_a if r['silo']=='A' else suppliers_b).append(u)
        elif r['entity_type']=='StockItem' and r['silo']=='B':items_b.append(u)
    item_index,supplier_index = defaultdict(set),defaultdict(set)
    for u in items_b:item_index[mapping[u]].add(u)
    for u in suppliers_b:supplier_index[mapping[u]].add(u)
    answers={q:set() for q in SQL}
    sums=defaultdict(lambda:[0,0])
    for supplier in suppliers_a:
        lines=[]
        for po in incoming[supplier]['ORDERED_FROM']:
            lines.extend(incoming[po]['PART_OF_PO'])
        for line in lines:
            for item in outgoing[line]['FOR_STOCK_ITEM']:
                for warehouse in item_index[mapping[item]]:answers['Q1'].add((supplier,warehouse))
        for warehouse_supplier in supplier_index[mapping[supplier]]:
            for line in lines:
                r=graph.nodes[line]
                sums[warehouse_supplier][0]+=r['ordered'];sums[warehouse_supplier][1]+=r['received']
            for tx in incoming[warehouse_supplier]['RELATES_TO_SUPPLIER']:answers['Q5'].add((supplier,tx))
            answers['Q6'].add((supplier,warehouse_supplier))
    for supplier,(ordered,received) in sums.items():answers['Q3'].add((supplier,ordered,received))
    for u,r in graph.nodes(data=True):
        if r['entity_type']=='PurchaseOrderLine' and r['ordered']>r['received']:
            if outgoing[u]['FOR_STOCK_ITEM']:answers['Q2'].add((u,))
    for item in items_b:
        if incoming[item]['MOVES_STOCK_ITEM']:answers['Q4'].add((item,))
    return answers
