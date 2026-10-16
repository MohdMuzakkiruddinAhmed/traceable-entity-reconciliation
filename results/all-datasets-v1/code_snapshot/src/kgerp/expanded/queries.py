"""Eight source-UID answer tasks with identical graph/SQL identity maps."""
from collections import defaultdict
import sqlite3

SQL={
 'Q1':'''WITH categories AS (SELECT DISTINCT o.master,t.label FROM own o JOIN tags t ON t.tx=o.tx)
         SELECT a.uid,b.uid,x.label FROM masters a JOIN masters b ON a.canonical=b.canonical AND a.silo='A' AND b.silo='B'
         JOIN categories x ON x.master=a.uid JOIN categories y ON y.master=b.uid AND y.label=x.label''',
 'Q2':'''SELECT uid FROM transactions WHERE amount_cents>0''',
 'Q3':'''SELECT a.uid,SUM(t.amount_cents),COUNT(t.uid) FROM masters a JOIN masters b ON a.canonical=b.canonical AND a.silo='A' AND b.silo='B'
         JOIN own o ON o.master=b.uid JOIN transactions t ON t.uid=o.tx GROUP BY a.uid''',
 'Q4':'''SELECT o.master,o.tx,c.label FROM own o JOIN tags c ON c.tx=o.tx''',
 'Q5':'''SELECT a.uid,t.uid FROM masters a JOIN masters b ON a.canonical=b.canonical AND a.silo='A' AND b.silo='B'
         JOIN own o ON o.master=b.uid JOIN transactions t ON t.uid=o.tx WHERE t.amount_cents>0''',
 'Q6':'''SELECT a.uid,b.uid FROM masters a JOIN masters b ON a.canonical=b.canonical AND a.silo='A' AND b.silo='B' ''',
 'Q7':'''SELECT a.uid,c.label,SUM(t.amount_cents),COUNT(t.uid) FROM masters a JOIN masters b ON a.canonical=b.canonical AND a.silo='A' AND b.silo='B'
         JOIN own o ON o.master=b.uid JOIN transactions t ON t.uid=o.tx JOIN tags c ON c.tx=t.uid GROUP BY a.uid,c.label''',
 'Q8':'''SELECT a.uid,t.uid,t.source_table FROM masters a JOIN masters b ON a.canonical=b.canonical AND a.silo='A' AND b.silo='B'
         JOIN own o ON o.master=b.uid JOIN transactions t ON t.uid=o.tx'''
}


def catalog(dataset):
    return {'dataset':dataset,'answers':'tuples of independently remapped source UIDs; monetary sums in integer cent-rounded source units',
            'source_local_controls':['Q2','Q4'], 'identity_sensitive':['Q1','Q3','Q5','Q6','Q7','Q8'],
            'amount_semantics':'initial anonymized purchase-item value, not paid invoices' if dataset=='BPI2019' else 'source order-line amount',
            'Q1':'matched source entities sharing an observed transaction category', 'Q2':'local positive-value transactions',
            'Q3':'source-B amount/count attributed to a reconciled source-A entity','Q4':'local typed entity/transaction/category paths',
            'Q5':'positive-value source-B transactions linked to source-A entities','Q6':'cross-source identity pairs',
            'Q7':'source-B amounts/counts by visible category attributed to source A','Q8':'source-table lineage for reconciled transaction paths',
            'scope':'same sampled independent transaction histories as matching; full raw source fidelity is separate E1'}


def scoped_graph(graph,records):
    nodes={r['source_uid'] for r in records}
    for uid in list(nodes):
        for _,tx,k in graph.out_edges(uid,keys=True):
            if k=='HAS_TRANSACTION':
                nodes.add(tx)
                nodes.update(t for _,t,key in graph.out_edges(tx,keys=True) if key=='HAS_CATEGORY')
    return graph.subgraph(nodes).copy()


def cents(attrs):
    value=attrs.get('amount')
    return int(round(float(value)*100)) if value is not None else 0


class PreparedQueries:
    def __init__(self,graph,mapping):
        self.masters={u:a for u,a in graph.nodes(data=True) if a['entity_type'] not in {'Transaction','ContextCategory'}}
        self.transactions={u:a for u,a in graph.nodes(data=True) if a['entity_type']=='Transaction'}
        self.owned=defaultdict(list);self.tags=defaultdict(list);identities=defaultdict(dict)
        for a,b,k in graph.edges(keys=True):
            if k=='HAS_TRANSACTION':self.owned[a].append(b)
            elif k=='HAS_CATEGORY':self.tags[a].append(graph.nodes[b]['label'])
        for u,a in self.masters.items():identities[mapping.get(u,u)][a['silo']]=u
        self.pairs=[(s['A'],s['B']) for s in identities.values() if 'A' in s and 'B' in s]

    def query(self,q):
        if q=='Q1':
            output=set()
            for a,b in self.pairs:
                ca={c for tx in self.owned[a] for c in self.tags[tx]};cb={c for tx in self.owned[b] for c in self.tags[tx]}
                output.update((a,b,c) for c in ca&cb)
            return output
        if q=='Q2':return {(u,) for u,a in self.transactions.items() if cents(a)>0}
        if q=='Q3':return {(a,sum(cents(self.transactions[t]) for t in self.owned[b]),len(self.owned[b])) for a,b in self.pairs if self.owned[b]}
        if q=='Q4':return {(a,t,c) for a,ts in self.owned.items() for t in ts for c in self.tags[t]}
        if q=='Q5':return {(a,t) for a,b in self.pairs for t in self.owned[b] if cents(self.transactions[t])>0}
        if q=='Q6':return set(self.pairs)
        if q=='Q7':
            output=set()
            for a,b in self.pairs:
                totals=defaultdict(lambda:[0,0])
                for tx in self.owned[b]:
                    for c in self.tags[tx]:totals[c][0]+=cents(self.transactions[tx]);totals[c][1]+=1
                output.update((a,c,v[0],v[1]) for c,v in totals.items())
            return output
        if q=='Q8':return {(a,t,self.transactions[t]['source_table']) for a,b in self.pairs for t in self.owned[b]}
        raise ValueError(q)

    def execute(self):return {q:self.query(q) for q in SQL}


class SQLQueries:
    def __init__(self,graph,mapping):
        self.db=sqlite3.connect(':memory:')
        self.db.executescript('CREATE TABLE masters(uid TEXT PRIMARY KEY,silo TEXT,kind TEXT,canonical TEXT); CREATE TABLE transactions(uid TEXT PRIMARY KEY,amount_cents INTEGER,source_table TEXT); CREATE TABLE own(master TEXT,tx TEXT); CREATE TABLE tags(tx TEXT,label TEXT);')
        self.db.executemany('INSERT INTO masters VALUES (?,?,?,?)',[(u,a['silo'],a['entity_type'],mapping.get(u,u)) for u,a in graph.nodes(data=True) if a['entity_type'] not in {'Transaction','ContextCategory'}])
        self.db.executemany('INSERT INTO transactions VALUES (?,?,?)',[(u,cents(a),a['source_table']) for u,a in graph.nodes(data=True) if a['entity_type']=='Transaction'])
        self.db.executemany('INSERT INTO own VALUES (?,?)',[(a,b) for a,b,k in graph.edges(keys=True) if k=='HAS_TRANSACTION'])
        self.db.executemany('INSERT INTO tags VALUES (?,?)',[(a,graph.nodes[b]['label']) for a,b,k in graph.edges(keys=True) if k=='HAS_CATEGORY'])
        self.db.executescript('CREATE INDEX canonical_idx ON masters(canonical,silo); CREATE INDEX own_master ON own(master); CREATE INDEX own_tx ON own(tx); CREATE INDEX tags_tx ON tags(tx);')

    def query(self,q):return set(self.db.execute(SQL[q]).fetchall())
    def execute(self):return {q:self.query(q) for q in SQL}
    def close(self):self.db.close()
