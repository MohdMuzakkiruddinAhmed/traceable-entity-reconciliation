"""Preserve the six WWI tasks and add planned category/lineage queries."""
from collections import defaultdict
from ..io import write_json,write_rows
from ..metrics import set_metrics
from ..oracle import oracle_mapping
from ..queries import sql_database,sql_answers
from ..research_plus.queries import PreparedGraphQueries

EXTRA={
 'Q7':'''SELECT w.uid,p.category,SUM(l.ordered),SUM(l.received) FROM nodes po
 JOIN edges es ON es.a=po.uid AND es.relation='ORDERED_FROM' JOIN nodes s ON s.uid=es.b
 JOIN edges ep ON ep.b=po.uid AND ep.relation='PART_OF_PO' JOIN nodes l ON l.uid=ep.a
 JOIN identities a ON a.uid=s.uid JOIN identities b ON b.canonical=a.canonical
 JOIN nodes w ON w.uid=b.uid AND w.kind='Supplier' AND w.silo='B' JOIN props p ON p.uid=w.uid
 WHERE s.split='test' AND w.split='test' GROUP BY w.uid,p.category''',
 'Q8':'''SELECT a.uid,t.uid,p.source_table,p.source_batch,p.mapper_version FROM nodes t
 JOIN edges e ON e.a=t.uid AND e.relation='RELATES_TO_SUPPLIER' JOIN nodes b ON b.uid=e.b
 JOIN identities ib ON ib.uid=b.uid JOIN identities ia ON ia.canonical=ib.canonical
 JOIN nodes a ON a.uid=ia.uid AND a.kind='Supplier' AND a.silo='A' JOIN props p ON p.uid=t.uid
 WHERE b.split='test' AND a.split='test' '''
}


def sql_extra(graph,mapping):
    sql=sql_database(graph,mapping)
    sql.execute('CREATE TABLE props(uid TEXT PRIMARY KEY,category TEXT,source_table TEXT,source_batch TEXT,mapper_version TEXT)')
    sql.executemany('INSERT INTO props VALUES (?,?,?,?,?)',[(u,a.get('category',''),a.get('source_table',''),a.get('source_batch',''),a.get('mapper_version','')) for u,a in graph.nodes(data=True)])
    result=sql_answers(sql);result.update({q:set(sql.execute(s).fetchall()) for q,s in EXTRA.items()});sql.close();return result


def graph_extra(graph,mapping):
    result=PreparedGraphQueries(graph,mapping).execute()
    result['Q7']={(u,graph.nodes[u].get('category',''),ordered,received) for u,ordered,received in result['Q3']}
    result['Q8']={(a,t,graph.nodes[t]['source_table'],graph.nodes[t]['source_batch'],graph.nodes[t]['mapper_version']) for a,t in result['Q5']}
    return result


def evaluate(bundle,out,seed):
    graph=bundle['graph'];records=bundle['records'];gold=bundle['gold'];mapping=bundle['mapping']
    oracle=oracle_mapping(records,gold);expected=sql_extra(graph,oracle)
    if any(not a for a in expected.values()):raise ValueError('Empty native WWI eight-query oracle')
    rows=[]
    for name,identity in [('Unreconciled',{r['source_uid']:r['source_uid'] for r in records}),('Predicted',mapping),('Oracle',oracle)]:
        actual=graph_extra(graph,identity)
        if actual!=sql_extra(graph,identity):raise ValueError('Native WWI Q1-Q8 parity')
        rows.extend({'seed':seed,'identity':name,'query':q,'SQL_KG_exact':True,**set_metrics(expected[q],actual[q])} for q in actual)
    write_rows(out/f'seed_{seed}/native_query_accuracy.csv',rows)
    write_json(out/f'seed_{seed}/native_query_catalog.json',{'Q1_Q6':'preserved original tasks',
        'Q7':'ordered/received purchase outers by reconciled source-B supplier and visible supplier category',
        'Q8':'source table/batch/mapping version for reconciled supplier financial-transaction paths',
        'no_changes_to_original_query_results_or_schema':True})
