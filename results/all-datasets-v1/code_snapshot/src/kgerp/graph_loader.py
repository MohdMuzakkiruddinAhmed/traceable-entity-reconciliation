import pickle
import networkx as nx
from .io import ROOT, digest
from .schema import TABLES, reference_edges, uid

def build_graph(tables, graph=None, batch="wwi-standard-v1"):
    graph = nx.MultiDiGraph() if graph is None else graph
    for table, rows in tables.items():
        kind, pk = TABLES[table]
        for row in rows:
            source_uid = uid(table, row[pk])
            graph.add_node(source_uid, entity_type=kind, source_uid=source_uid,
                           source_table=table, source_record_id=str(row[pk]),
                           source_batch=batch, mapper_version="1", attributes=row)
    edges, invalid = reference_edges(tables)
    if invalid:
        raise ValueError(f"Cannot build graph with {len(invalid)} orphan foreign keys")
    for left, relation, right in sorted(edges):
        graph.add_edge(left, right, key=relation, relation=relation, mapper_version="1", source_batch=batch)
    return graph

def save_graph(graph):
    path = ROOT / "data/generated/source_graph.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return digest(path)
