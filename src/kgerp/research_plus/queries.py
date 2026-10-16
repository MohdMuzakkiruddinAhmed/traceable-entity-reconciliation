"""Prepared query view: build adjacency once, then measure actual query work."""
from collections import defaultdict

class PreparedGraphQueries:
    def __init__(self, graph, mapping):
        self.nodes = {u: dict(r) for u, r in graph.nodes(data=True)}
        self.identities = dict(mapping)
        self.outgoing = defaultdict(lambda: defaultdict(set))
        self.incoming = defaultdict(lambda: defaultdict(set))
        for a, b, relation in graph.edges(keys=True):
            self.outgoing[a][relation].add(b)
            self.incoming[b][relation].add(a)
        self.suppliers_a = []
        self.items_b = []
        self.item_index = defaultdict(set)
        self.supplier_index = defaultdict(set)
        for u, r in self.nodes.items():
            if r.get('split') != 'test':
                continue
            if r['entity_type'] == 'Supplier' and r['silo'] == 'A':
                self.suppliers_a.append(u)
            elif r['entity_type'] == 'Supplier' and r['silo'] == 'B':
                self.supplier_index[self.identities[u]].add(u)
            elif r['entity_type'] == 'StockItem' and r['silo'] == 'B':
                self.items_b.append(u)
                self.item_index[self.identities[u]].add(u)
        self.lines_by_supplier = {}
        for supplier in self.suppliers_a:
            self.lines_by_supplier[supplier] = tuple(line for po in self.incoming[supplier]['ORDERED_FROM'] for line in self.incoming[po]['PART_OF_PO'])
        self.unreceived = tuple(u for u, r in self.nodes.items() if r['entity_type'] == 'PurchaseOrderLine' and
                                r['ordered'] > r['received'] and self.outgoing[u]['FOR_STOCK_ITEM'])

    def query(self, name):
        answer = set()
        if name == 'Q1':
            for supplier, lines in self.lines_by_supplier.items():
                for line in lines:
                    for item in self.outgoing[line]['FOR_STOCK_ITEM']:
                        answer.update((supplier, w) for w in self.item_index[self.identities[item]])
        elif name == 'Q2':
            answer = {(u,) for u in self.unreceived}
        elif name == 'Q3':
            sums = defaultdict(lambda: [0, 0])
            for supplier, lines in self.lines_by_supplier.items():
                for target in self.supplier_index[self.identities[supplier]]:
                    for line in lines:
                        sums[target][0] += self.nodes[line]['ordered']
                        sums[target][1] += self.nodes[line]['received']
            answer = {(u, values[0], values[1]) for u, values in sums.items()}
        elif name == 'Q4':
            answer = {(u,) for u in self.items_b if self.incoming[u]['MOVES_STOCK_ITEM']}
        elif name == 'Q5':
            for supplier in self.suppliers_a:
                for target in self.supplier_index[self.identities[supplier]]:
                    answer.update((supplier, tx) for tx in self.incoming[target]['RELATES_TO_SUPPLIER'])
        elif name == 'Q6':
            for supplier in self.suppliers_a:
                answer.update((supplier, target) for target in self.supplier_index[self.identities[supplier]])
        else:
            raise ValueError('Unknown query: ' + name)
        return answer

    def execute(self):
        return {f'Q{i}': self.query(f'Q{i}') for i in range(1, 7)}
