"""Verified in-scope source mapping. No transaction-to-PO invention."""
TABLES = {
    "Purchasing.Suppliers": ("Supplier", "SupplierID"),
    "Purchasing.PurchaseOrders": ("PurchaseOrder", "PurchaseOrderID"),
    "Purchasing.PurchaseOrderLines": ("PurchaseOrderLine", "PurchaseOrderLineID"),
    "Warehouse.StockItems": ("StockItem", "StockItemID"),
    "Warehouse.StockItemTransactions": ("StockMovement", "StockItemTransactionID"),
    "Purchasing.SupplierTransactions": ("SupplierTransaction", "SupplierTransactionID"),
}
RELATIONS = [
    ("Purchasing.PurchaseOrders", "SupplierID", "Purchasing.Suppliers", "ORDERED_FROM"),
    ("Purchasing.PurchaseOrderLines", "PurchaseOrderID", "Purchasing.PurchaseOrders", "PART_OF_PO"),
    ("Purchasing.PurchaseOrderLines", "StockItemID", "Warehouse.StockItems", "FOR_STOCK_ITEM"),
    ("Warehouse.StockItems", "SupplierID", "Purchasing.Suppliers", "FROM_SUPPLIER"),
    ("Warehouse.StockItemTransactions", "StockItemID", "Warehouse.StockItems", "MOVES_STOCK_ITEM"),
    ("Warehouse.StockItemTransactions", "PurchaseOrderID", "Purchasing.PurchaseOrders", "FOR_PO"),
    ("Warehouse.StockItemTransactions", "SupplierID", "Purchasing.Suppliers", "RELATES_TO_SUPPLIER"),
    ("Purchasing.SupplierTransactions", "SupplierID", "Purchasing.Suppliers", "RELATES_TO_SUPPLIER"),
]

def uid(table, row_id):
    return f"wwi:{table}:{row_id}"

def reference_edges(tables):
    keys = {t: {r[pk] for r in rows} for t, rows in tables.items() for _, pk in [TABLES[t]]}
    edges, invalid = set(), []
    for source, fk, target, relation in RELATIONS:
        for row in tables[source]:
            value = row.get(fk)
            if value is None:
                continue
            if value not in keys[target]:
                invalid.append({"table": source, "row_id": row[TABLES[source][1]], "column": fk, "value": value})
            else:
                edges.add((uid(source, row[TABLES[source][1]]), relation, uid(target, value)))
    return edges, invalid
