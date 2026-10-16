from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import time
from .io import ROOT, digest, read_json, write_json, write_rows
from .schema import TABLES, RELATIONS, reference_edges

def extract():
    spec = spec_from_file_location("restore_wwi", ROOT / "scripts/restore_wwi.py")
    sql = module_from_spec(spec)
    spec.loader.exec_module(sql)
    inventory = sql.query("""USE KGErpWWI;
        SELECT SCHEMA_NAME(t.schema_id)+'.'+t.name AS table_name, c.name AS column_name,
        TYPE_NAME(c.user_type_id) AS data_type, c.is_nullable
        FROM sys.tables t JOIN sys.columns c ON t.object_id=c.object_id
        WHERE SCHEMA_NAME(t.schema_id) IN ('Purchasing','Warehouse')
        ORDER BY table_name,c.column_id FOR JSON PATH""", True)
    foreign_keys = sql.query("""USE KGErpWWI;
        SELECT SCHEMA_NAME(a.schema_id)+'.'+a.name AS source_table, ca.name AS source_column,
        SCHEMA_NAME(b.schema_id)+'.'+b.name AS target_table, cb.name AS target_column
        FROM sys.foreign_key_columns f JOIN sys.tables a ON f.parent_object_id=a.object_id
        JOIN sys.columns ca ON ca.object_id=a.object_id AND ca.column_id=f.parent_column_id
        JOIN sys.tables b ON f.referenced_object_id=b.object_id
        JOIN sys.columns cb ON cb.object_id=b.object_id AND cb.column_id=f.referenced_column_id
        FOR JSON PATH""", True)
    actual = {(x['source_table'], x['source_column'], x['target_table']) for x in foreign_keys}
    for source, column, target, _ in RELATIONS:
        if (source, column, target) not in actual:
            raise ValueError(f"Unverified relationship: {source}.{column} -> {target}")
    write_json(ROOT / "data/manifests/schema_inventory.json", inventory)
    write_json(ROOT / "data/manifests/foreign_keys.json", foreign_keys)
    reference_counts = []
    for source, column, target, relation in RELATIONS:
        target_pk = TABLES[target][1]
        count = int(sql.query(f"USE KGErpWWI; SELECT COUNT(*) FROM {source} s JOIN {target} t ON s.[{column}]=t.[{target_pk}]").strip())
        reference_counts.append({'source_table':source,'source_column':column,'target_table':target,
                                 'relation':relation,'valid_source_join_rows':count})
    write_json(ROOT / "data/manifests/sql_reference_counts.json", reference_counts)
    manifest = {}
    for table in TABLES:
        path = ROOT / "data/staged" / (table + ".json")
        started = time.perf_counter()
        if not path.exists():
            columns = [x for x in inventory if x['table_name'] == table]
            projection = ','.join(f"[{x['column_name']}].ToString() AS [{x['column_name']}]" if x['data_type'] in {'geography', 'geometry'} else f"[{x['column_name']}]" for x in columns)
            rows = sql.query(f"USE KGErpWWI; SELECT {projection} FROM {table} ORDER BY {TABLES[table][1]} FOR JSON PATH, INCLUDE_NULL_VALUES", True)
            write_json(path, rows)
        else:
            rows = read_json(path)
        count = int(sql.query(f"USE KGErpWWI; SELECT COUNT(*) FROM {table}").strip())
        if len(rows) != count:
            raise ValueError(f"Source/staged count mismatch in {table}")
        manifest[table] = {"rows": count, "sha256": digest(path), "source_count_verified": True,
                           "extract_seconds": time.perf_counter() - started}
        print(f"Staged {table}: {count:,} rows", flush=True)
    write_json(ROOT / "data/manifests/wwi_snapshot.json", manifest)
    return profile()

def load_tables():
    return {t: read_json(ROOT / "data/staged" / (t + ".json")) for t in TABLES}

def profile():
    tables = load_tables()
    expected, invalid = reference_edges(tables)
    if invalid:
        raise ValueError(f"{len(invalid)} unexpected in-scope orphan foreign keys")
    rows = []
    for table, records in tables.items():
        pk = TABLES[table][1]
        n = len(records)
        if len({r[pk] for r in records}) != n:
            raise ValueError(f"Duplicate primary key in {table}")
        rows.append({"table": table, "rows": n, "unique_pk": n,
                     "outgoing_valid_edges": sum(a.startswith(f"wwi:{table}:") for a, _, _ in expected)})
    write_rows(ROOT / "data/manifests/profile.csv", rows)
    missing = [{"table": t, "column": c, "nulls": sum(r.get(c) is None for r in records), "rows": len(records)}
               for t, records in tables.items() if records for c in records[0]]
    write_rows(ROOT / "data/manifests/null_profile.csv", missing)
    return rows
