"""Reexecute query controls from frozen predictions and independently audit WDC.

Run against the complete KG-ERP artifact. No models, thresholds, or source inputs
are changed. Outputs are written to a new directory supplied with --output.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
import zipfile


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new empty output directory to preserve previous measurements")
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(project / "src"))
    import numpy as np
    import pandas as pd
    import joblib
    import yaml
    from kgerp.expanded import protocol
    from kgerp.expanded.queries import scoped_graph, PreparedQueries, SQLQueries
    from kgerp.expanded.features import graph_profiles, context
    from kgerp.metrics import set_metrics, binary
    from kgerp.oracle import true_pair_count

    run = project / "results/all-datasets-v1"
    config = yaml.safe_load((run / "config_snapshot.yaml").read_text())
    checks = []
    inputs = {}

    def check(name, condition, detail=None):
        if not condition:
            raise AssertionError(name + ": " + str(detail))
        checks.append({"check": name, "passed": True, "detail": detail})

    def input_file(path):
        inputs[str(path.relative_to(project)).replace("\\", "/")] = sha(path)

    def canonicalize(records, predictions, method, threshold):
        kinds = {r["source_uid"]: r["entity_type"] for r in records}
        mapping = {uid: uid for uid in kinds}
        used = set()
        accepted = []
        ordered = sorted(predictions, key=lambda p: (
            -float(p[method]), (p["left_source_uid"], p["right_source_uid"])))
        for pred in ordered:
            a, b = pred["left_source_uid"], pred["right_source_uid"]
            if float(pred[method]) >= threshold and kinds[a] == kinds[b] and a not in used and b not in used:
                used.update((a, b))
                cid = "canonical:" + hashlib.sha256((a + "|" + b).encode()).hexdigest()[:24]
                mapping[a] = mapping[b] = cid
                accepted.append((a, b))
        return mapping, set(accepted)

    input_file(run / "config_snapshot.yaml")
    # Generic preparation normally persists truth. This analysis needs only the
    # deterministic in-memory return value, so suppress that write explicitly.
    previous_writer = protocol.write_json
    protocol.write_json = lambda *unused_args, **unused_kwargs: None
    query_rows, identity_rows, gate_rows = [], [], []
    try:
        for dataset in ("WWI", "Olist", "DataCo", "BPI2019"):
            if dataset != "WWI":
                input_file(project / f"data/expanded/{dataset}/base_views.json")
            else:
                input_file(project / "data/staged/Warehouse.StockItems.json")
            for seed in config["seeds"]:
                folder = run / dataset / f"seed_{seed}"
                for name in ("predictions.json", "freeze.json", "decisions.json", "query_accuracy.csv", "robustness.csv"):
                    input_file(folder / name)
                if dataset == "WWI":
                    for path in (project / f"data/generated/seed_{seed}/records.json",
                                 project / f"data/generated/seed_{seed}/graph.pkl",
                                 project / f"data/gold/seed_{seed}/truth.json"):
                        input_file(path)
                predictions, freeze = read(folder / "predictions.json"), read(folder / "freeze.json")
                records, graph, gold, _ = protocol.build_views(dataset, seed, config)
                test = [r for r in records if r["split"] == "test"]
                check(f"{dataset}_{seed}_labels", all(
                    int(gold[p["left_source_uid"]]["canonical"] == gold[p["right_source_uid"]]["canonical"]) == p["label"]
                    for p in predictions))
                scope = scoped_graph(graph, test)
                oracle = {r["source_uid"]: "oracle:" + gold[r["source_uid"]]["canonical"] for r in test}
                oracle_sql = SQLQueries(scope, oracle)
                expected = oracle_sql.execute()
                oracle_sql.close()
                check(f"{dataset}_{seed}_oracle_nonempty", all(expected.values()))
                check(f"{dataset}_{seed}_oracle_parity", expected == PreparedQueries(scope, oracle).execute())
                total = true_pair_count(test, gold)
                original = pd.read_csv(folder / "query_accuracy.csv")
                for method in ("B2Attributes", "B5Graph"):
                    mapping, accepted = canonicalize(test, predictions, method, freeze["accepts"][method])
                    check(f"{dataset}_{seed}_{method}_one_to_one", len({a for a, b in accepted}) == len(accepted) == len({b for a, b in accepted}))
                    if method == "B5Graph":
                        prior = {(d["left_source_uid"], d["right_source_uid"]) for d in read(folder / "decisions.json") if d["status"] == "accepted"}
                        check(f"{dataset}_{seed}_original_context_links_reproduced", accepted == prior)
                    truth = {(r["source_uid"], s["source_uid"]) for r in test for s in test
                             if r["silo"] == "A" and s["silo"] == "B" and gold[r["source_uid"]]["canonical"] == gold[s["source_uid"]]["canonical"]}
                    link_metrics = set_metrics(truth, accepted)
                    tp = link_metrics["tp"]
                    identity_rows.append({"dataset": dataset, "seed": seed, "method": method,
                                          "threshold": freeze["accepts"][method], "test_groups": total,
                                          "accepted_links": len(accepted), "false_merges": len(accepted) - tp,
                                          "true_accepted": tp, "precision": link_metrics["precision"],
                                          "recall": link_metrics["recall"], "f1": link_metrics["f1"]})
                    actual = PreparedQueries(scope, mapping).execute()
                    sql = SQLQueries(scope, mapping)
                    check(f"{dataset}_{seed}_{method}_query_parity", actual == sql.execute())
                    sql.close()
                    for q, answer in actual.items():
                        metrics = set_metrics(expected[q], answer)
                        row = {"dataset": dataset, "seed": seed, "method": method, "query": q,
                               "SQL_KG_exact": True, **metrics}
                        query_rows.append(row)
                        check(f"{dataset}_{seed}_{method}_{q}_counts", metrics["tp"] <= min(len(expected[q]), len(answer)))
                        if q in ("Q2", "Q4"):
                            check(f"{dataset}_{seed}_{method}_{q}_local_control", metrics["f1"] == 1.)
                        if method == "B5Graph":
                            old = original[(original.identity == "Predicted") & (original["query"] == q)].iloc[0]
                            check(f"{dataset}_{seed}_{q}_original_metric_reproduced", all(abs(float(old[k]) - float(v)) < 1e-12 for k, v in metrics.items()))
                if dataset == "BPI2019":
                    model_path = folder / "frozen_models.joblib"
                    input_file(model_path)
                    check(f"BPI_{seed}_model_hash", sha(model_path) == freeze["model_sha256"])
                    fitted = joblib.load(model_path)
                    pairs = [(p["left_source_uid"], p["right_source_uid"]) for p in predictions]
                    by_uid = {r["source_uid"]: r for r in test}
                    ax = fitted["attributes"].transform(pairs, by_uid)
                    p_attr = fitted["models"]["calibration"]["attribute"].transform(
                        fitted["models"]["attribute"].predict_proba(ax)[:, 1])
                    check(f"BPI_{seed}_frozen_attribute_replay", np.max(np.abs(p_attr - [p["B2Attributes"] for p in predictions])) <= 1e-12)
                    changed, view, _ = protocol.perturb(test, graph, seed + 701, no_anchors=True)
                    cx, _ = context(pairs, graph_profiles(view, changed), config["models"]["min_anchors"])
                    check(f"BPI_{seed}_all_absent_support_gates_zero", not (cx[:, -1] > 0).any())
                    # Equation (3) now returns the same attribute probabilities;
                    # evaluate them using the two already frozen policies.
                    saved_stress = pd.read_csv(folder / "robustness.csv")
                    for method in ("B2Attributes", "B5Graph"):
                        metrics = binary([p["label"] for p in predictions], p_attr >= freeze["accepts"][method], p_attr, total)
                        old = saved_stress[(saved_stress.scenario == "no_anchors") & (saved_stress.model == method)].iloc[0]
                        check(f"BPI_{seed}_{method}_absence_stress_reproduced", all(abs(float(old[k]) - float(metrics[k])) < 1e-12 for k in ("tp", "fp", "fn", "tn", "end_to_end_f1")))
                        gate_rows.append({"dataset": dataset, "seed": seed, "method": method,
                                          "gate_zero_pairs": len(pairs), "score_max_difference_from_attributes": 0.,
                                          "accept_threshold": freeze["accepts"][method], **metrics})
                print(f"{dataset} seed {seed}: paired query execution and canonical-link controls passed", flush=True)
    finally:
        protocol.write_json = previous_writer

    qframe = pd.DataFrame(query_rows)
    qmeans = qframe.groupby(["dataset", "method", "query"], sort=False)[["precision", "recall", "f1"]].mean().reset_index()
    paired = qframe.pivot(index=["dataset", "seed", "query"], columns="method", values="f1").reset_index()
    paired["delta_f1"] = paired.B5Graph - paired.B2Attributes
    paired_means = paired.groupby(["dataset", "query"], sort=False)[["B2Attributes", "B5Graph", "delta_f1"]].mean().reset_index()
    idframe = pd.DataFrame(identity_rows)
    totals = idframe.groupby(["dataset", "method"], sort=False)[["test_groups", "accepted_links", "false_merges", "true_accepted"]].sum().reset_index()
    totals["precision"] = totals.true_accepted / totals.accepted_links
    totals["recall"] = totals.true_accepted / totals.test_groups
    totals["f1"] = 2 * totals.true_accepted / (totals.accepted_links + totals.test_groups)
    for name, rows in (("query_controls.csv", query_rows), ("identity_controls.csv", identity_rows), ("gate_policy_diagnostic.csv", gate_rows)):
        write_csv(output / name, rows)
    for name, frame in (("query_means.csv", qmeans), ("query_paired_differences.csv", paired),
                        ("query_paired_means.csv", paired_means), ("identity_totals.csv", totals)):
        frame.to_csv(output / name, index=False)

    # Independent parser: compare publisher offer IDs on either pair endpoint,
    # then confirm that intersections contain identical fields and cluster IDs.
    field_names = ("cluster_id", "title", "description", "brand", "price", "priceCurrency")
    audit, examples, archive_info = [], [], []
    prior_audit = pd.read_csv(run / "WDC/split_integrity.csv")
    input_file(run / "WDC/split_integrity.csv")
    for hardness in (20, 50, 80):
        archive_path = project / f"data/raw/wdc-products-{hardness}pair.zip"
        input_file(archive_path)
        archive_info.append({"corner_cases": hardness, "archive": archive_path.name,
                             "sha256": sha(archive_path), "bytes": archive_path.stat().st_size,
                             "url": f"https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/wdc-products/{hardness}pair.zip"})
        with zipfile.ZipFile(archive_path) as archive:
            prefix = f"wdcproducts{hardness}cc{100-hardness}rnd"
            cache = {}

            def parse(member):
                if member not in cache:
                    payload = gzip.decompress(archive.read(member))
                    pairs = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
                    records = {}
                    for pair in pairs:
                        for side in ("left", "right"):
                            uid = str(pair["id_" + side])
                            attrs = {f: pair.get(f + "_" + side) for f in field_names}
                            if uid in records and records[uid] != attrs:
                                raise ValueError("Inconsistent attributes for publisher offer ID " + uid)
                            records[uid] = attrs
                    cache[member] = (records, pairs, hashlib.sha256(payload).hexdigest())
                return cache[member]

            for size in ("small", "medium", "large"):
                tm, vm = prefix + "000un_train_" + size + ".json.gz", prefix + "000un_valid_" + size + ".json.gz"
                train, train_pairs, th = parse(tm)
                valid, valid_pairs, vh = parse(vm)
                for unseen in (0, 50, 100):
                    em = prefix + f"{unseen:03d}un_gs.json.gz"
                    test, test_pairs, eh = parse(em)
                    row = {"corner_cases": hardness, "development_size": size, "unseen_percent": unseen,
                           "train_member": tm, "validation_member": vm, "test_member": em,
                           "train_pairs": len(train_pairs), "validation_pairs": len(valid_pairs), "test_pairs": len(test_pairs),
                           "train_decompressed_sha256": th, "validation_decompressed_sha256": vh, "test_decompressed_sha256": eh}
                    for split, records in (("train", train), ("validation", valid)):
                        intersection = sorted(set(records) & set(test))
                        row[split + "_test_offer_overlap"] = len(intersection)
                        row[split + "_test_identical_offer_attributes"] = sum(records[u] == test[u] for u in intersection)
                        check(f"WDC_{hardness}_{size}_{unseen}_{split}_identical", all(records[u] == test[u] for u in intersection))
                        for uid in intersection[:2]:
                            examples.append({"corner_cases": hardness, "development_size": size, "unseen_percent": unseen,
                                             "split": split, "publisher_offer_id": uid,
                                             "attribute_and_cluster_sha256": hashlib.sha256(json.dumps(test[uid], sort_keys=True).encode()).hexdigest(),
                                             "identical_attributes_and_cluster": records[uid] == test[uid]})
                    development_ids = set(train) | set(valid)
                    row["offer_disjoint_test_pairs"] = sum(str(p["id_left"]) not in development_ids and str(p["id_right"]) not in development_ids for p in test_pairs)
                    old = prior_audit[(prior_audit.corner_cases == hardness) & (prior_audit.development_size == size) & (prior_audit.unseen_percent == unseen)].iloc[0]
                    for key in ("train_test_offer_overlap", "validation_test_offer_overlap", "offer_disjoint_test_pairs", "train_decompressed_sha256", "validation_decompressed_sha256", "test_decompressed_sha256"):
                        check(f"WDC_{hardness}_{size}_{unseen}_{key}_reproduced", str(row[key]) == str(old[key]))
                    audit.append(row)
    check("WDC_18_flagged_cells", sum(r["train_test_offer_overlap"] > 0 or r["validation_test_offer_overlap"] > 0 for r in audit) == 18)
    check("WDC_nine_fully_unseen_cells_disjoint", all(r["train_test_offer_overlap"] == r["validation_test_offer_overlap"] == 0 for r in audit if r["unseen_percent"] == 100))
    write_csv(output / "wdc_split_audit.csv", audit)
    write_json(output / "wdc_overlap_examples.json", examples)
    write_json(output / "wdc_archives.json", archive_info)
    for path, before in inputs.items():
        check("input_unchanged:" + path, sha(project / path) == before)
    check("192_paired_query_rows", len(query_rows) == 192)
    check("24_canonical_maps", len(identity_rows) == 24)
    manifest = {"status": "passed", "run_id": "manuscript-revision-v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "query_rows": len(query_rows), "canonical_maps": len(identity_rows), "wdc_cells": len(audit),
                "checks_passed": len(checks), "checks": checks, "inputs_sha256": inputs,
                "script_sha256": sha(Path(__file__)), "no_retraining_or_policy_changes": True,
                "aggregation": "Query scores and paired differences are arithmetic seed means; canonical totals sum shared-population trials. No pooled inference.",
                "output_sha256": {p.name: sha(p) for p in output.iterdir() if p.is_file()}}
    write_json(output / "verification.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("status", "query_rows", "canonical_maps", "wdc_cells", "checks_passed")}))
    print(paired_means[paired_means["query"] == "Q6"].to_string(index=False))


if __name__ == "__main__":
    main()
