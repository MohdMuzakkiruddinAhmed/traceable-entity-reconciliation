"""Recompute extension metrics and inspect artifact integrity without retraining."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import pickle
import xml.etree.ElementTree as ET

import joblib
import numpy as np
import yaml
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, average_precision_score
from kgerp.blocking import candidates
from kgerp.oracle import labels
from kgerp.queries import sql_database, sql_answers, graph_answers
from kgerp.research_plus.protocol import partition_validation, role_masks
from kgerp.research_plus.runner import load_seed, matrices, method_scores
from kgerp.research_plus.queries import PreparedGraphQueries

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def verify(run_id, junit):
    out = ROOT / 'results' / run_id
    old = ROOT / 'results/initial-core-v3'
    manifest = read(out / 'manifest.json')
    checks = []

    def check(name, condition, detail=None):
        if not condition:
            raise AssertionError(f'{name}: {detail}')
        checks.append({'check': name, 'passed': True, 'detail': detail})

    def hashes(name, base, mapping):
        failures = [p for p, expected in mapping.items() if not (base / p).is_file() or sha(base / p) != expected]
        check(name, not failures, {'files': len(mapping), 'mismatches': failures})

    check('completed_exploratory_run', manifest['status'] == 'completed' and manifest['classification'] == 'exploratory_post_test_extension')
    hashes('extension_results', out, manifest['result_sha256'])
    hashes('extension_code', ROOT, manifest['code_sha256'])
    check('extension_configuration', sha(ROOT / 'configs/improvements.yaml') == manifest['config_sha256'])
    snapshot = read(out / 'code_snapshot_manifest.json')
    hashes('archived_code_snapshot', out / 'code_snapshot', snapshot)
    hashes('current_code_matches_snapshot', ROOT, snapshot)
    primary = read(old / 'run_manifest.json')
    hashes('preserved_primary_code', ROOT, primary['code_sha256'])
    hashes('preserved_primary_results', old, primary['result_sha256'])
    paper = read(ROOT / 'paper/verification/manifest.json')
    check('preserved_original_paper', sha(ROOT / 'output/pdf/kg-erp-workshop-draft.pdf') == paper['pdf_sha256'])
    check('separate_latest_pointers', read(ROOT / 'results/latest.json')['run_id'] == 'initial-core-v3' and read(ROOT / 'results/latest_improvements.json')['run_id'] == run_id)

    tree = ET.parse(ROOT / junit)
    suites = [tree.getroot()] if tree.getroot().tag == 'testsuite' else list(tree.getroot())
    test_count = sum(int(s.get('tests', 0)) for s in suites)
    check('locked_environment_tests', test_count >= 25 and all(int(s.get('failures', 0)) + int(s.get('errors', 0)) + int(s.get('skipped', 0)) == 0 for s in suites), {'tests': test_count, 'junit': junit})

    def metric_check(name, y, probabilities, accept, reported):
        pred = np.asarray(probabilities) >= accept
        y = np.asarray(y, dtype=int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average='binary', zero_division=0)
        values = {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn, 'n_pairs': len(y), 'positives': int(y.sum()), 'precision': precision, 'recall': recall, 'f1': f1,
                  'average_precision': average_precision_score(y, probabilities)}
        check(name, all(abs(float(reported[k]) - float(v)) < 1e-12 for k, v in values.items()))

    config = yaml.safe_load((out / 'config_snapshot.yaml').read_text())
    base = yaml.safe_load((old / 'config_snapshot.yaml').read_text())
    for seed in config['seeds']:
        folder = out / f'wwi/seed_{seed}'
        pred = read(folder / 'predictions.json')
        freeze = read(folder / 'freeze.json')
        legacy = read(old / f'e2/seed_{seed}/freeze.json')['thresholds']
        accepts = {**freeze['thresholds'], 'OriginalB2': legacy['B2'], 'OriginalB5': legacy['B5']}
        check(f'WWI_{seed}_frozen_model_hash', sha(folder / 'frozen_models.joblib') == freeze['model_sha256'])
        records, graph, gold = load_seed(seed)
        test_records = [r for r in records if r['split'] == 'test']
        pairs = [(r['left_source_uid'], r['right_source_uid']) for r in pred]
        expected_pairs, _ = candidates(test_records, base['blocker'])
        check(f'WWI_{seed}_common_candidate_order_and_truth', pairs == expected_pairs and np.array_equal(labels(pairs, gold), [r['label'] for r in pred]))
        for row in rows(folder / 'matching.csv'):
            subset = pred if row['stratum'] == 'all' else [r for r in pred if r['ambiguous']]
            metric_check(f'WWI_{seed}_{row["model"]}_{row["stratum"]}', [r['label'] for r in subset], [r[row['model']] for r in subset], accepts[row['model']], row)
        check(f'WWI_{seed}_representation_and_ablation_parity', all(r['FlatWeighted'] == r['GraphWeighted'] and r['GraphNoContext'] == r['AttrPlus'] for r in pred))
        bundle = joblib.load(folder / 'frozen_models.joblib')
        matrix = matrices(test_records, graph, bundle['features'], base['blocker'], config['models'], pairs)
        scores, _ = method_scores(bundle['models'], matrix)
        max_difference = max(float(np.max(np.abs(p - np.array([r[name] for r in pred])))) for name, p in scores.items())
        check(f'WWI_{seed}_frozen_model_replay', max_difference <= 1e-12, {'max_difference': max_difference})
        validation = [r for r in records if r['split'] == 'validation']
        vpairs, _ = candidates(validation, base['blocker'])
        assignment, groups = partition_validation(validation, gold, config['development_partition_seed'])
        masks = role_masks(vpairs, assignment)
        role_sets = [set(group) for group in groups.values()]
        check(f'WWI_{seed}_development_group_disjoint', all(not a.intersection(b) for i, a in enumerate(role_sets) for b in role_sets[i+1:]))
        check(f'WWI_{seed}_role_denominators', all(int(mask.sum()) == freeze['validation_roles'][role]['pairs'] and len(groups[role]) == freeze['validation_roles'][role]['canonical_groups'] for role, mask in masks.items()) and len(vpairs) - sum(int(m.sum()) for m in masks.values()) == freeze['pairs_between_roles_omitted'])
        decisions = read(folder / 'decisions.json')
        audit = read(folder / 'identity_audit.json')
        with (folder / 'canonical_graph.pkl').open('rb') as handle:
            identity_graph = pickle.load(handle)
        check(f'WWI_{seed}_physical_view_preserves_sources', all(identity_graph.nodes[u] == graph.nodes[u] for u in graph) and all(dict(identity_graph[a][b][k]) == dict(graph[a][b][k]) for a, b, k in graph.edges(keys=True)))
        mapping = {u: u for u in graph}
        for a, b, attrs in identity_graph.edges(data=True):
            if attrs.get('relation') == 'REFERS_TO':
                mapping[a] = b
        prepared = PreparedGraphQueries(graph, mapping).execute()
        with sql_database(graph, mapping) as sql:
            sql_result = sql_answers(sql)
        check(f'WWI_{seed}_six_query_answer_parity', prepared == sql_result == graph_answers(graph, mapping) and len(prepared) == 6)
        accepted = [d for d in decisions if d['status'] == 'accepted']
        check(f'WWI_{seed}_accepted_identity_audit', len(accepted) == audit['accepted_links'] and sum(gold[d['left_source_uid']]['canonical'] != gold[d['right_source_uid']]['canonical'] for d in accepted) == audit['false_merges'] and all(d['feature_values'] and d['method_version'] and d['decision_trigger'] for d in accepted))
        stress = rows(folder / 'robustness.csv')
        counts = Counter(r['scenario'] for r in stress)
        check(f'WWI_{seed}_robustness_completeness', len(counts) == 12 and set(counts.values()) == {9} and all(r['same_frozen_candidates'] == 'True' for r in stress))
        for r in stress:
            tp, fp, fn, tn = (int(r[k]) for k in ['tp', 'fp', 'fn', 'tn'])
            check(f'WWI_{seed}_{r["scenario"]}_{r["model"]}_stress_counts', tp+fp+fn+tn == len(pred) and abs(float(r['f1']) - 2*tp/max(1, 2*tp+fp+fn)) < 1e-12)
        first = ROOT / f'results/improvements-v1/wwi/seed_{seed}/predictions.json'
        if first.exists():
            v1 = read(first)
            check(f'WWI_{seed}_v1_v2_accuracy_unchanged', len(pred) == len(v1) and all(a['candidate_id'] == b['candidate_id'] and all(abs(a[k] - b[k]) <= 1e-12 for k in accepts if k != 'FlatWeighted') for a, b in zip(pred, v1)))

    wp = read(out / 'wdc/predictions.json')
    wf = read(out / 'wdc/freeze.json')
    check('WDC_frozen_model', sha(out / 'wdc/frozen_model.joblib') == wf['model_sha256'])
    legacy = read(old / 'e2/wdc/predictions.json')
    check('WDC_official_test_order_labels_scores', len(wp) == 4500 and sum(r['label'] for r in wp) == 500 and all(a['pair_id'] == b['pair_id'] and a['label'] == b['label'] and abs(a['OriginalB2'] - b['B2']) <= 1e-12 for a, b in zip(wp, legacy)))
    for row in rows(out / 'wdc/matching.csv'):
        accept = wf['accept'] if row['model'] == 'AttrPlus' else read(old / 'e2/wdc/freeze.json')['B2_threshold']
        metric_check(f'WDC_{row["model"]}_metrics', [r['label'] for r in wp], [r[row['model']] for r in wp], accept, row)
    check('WDC_development_denominators', sum(r['groups'] for r in wf['validation_roles'].values()) == 500 and sum(r['pairs'] for r in wf['validation_roles'].values()) + wf['validation_cross_role_pairs_omitted'] == 2500)

    bs = rows(out / 'e5/blocking_scale.csv')
    check('E5_blocking_trials', len(bs) == 18 and set(int(r['factor']) for r in bs) == {1, 3, 10} and set(int(r['trial']) for r in bs) == {0, 1, 2})
    for r in bs:
        universe = 49 * int(r['factor']) ** 2
        valid = int(r['true_pair_universe']) == universe and abs(float(r['blocking_recall']) - int(r['retained_true_pairs']) / universe) < 1e-12
        if r['blocker'] == 'bounded':
            valid &= int(r['candidate_count']) <= (int(r['master_records']) // 2) * int(r['per_left_limit']) and r['cardinality_bound_respected'] == 'True' and float(r['full_feature_seconds']) >= 0
        check(f'E5_blocking_{r["factor"]}_{r["trial"]}_{r["blocker"]}', valid)
    qs = rows(out / 'e5/query_scale.csv')
    check('E5_query_trials_and_same_identity_controls', len(qs) == 135 and all(r['same_identities'] == 'True' and int(r['warm_repetitions']) == 5 and 0 <= float(r['latency_p50_ms']) <= float(r['latency_p95_ms']) for r in qs))
    result = {'status': 'passed', 'run_id': run_id, 'verification_kind': 'independent_sklearn_metric_recomputation_and_frozen_artifact_replay',
              'not_a_fresh_retraining_claim': True, 'tests': test_count, 'checks_passed': len(checks), 'checks': checks}
    (out / 'verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'passed', 'tests': test_count, 'checks': len(checks)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='improvements-v2')
    parser.add_argument('--junit', default='results/improvements-tests.xml')
    args = parser.parse_args()
    verify(args.run_id, args.junit)
