"""P9 internal replication in a fresh locked environment, without replacing runs.

Regenerates all WWI aliases, retrains on train/validation only, and checks frozen
test probabilities/decisions and query answers. Repeats WDC attribute fitting.
E5 hardware timings are intentionally not required to match.
"""
import argparse
from pathlib import Path
import sys
import time
import joblib
import numpy as np
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from kgerp.io import ROOT, digest, read_json, write_json, write_rows
from kgerp.ingest import load_tables
from kgerp.evaluations import development, environment
from kgerp.silo_generator import generate
from kgerp.blocking import pair_id
from kgerp.matchers import score, acceptance, fit_learned, threshold
from kgerp.features_attribute import AttributeFeatures
from kgerp.identity import resolve
from kgerp.queries import graph_answers, sql_database, sql_answers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-run', default='initial-core-v3')
    parser.add_argument('--run-id', default='p9-replication-v1')
    args = parser.parse_args()
    for value in [args.source_run, args.run_id]:
        if Path(value).name != value or value in {'.', '..'}:
            raise ValueError('Invalid run name')
    source = ROOT / 'results' / args.source_run
    out = ROOT / 'results' / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load((source / 'config_snapshot.yaml').read_text(encoding='utf-8'))
    manifest = read_json(source / 'run_manifest.json')
    if manifest['status'] != 'completed':
        raise ValueError('Source experiment was not completed')
    started = time.perf_counter()
    write_json(out / 'manifest.json', {'status': 'running', 'source_run': args.source_run})
    checks = []
    try:
        for path, checksum in manifest['code_sha256'].items():
            if digest(ROOT / path) != checksum:
                raise ValueError(f'Frozen source changed: {path}')
        for path, checksum in manifest['result_sha256'].items():
            if digest(source / path) != checksum:
                raise ValueError(f'Frozen result changed: {path}')
        for table, snapshot in read_json(source / 'input_manifests/wwi_snapshot.json').items():
            if digest(ROOT / 'data/staged' / (table + '.json')) != snapshot['sha256']:
                raise ValueError(f'Input checksum mismatch: {table}')
        checks.append({'dataset': 'inputs', 'seed': '', 'check': 'frozen_code_results_and_WWI_source_hashes', 'status': 'passed', 'max_probability_difference': 0.})
        tables = load_tables()
        for seed in config['seeds']:
            records, graph, gold = generate(tables, seed, config['corruption'], config['group_split_seed'])
            if records != read_json(ROOT / f'data/generated/seed_{seed}/records.json'):
                raise ValueError(f'Generated records changed for seed {seed}')
            if gold != read_json(ROOT / f'data/gold/seed_{seed}/truth.json'):
                raise ValueError(f'Generated truth changed for seed {seed}')
            features, models, matrices = development(records, graph, gold, config)
            test = matrices['test']
            scored = score(models, test['ax'], test['cx'])
            original = read_json(source / f'e2/seed_{seed}/predictions.json')
            by_id = {p['candidate_id']: p for p in original}
            if len(by_id) != len(test['pairs']):
                raise ValueError('Candidate count changed')
            maximum = 0.
            for i, (a, b) in enumerate(test['pairs']):
                old = by_id[pair_id(a, b)]
                if old['left_source_uid'] != a or old['right_source_uid'] != b:
                    raise ValueError('Candidate identity changed')
                for name in ['B0', 'B1', 'B2', 'B4', 'B5']:
                    difference = abs(float(scored[name][i]) - old[name])
                    maximum = max(maximum, difference)
                    if difference > 1e-12:
                        raise ValueError(f'{seed}/{name} probability mismatch: {difference}')
            freeze = read_json(source / f'e2/seed_{seed}/freeze.json')
            for name, old_threshold in freeze['thresholds'].items():
                if abs(acceptance(models, name) - old_threshold) > 1e-12:
                    raise ValueError(f'{seed}/{name} threshold changed')
            all_records = {r['source_uid']: r for r in records}
            mapping, decisions = resolve(test['pairs'], scored['B5'], all_records, test['evidence'],
                                         acceptance(models, 'B5'), models['reject'], 'replication', 'replication')
            old_decisions = read_json(source / f'e2/seed_{seed}/decisions.json')
            if {(d['candidate_id'], d['status']) for d in decisions} != {(d['candidate_id'], d['status']) for d in old_decisions}:
                raise ValueError('Accepted/rejected/abstained decisions changed')
            actual = graph_answers(graph, mapping)
            old = read_json(source / f'e3/seed_{seed}/answers.json')['R1_predicted_KG']
            if actual != {q: {tuple(row) for row in answer} for q, answer in old.items()}:
                raise ValueError('Reconstructed query answers changed')
            with sql_database(graph, mapping) as sql:
                if sql_answers(sql) != actual:
                    raise ValueError('Replicated SQL/graph parity failed')
            joblib.dump({'features': features, 'models': models}, out / f'wwi_seed_{seed}.joblib')
            checks.append({'dataset': 'WWI', 'seed': seed, 'check': 'aliases_retraining_thresholds_probabilities_identity_and_six_queries',
                           'status': 'passed', 'max_probability_difference': maximum})
            print(f'Replicated WWI seed {seed}: all probabilities, decisions and query answers agree', flush=True)
        data = {s: read_json(ROOT / 'data/wdc' / (s + '.json')) for s in ['train', 'validation', 'test']}
        features = AttributeFeatures().fit([r for p in data['train'] for r in [p['left'], p['right']]])
        matrices = {}
        for split, pairs in data.items():
            records = {r['source_uid']: r for p in pairs for r in [p['left'], p['right']]}
            matrices[split] = (features.transform([(p['left']['source_uid'], p['right']['source_uid']) for p in pairs], records),
                               np.array([p['label'] for p in pairs]))
        tx, ty = matrices['train']; vx, vy = matrices['validation']; test_x, y = matrices['test']
        learned = fit_learned(tx, ty, vx, vy, [.1, 1., 10.])
        fuzzy = threshold(vx[:, 1], vy, [.55, .65, .75, .85, .95])
        probabilities = learned['estimator'].predict_proba(test_x)[:, 1]
        old = read_json(source / 'e2/wdc/predictions.json')
        maximum = float(np.max(np.abs(probabilities - np.array([p['B2'] for p in old]))))
        if maximum > 1e-12 or not np.array_equal(y, [p['label'] for p in old]):
            raise ValueError('Replicated WDC score/label mismatch')
        freeze = read_json(source / 'e2/wdc/freeze.json')
        if not np.isclose(learned['accept'], freeze['B2_threshold'], atol=1e-12, rtol=0) or fuzzy != freeze['B1_threshold']:
            raise ValueError('Replicated WDC thresholds differ')
        joblib.dump({'features': features, 'model': learned}, out / 'wdc_model.joblib')
        checks.append({'dataset': 'WDC', 'seed': '', 'check': 'attribute_retraining_test_probabilities_and_thresholds', 'status': 'passed', 'max_probability_difference': maximum})
        write_rows(out / 'checks.csv', checks)
        write_json(out / 'manifest.json', {'status': 'completed', 'source_run': args.source_run, 'checks': checks,
                   'environment': environment(), 'executable': sys.executable, 'script_sha256': digest(Path(__file__)),
                   'requirements_sha256': digest(ROOT / 'requirements.lock'), 'duration_seconds': time.perf_counter() - started,
                   'scope': 'all three WWI primary seeds, identity/query replay and WDC attribute refit; no retuning or E5 timing comparison'})
        print(f'Clean-environment replication passed: {out}', flush=True)
    except Exception as exc:
        write_json(out / 'manifest.json', {'status': 'failed', 'error': str(exc), 'checks': checks})
        raise

if __name__ == '__main__':
    main()
