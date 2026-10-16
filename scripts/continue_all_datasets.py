"""Resume the interrupted five-dataset run, preserving completed model fits."""
import argparse
import hashlib
import shutil
import traceback
from datetime import datetime, timezone

import joblib
import yaml
from kgerp.io import ROOT, read_json, write_json, digest
from kgerp.expanded.sources import prepare, enrich_olist_cache
from kgerp.expanded.protocol import build_views
from kgerp.expanded.queries import scoped_graph
from kgerp.expanded.embeddings import FrozenEmbeddings
from kgerp.expanded.experiments import evaluate_seed
from kgerp.expanded.scaling import benchmark
from kgerp.expanded.wdc_sweep import sweep
from kgerp.expanded.wdc_scaling import import_integrity, benchmark as wdc_benchmark
from kgerp.expanded.wwi_queries import evaluate as native_wwi_queries


def frozen_bundle(dataset, seed, config, folder):
    records, graph, gold, _ = build_views(dataset, seed, config)
    frozen = joblib.load(folder / 'frozen_models.joblib')
    if digest(folder / 'frozen_models.joblib') != read_json(folder / 'freeze.json')['model_sha256']:
        raise ValueError('Frozen model checksum mismatch')
    test = [r for r in records if r['split'] == 'test']
    mapping = {r['source_uid']: r['source_uid'] for r in records}
    for decision in read_json(folder / 'decisions.json'):
        if decision['status'] == 'accepted':
            a, b = decision['left_source_uid'], decision['right_source_uid']
            mapping[a] = mapping[b] = 'canonical:' + hashlib.sha256((a + '|' + b).encode()).hexdigest()[:24]
    return dict(dataset=dataset, records=records, graph=graph, gold=gold,
                test_records=test, scope=scoped_graph(graph, test), mapping=mapping,
                attribute=frozen['attributes'], models=frozen['models'])


def main(run_id):
    out = ROOT / 'results' / run_id
    manifest = read_json(out / 'manifest.json')
    if manifest['status'] == 'completed':
        raise ValueError('Run already completed')
    config = yaml.safe_load((out / 'config_snapshot.yaml').read_text(encoding='utf-8'))
    completed = manifest['completed']
    original = read_json(ROOT / 'results/initial-core-v3/run_manifest.json')
    assert all(digest(ROOT / p) == h for p, h in original['code_sha256'].items())
    assert all(digest(ROOT / 'results/initial-core-v3' / p) == h for p, h in original['result_sha256'].items())
    changed = [p for p, h in manifest['code_sha256'].items() if digest(ROOT / p) != h]
    amendment_path = 'src/kgerp/expanded/queries.py'
    if changed:
        if changed != [amendment_path]:
            raise ValueError('Unexpected source changes: ' + str(changed))
        old = out / 'code_snapshot' / amendment_path
        backup = out / 'code_amendments/before_Q1_optimization.py'
        backup.parent.mkdir(parents=True, exist_ok=True)
        if backup.exists():
            raise FileExistsError(backup)
        shutil.copy2(old, backup)
        prior_e5 = out / 'WWI/e5-before-Q1-optimization'
        if (out / 'WWI/e5').exists():
            shutil.copytree(out / 'WWI/e5', prior_e5)
        previous_hash = manifest['code_sha256'][amendment_path]
        shutil.copy2(ROOT / amendment_path, old)
        manifest['code_sha256'][amendment_path] = digest(old)
        manifest.setdefault('amendments', []).append({
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'file': amendment_path, 'previous_sha256': previous_hash,
            'new_sha256': digest(old),
            'reason': 'Q1 deduplicates source entity/category incidence before joining reconciled sources, avoiding transaction Cartesian products; SQL/graph answer parity tested.',
            'preservation': 'Prior Q1 source and six completed WWI E5 trials preserved; all WWI E5 trials rerun consistently with the revised SQL. Matching estimators, features, thresholds and predictions unchanged.'})
    manifest['status'] = 'running'
    manifest.pop('error', None); manifest.pop('traceback', None)
    manifest.setdefault('continuations', []).append({
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'reason': 'User requested continuation after interrupted execution; completed model fits retained.',
        'driver_sha256': digest(__file__)})
    write_json(out / 'manifest.json', manifest)
    embedding = FrozenEmbeddings()

    def checkpoint(item):
        if item not in completed:
            completed.append(item)
        write_json(out / 'manifest.json', manifest)

    try:
        for dataset in manifest['datasets']:
            dataset_out = out / dataset
            dataset_out.mkdir(parents=True, exist_ok=True)
            print('Continuing ' + dataset, flush=True)
            if dataset == 'WDC':
                offers, groups = import_integrity(config, dataset_out)
                checkpoint('WDC_E1_offer_import_integrity')
                if 'WDC_E2_27_configurations_3_seeds' not in completed:
                    sweep(config, embedding, out)
                    checkpoint('WDC_E2_27_configurations_3_seeds')
                    checkpoint('WDC_E4_attribute_robustness_calibration_hardness_unseen')
                if 'WDC_E5_offer_workload' not in completed:
                    wdc_benchmark(config, embedding, dataset_out, offers, groups)
                    checkpoint('WDC_E5_offer_workload')
                checkpoint('WDC_E3_ERP_queries_not_applicable_no_transaction_graph')
                continue
            if dataset + '_E1' not in completed:
                if dataset == 'WWI':
                    raise ValueError('Expected completed WWI source audit')
                prepare(dataset, dataset_out)
                if dataset == 'Olist':
                    enrich_olist_cache()
                checkpoint(dataset + '_E1')
            bundle = None
            for seed in config['seeds']:
                item = f'{dataset}_seed_{seed}_E2_E3_E4'
                if item in completed:
                    if seed == config['seeds'][-1]:
                        bundle = frozen_bundle(dataset, seed, config, dataset_out / f'seed_{seed}')
                    continue
                folder = dataset_out / f'seed_{seed}'
                if (folder / 'freeze.json').exists():
                    raise ValueError('Uncheckpointed frozen fit requires explicit artifact recovery: ' + str(folder))
                bundle = evaluate_seed(dataset, seed, config, embedding, dataset_out)
                if dataset == 'WWI':
                    native_wwi_queries(bundle, dataset_out, seed)
                checkpoint(item)
            item = dataset + '_E5_3_targets_3_trials'
            if item not in completed:
                benchmark(bundle, config, embedding, dataset_out)
                checkpoint(item)
            del bundle
        shutil.copy2(__file__, out / 'continuation_driver.py')
        manifest.update(status='completed',
            original_code_unchanged=all(digest(ROOT / p) == h for p, h in original['code_sha256'].items()),
            original_results_unchanged=all(digest(ROOT / 'results/initial-core-v3' / p) == h for p, h in original['result_sha256'].items()),
            result_sha256={str(p.relative_to(out)).replace('\\', '/'): digest(p) for p in out.rglob('*') if p.is_file() and p.name != 'manifest.json'})
        write_json(out / 'manifest.json', manifest)
        write_json(ROOT / 'results/latest_all_datasets.json', {'run_id': run_id, 'status': 'completed'})
        print('All applicable five-dataset experiments completed', flush=True)
    except Exception as error:
        manifest.update(status='failed', error=str(error), traceback=traceback.format_exc())
        write_json(out / 'manifest.json', manifest)
        raise
    finally:
        embedding.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='all-datasets-v1')
    main(parser.parse_args().run_id)
