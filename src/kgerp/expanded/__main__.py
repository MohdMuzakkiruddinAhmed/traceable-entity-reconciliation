"""Execute and checkpoint all applicable non-human studies on five datasets."""
import argparse
import json
from pathlib import Path
import shutil
import traceback
import yaml
from ..io import ROOT,read_json,write_json,digest
from ..evaluations import environment
from .sources import prepare,enrich_olist_cache
from .embeddings import FrozenEmbeddings
from .experiments import evaluate_seed
from .scaling import benchmark
from .wdc_sweep import sweep
from .wdc_scaling import import_integrity,benchmark as wdc_benchmark
from .wwi_queries import evaluate as native_wwi_queries


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',default='all-datasets-v1')
    parser.add_argument('--config',default=str(ROOT/'configs/all_datasets.yaml'))
    parser.add_argument('--datasets',nargs='+',default=['WWI','Olist','DataCo','BPI2019','WDC'])
    args=parser.parse_args()
    if Path(args.run_id).name!=args.run_id or args.run_id in {'.','..'}:raise ValueError('Invalid run ID')
    out=ROOT/'results'/args.run_id;out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise FileExistsError('Run already has a manifest; choose a new ID')
    config=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    original=read_json(ROOT/'results/initial-core-v3/run_manifest.json')
    if any(digest(ROOT/p)!=sha for p,sha in original['code_sha256'].items()):raise ValueError('Original research code changed')
    if any(digest(ROOT/'results/initial-core-v3'/p)!=sha for p,sha in original['result_sha256'].items()):raise ValueError('Original results changed')
    shutil.copy2(args.config,out/'config_snapshot.yaml');shutil.copy2(ROOT/'requirements.fullstudy.lock',out/'requirements.lock')
    for path in (ROOT/'src/kgerp').rglob('*.py'):
        target=out/'code_snapshot'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    code_hashes={str(p.relative_to(out/'code_snapshot')).replace('\\','/'):digest(p) for p in (out/'code_snapshot').rglob('*.py')}
    completed=[]
    manifest={'status':'running','classification':config['classification'],'datasets':args.datasets,'human_study':'excluded_by_explicit_user_instruction',
              'code_sha256':code_hashes,'environment':environment(),'completed':completed,'source_primary_run':'initial-core-v3'}
    write_json(out/'manifest.json',manifest)
    embedding=FrozenEmbeddings()
    try:
        for dataset in args.datasets:
            dataset_out=out/dataset;dataset_out.mkdir(parents=True,exist_ok=True)
            if dataset=='WDC':
                offers,groups=import_integrity(config,dataset_out)
                completed.append('WDC_E1_offer_import_integrity');write_json(out/'manifest.json',manifest)
                sweep(config,embedding,out)
                completed.extend(['WDC_E2_27_configurations_3_seeds','WDC_E4_attribute_robustness_calibration_hardness_unseen'])
                wdc_benchmark(config,embedding,dataset_out,offers,groups);completed.append('WDC_E5_offer_workload')
                completed.append('WDC_E3_ERP_queries_not_applicable_no_transaction_graph')
            else:
                if dataset=='WWI':
                    write_json(dataset_out/'e1.json',{**read_json(ROOT/'results/initial-core-v3/e1/summary.json'),'reuse':'preserved verified full-source WWI E1; new native transaction-context projections checked separately'})
                else:
                    prepare(dataset,dataset_out)
                    if dataset=='Olist':enrich_olist_cache()
                completed.append(dataset+'_E1');write_json(out/'manifest.json',manifest)
                for seed in config['seeds']:
                    bundle=evaluate_seed(dataset,seed,config,embedding,dataset_out)
                    if dataset=='WWI':native_wwi_queries(bundle,dataset_out,seed)
                    completed.append(f'{dataset}_seed_{seed}_E2_E3_E4');write_json(out/'manifest.json',manifest)
                benchmark(bundle,config,embedding,dataset_out);completed.append(dataset+'_E5_3_targets_3_trials')
                del bundle
            write_json(out/'manifest.json',manifest)
        manifest.update({'status':'completed','original_code_unchanged':all(digest(ROOT/p)==sha for p,sha in original['code_sha256'].items()),
                         'original_results_unchanged':all(digest(ROOT/'results/initial-core-v3'/p)==sha for p,sha in original['result_sha256'].items()),
                         'result_sha256':{str(p.relative_to(out)).replace('\\','/'):digest(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}})
        write_json(out/'manifest.json',manifest)
        write_json(ROOT/'results/latest_all_datasets.json',{'run_id':args.run_id,'status':'completed'})
        print('All applicable non-human experiments completed: '+str(out),flush=True)
    except Exception as error:
        manifest.update({'status':'failed','error':str(error),'traceback':traceback.format_exc()})
        write_json(out/'manifest.json',manifest)
        raise
    finally:embedding.close()


if __name__=='__main__':main()
