"""Correct projected transaction origin, preserving and replaying fitted models."""
import shutil
import traceback
from datetime import datetime, timezone
import yaml
from kgerp.io import ROOT, read_json, write_json, digest


def main():
    out=ROOT/'results/all-datasets-v1';manifest=read_json(out/'manifest.json')
    if manifest['status']!='completed':raise ValueError('Finish the main timing run before repair')
    relative='src/kgerp/expanded/protocol.py';source=ROOT/relative
    old_text=source.read_text(encoding='utf-8')
    previous=manifest['code_sha256'][relative]
    if digest(source)!=previous:raise ValueError('Unexpected protocol mutation')
    before="'source_table':view['source_table'],\n                    'source_batch':record['source_batch'],'mapper_version':'expanded-v4',"
    after="'source_table':'olist_order_items_dataset.csv' if dataset=='Olist' else view['source_table'],\n                    'source_batch':record['source_batch'],'mapper_version':'expanded-v4',"
    if old_text.count(before)!=1:raise ValueError('Expected one projected-transaction metadata assignment')
    backup=out/'code_amendments/before_Olist_transaction_origin_correction.py'
    shutil.copy2(source,backup)
    source.write_text(old_text.replace(before,after),encoding='utf-8')
    shutil.copy2(source,out/'code_snapshot'/relative)
    manifest['code_sha256'][relative]=digest(source)
    manifest.setdefault('amendments',[]).append({
        'timestamp_utc':datetime.now(timezone.utc).isoformat(),'file':relative,
        'previous_sha256':previous,'new_sha256':digest(source),
        'reason':'Olist projected transaction origin inherited its owner customer filename; correct it to the native order-item file. Date/status/category/seller context derivations remain native joined source fields.',
        'model_control':'No model-visible descriptive/numerical value, metadata completeness, estimator or threshold changes. All frozen probabilities are independently replayed after repair.',
        'revalidation':'Rerun Olist eight-query accuracy controls for all seeds and all nine E5 trials. Preserve previous query metrics and E5 measurements.'})
    manifest['status']='running_lineage_revalidation';write_json(out/'manifest.json',manifest)
    # Import after changing the source, so this fresh interpreter uses the fix.
    from kgerp.expanded.embeddings import FrozenEmbeddings
    from kgerp.expanded.experiments import query_evaluation
    from kgerp.expanded.scaling import benchmark
    from continue_all_datasets import frozen_bundle
    config=yaml.safe_load((out/'config_snapshot.yaml').read_text())
    embedding=FrozenEmbeddings()
    try:
        backup_folder=out/'Olist/before_transaction_origin_correction';backup_folder.mkdir(exist_ok=False)
        shutil.copytree(out/'Olist/e5',backup_folder/'e5')
        for seed in config['seeds']:
            folder=out/'Olist'/f'seed_{seed}'
            target=backup_folder/f'seed_{seed}';target.mkdir()
            for name in ['query_accuracy.csv','query_catalog.json','oracle_query_denominators.json']:
                shutil.copy2(folder/name,target/name)
            bundle=frozen_bundle('Olist',seed,config,folder)
            query_evaluation('Olist',bundle['records'],bundle['graph'],bundle['gold'],bundle['mapping'],folder,seed)
            if (folder/'query_accuracy.csv').read_bytes()!=(target/'query_accuracy.csv').read_bytes():
                raise ValueError('Unexpected query accuracy change from origin-field repair')
            print(f'Olist seed {seed}: corrected source origin, query accuracy unchanged',flush=True)
        benchmark(bundle,config,embedding,out/'Olist')
    except Exception as error:
        manifest.update(status='failed_lineage_revalidation',error=str(error),traceback=traceback.format_exc())
        write_json(out/'manifest.json',manifest)
        raise
    finally:embedding.close()
    shutil.copy2(__file__,out/'lineage_repair_driver.py')
    manifest['status']='completed'
    manifest['result_sha256']={str(p.relative_to(out)).replace('\\','/'):digest(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}
    write_json(out/'manifest.json',manifest)
    print('Olist transaction provenance repair and revalidation completed',flush=True)


if __name__=='__main__':main()
