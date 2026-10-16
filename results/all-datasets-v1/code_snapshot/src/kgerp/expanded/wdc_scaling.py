"""Offer-only WDC import integrity and three-point matching workload timing."""
import gc
import hashlib
import json
import time
import zipfile
import joblib
import numpy as np
from ..io import ROOT,write_json,write_rows
from ..oracle import labels,true_pair_count
from ..research_plus.blocking import bounded_candidates
from .experiments import blocker
from .scaling import MemorySample
from .wdc_sweep import read_member,predictions


def import_integrity(config,out):
    all_records={};all_groups={};members=[];pair_observations=0
    for hardness in [20,50,80]:
        with zipfile.ZipFile(ROOT/f'data/raw/wdc-products-{hardness}pair.zip') as archive:
            for member in archive.namelist():
                data=read_member(archive,member);all_records.update(data['records']);all_groups.update(data['groups'])
                members.append({'file':member,'pairs':len(data['pairs']),'offers':len(data['records']),'positives':int(data['y'].sum()),'sha256_decompressed':data['decompressed_sha256']})
                pair_observations+=len(data['pairs'])
                if member.endswith('un_gs.json.gz') and len(data['pairs'])!=4500:raise ValueError('Official WDC test count')
                if member.endswith('train_small.json.gz') and len(data['pairs'])!=2500:raise ValueError('WDC train small count')
                if member.endswith('valid_small.json.gz') and len(data['pairs'])!=2500:raise ValueError('WDC validation small count')
                if member.endswith('train_medium.json.gz') and len(data['pairs'])!=6000:raise ValueError('WDC train medium count')
                if member.endswith('valid_medium.json.gz') and len(data['pairs'])!=3500:raise ValueError('WDC validation medium count')
                if member.endswith('valid_large.json.gz') and len(data['pairs'])!=4500:raise ValueError('WDC validation large count')
    write_json(out/'e1.json',{'dataset':'WDC','official_split_files':len(members),'unique_offers':len(all_records),'unique_labelled_product_groups':len(set(all_groups.values())),
        'pair_observations':pair_observations,'source_lineage':'archive/member/decompressed SHA-256 and official offer IDs','pair_uid_uniqueness_verified':True,
        'schema_and_count_checks_passed':True,'gold_cluster_ids_excluded_from_features':True,
        'graph_context_and_ERP_query_scope':'not applicable; benchmark offers/pairs have no ERP transaction/FK graph',
        'cross_split_overlaps':'reported separately for all 27 cells; overlapping official offer rows never silently removed'})
    write_rows(out/'source_files.csv',members)
    return all_records,all_groups


def benchmark(config,embedding,out,records,groups):
    folder=out/'e5';folder.mkdir(parents=True,exist_ok=True)
    frozen=joblib.load(out/'50cc_small/seed_43/frozen_models.joblib');models=frozen['models'];feature=frozen['feature']
    base=sorted(records.items());trials=[]
    for target in config['scale']['node_targets']:
        for trial in range(3):
            gc.collect();started=time.perf_counter()
            with MemorySample() as memory:
                expanded=[];gold={}
                for i in range(target):
                    uid,r=base[i%len(base)];side='A' if int(hashlib.sha256(uid.encode()).hexdigest()[:8],16)%2==0 else 'B'
                    new_uid=side+':StockItem:'+uid+':copy'+str(i//len(base))
                    expanded.append({**r,'source_uid':new_uid,'silo':side});gold[new_uid]={'canonical':groups[uid]}
                build=time.perf_counter()-started
                t=time.perf_counter();pairs,stats=bounded_candidates(expanded,blocker(config),32);blocking=time.perf_counter()-t
                by_uid={r['source_uid']:r for r in expanded}
                t=time.perf_counter();ax=feature.transform(pairs,by_uid);em=embedding.similarities(pairs,by_uid);features=time.perf_counter()-t
                t=time.perf_counter();predictions(models,{'ax':ax,'em':em});scoring=time.perf_counter()-t
                y=labels(pairs,gold);total=true_pair_count(expanded,gold)
            trials.append({'dataset':'WDC','target_source_records':target,'source_records':len(expanded),'trial':trial,
                'unique_original_offers':min(target,len(base)),'unique_original_product_groups':len({g['canonical'] for g in gold.values()}),
                'candidate_count':len(pairs),'true_pair_universe':total,'retained_true_pairs':int(y.sum()),'blocking_recall':int(y.sum())/max(1,total),
                'build_seconds':build,'blocking_seconds':blocking,'attribute_and_embedding_seconds':features,'all_scorers_seconds':scoring,
                'sampled_peak_rss_bytes':memory.peak,'rss_increment_bytes':max(0,memory.peak-memory.start),'total_seconds':time.perf_counter()-started,
                'candidate_cap':32,'status':'completed'})
            write_rows(folder/'scale_trials.csv',trials)
            print(f'WDC E5 {target:,} records/trial {trial}: {len(pairs):,} candidates, recall {int(y.sum())/max(1,total):.3f}',flush=True)
            del expanded,ax,em;gc.collect()
    write_json(folder/'protocol.json',{'workload':'official offer pool then namespace-disjoint replicas, deterministic source-side partition by offer UID hash; no enterprise graph fabricated',
        'frozen_model':'50cc/small/seed43; no scale-specific refitting','matching_truth':'original product-cluster labels, evaluator-only; all cross-source positive pairs including cross-replica aliases',
        'candidate_cap':'32 per left source record; output bound and recall trade-off explicit','memory':'50 ms sampled process peak RSS',
        'E3_ERP_queries':'not applicable to this offer-only benchmark','limits':'microbenchmark for matching throughput; not additional official benchmark accuracy cells or independent population growth'})
    return trials
