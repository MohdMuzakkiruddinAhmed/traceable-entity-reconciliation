"""Independent metric, source, policy, experiment-count and query verification."""
import argparse
import json
import csv
import zipfile
import sqlite3
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.metrics import confusion_matrix,precision_recall_fscore_support,average_precision_score
from kgerp.io import ROOT,digest,write_json
from kgerp.expanded.protocol import build_views
from kgerp.expanded.attributes import FastAttributes
from kgerp.expanded.embeddings import FrozenEmbeddings
from kgerp.expanded.experiments import matrix,score
from kgerp.expanded.queries import scoped_graph,PreparedQueries,SQLQueries
from kgerp.expanded.wwi_queries import sql_extra,graph_extra
from kgerp.expanded.wdc_sweep import read_member,attributes,predictions as wdc_predictions
from kgerp.oracle import labels,oracle_mapping
import joblib
import yaml


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def rows(path):
    with path.open(encoding='utf-8',newline='') as handle:return list(csv.DictReader(handle))


def verify(run_id):
    out=ROOT/'results'/run_id;manifest=read(out/'manifest.json');checks=[]
    def check(name,condition,detail=None):
        if not condition:raise AssertionError(name+': '+str(detail))
        checks.append({'check':name,'passed':True,'detail':detail})
    def hashes(name,base,mapping):
        bad=[p for p,h in mapping.items() if not (base/p).is_file() or digest(base/p)!=h]
        check(name,not bad,{'files':len(mapping),'mismatches':bad})
    check('complete_five_dataset_nonhuman_run',manifest['status']=='completed' and set(manifest['datasets'])=={'WWI','WDC','Olist','DataCo','BPI2019'} and manifest['human_study']=='excluded_by_explicit_user_instruction')
    hashes('run_results',out,manifest['result_sha256']);hashes('source_snapshot',out/'code_snapshot',manifest['code_sha256']);hashes('current_code_matches_used_snapshot',ROOT,manifest['code_sha256'])
    hashes('exact_staged_and_view_inputs',ROOT,manifest['input_files_sha256'])
    primary=read(ROOT/'results/initial-core-v3/run_manifest.json');hashes('original_primary_code',ROOT,primary['code_sha256']);hashes('original_primary_results',ROOT/'results/initial-core-v3',primary['result_sha256'])
    improvement=read(ROOT/'results/improvements-v2/manifest.json');hashes('prior_improvement_code',ROOT,improvement['code_sha256']);hashes('prior_improvement_results',ROOT/'results/improvements-v2',improvement['result_sha256'])
    paper=read(ROOT/'paper/verification/manifest.json');check('original_manuscript_pdf_preserved',digest(ROOT/'output/pdf/kg-erp-workshop-draft.pdf')==paper['pdf_sha256'])
    acquisition=read(ROOT/'data/manifests/expanded_downloads.json')
    check('expanded_downloads_verified',all(digest(ROOT/'data/raw'/f['file'])==f['sha256'] for f in acquisition['files']))
    root=ET.parse(ROOT/'results/all-datasets-tests.xml').getroot();suites=[root] if root.tag=='testsuite' else list(root)
    tests=sum(int(s.get('tests',0)) for s in suites)
    check('all_control_tests_pass',tests>=36 and all(int(s.get('errors',0))+int(s.get('failures',0))+int(s.get('skipped',0))==0 for s in suites),{'tests':tests})
    config=yaml.safe_load((out/'config_snapshot.yaml').read_text());embedding=FrozenEmbeddings()
    def metric(name,y,p,accept,row):
        y=np.array(y,dtype=int);p=np.array(p,dtype=float);pred=p>=accept
        if not len(y):
            check(name,all(float(row[k])==0 for k in ['n_pairs','positives','tp','fp','fn','tn','precision','recall','f1','average_precision']),{'empty_stratum':True})
            return
        tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel();precision,recall,f1,_=precision_recall_fscore_support(y,pred,average='binary',zero_division=0)
        actual={'n_pairs':len(y),'positives':int(y.sum()),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':precision,'recall':recall,'f1':f1,'average_precision':average_precision_score(y,p) if y.sum() else 0.}
        check(name,all(abs(float(row[k])-float(v))<1e-12 for k,v in actual.items()))
    try:
        for dataset in ['WWI','Olist','DataCo','BPI2019']:
            source=read(out/dataset/'e1.json')
            if dataset!='WWI':check(dataset+'_full_source_integrity',source['all_reference_sets_exact'] and source['lineage_coverage']==1. and source['dangling_edges']==0 and source['edge_import_idempotent'])
            for seed in config['seeds']:
                folder=out/dataset/f'seed_{seed}';freeze=read(folder/'freeze.json');pred=read(folder/'predictions.json')
                check(f'{dataset}_{seed}_frozen_model',digest(folder/'frozen_models.joblib')==freeze['model_sha256'])
                records,graph,gold,view_manifest=build_views(dataset,seed,config);test=[r for r in records if r['split']=='test']
                if dataset=='Olist':
                    with sqlite3.connect('file:'+str(ROOT/'data/expanded/Olist/source.sqlite').replace('\\','/')+'?mode=ro',uri=True) as db:
                        source_rows={hashlib.sha256(str(uid).encode()).hexdigest() for uid, in db.execute('SELECT __uid FROM raw_items')}
                    transactions=[a for _,a in graph.nodes(data=True) if a['entity_type']=='Transaction']
                    check(f'Olist_{seed}_all_transaction_origin_rows',all(a['source_table']=='olist_order_items_dataset.csv' and a['source_row_digest'] in source_rows for a in transactions),{'checked_transaction_projections':len(transactions)})
                pairs=[(r['left_source_uid'],r['right_source_uid']) for r in pred]
                check(f'{dataset}_{seed}_gold_labels',np.array_equal(labels(pairs,gold),[r['label'] for r in pred]))
                for row in rows(folder/'matching.csv'):
                    subset=pred if row['stratum']=='all' else [r for r in pred if r['ambiguous']]
                    metric(f'{dataset}_{seed}_{row["model"]}_{row["stratum"]}_metric',[r['label'] for r in subset],[r[row['model']] for r in subset],freeze['accepts'][row['model']],row)
                check(f'{dataset}_{seed}_no_context_and_flat_controls',all(abs(r['B4Flat']-r['B5Graph'])<=1e-12 and r['NoContext']==r['B2Attributes'] for r in pred) and freeze['accepts']['NoContext']==freeze['accepts']['B2Attributes'])
                frozen=joblib.load(folder/'frozen_models.joblib');m=matrix(test,graph,frozen['attributes'],embedding,config,pairs=pairs);p,_=score(frozen['models'],m)
                difference=max(float(np.max(np.abs(values-np.array([r[name] for r in pred])))) for name,values in p.items())
                check(f'{dataset}_{seed}_frozen_replay',difference<=1e-12,{'max_difference':difference})
                decisions=read(folder/'decisions.json');accepted=[d for d in decisions if d['status']=='accepted'];audit=read(folder/'identity_audit.json')
                check(f'{dataset}_{seed}_one_to_one_and_false_merges',len({d['left_source_uid'] for d in accepted})==len(accepted)==len({d['right_source_uid'] for d in accepted}) and len(accepted)==audit['accepted_links'] and sum(gold[d['left_source_uid']]['canonical']!=gold[d['right_source_uid']]['canonical'] for d in accepted)==audit['false_merges'])
                mapping={r['source_uid']:r['source_uid'] for r in records}
                for d in accepted:
                    cid='verify:'+d['candidate_id'];mapping[d['left_source_uid']]=mapping[d['right_source_uid']]=cid
                scope=scoped_graph(graph,test);sql=SQLQueries(scope,mapping);answers=PreparedQueries(scope,mapping).execute()
                check(f'{dataset}_{seed}_independent_query_reexecution',answers==sql.execute() and len(answers)==8);sql.close()
                if dataset=='WWI':check(f'WWI_{seed}_native_Q1_Q8_parity',graph_extra(graph,mapping)==sql_extra(graph,mapping))
                query=rows(folder/'query_accuracy.csv')
                check(f'{dataset}_{seed}_three_identity_controls',len(query)==24 and {r['identity'] for r in query}=={'Unreconciled','Predicted','Oracle'} and all(r['SQL_KG_exact']=='True' and int(r['expected_answers'])>0 for r in query))
                stress=rows(folder/'robustness.csv');counts={s:sum(r['scenario']==s for r in stress) for s in {r['scenario'] for r in stress}}
                check(f'{dataset}_{seed}_full_robustness',len(counts)==12 and set(counts.values())=={13} and all(r['same_frozen_pairs']=='True' for r in stress))
                for r in stress:
                    tp,fp,fn,tn=[int(r[k]) for k in ['tp','fp','fn','tn']]
                    check(f'{dataset}_{seed}_{r["scenario"]}_{r["model"]}_stress_denominator',tp+fp+fn+tn==len(pred) and abs(float(r['f1'])-2*tp/max(1,2*tp+fp+fn))<1e-12)
                check(f'{dataset}_{seed}_calibration_and_coverage_artifacts',len(rows(folder/'calibration.csv'))==4 and len(rows(folder/'coverage_risk.csv'))>=16 and len(rows(folder/'paired_ci.csv'))==4)
            scale=rows(out/dataset/'e5/scale_trials.csv')
            check(dataset+'_three_scale_points_three_trials',len(scale)==9 and {int(r['target_nodes']) for r in scale}=={1000,10000,100000} and all(r['status']=='completed' and int(r['actual_nodes'])>=int(r['target_nodes']) and int(r['sampled_peak_rss_bytes'])>0 for r in scale))
            for r in scale:
                check(f'{dataset}_scale_{r["target_nodes"]}_{r["trial"]}_recall_and_budget',int(r['candidate_count'])<=int(r['matching_records'])//2*int(r['per_left_candidate_cap']) and abs(float(r['blocking_recall'])-int(r['retained_true_pairs'])/int(r['true_positive_pair_universe']))<1e-12)
            qs=rows(out/dataset/'e5/query_latencies.csv')
            check(dataset+'_query_scale_parity_and_latency',len(qs)==162 and all(r['same_oracle_identity_map']=='True' and int(r['warm_repetitions'])==5 and float(r['p95_ms'])>=float(r['p50_ms'])>=0 for r in qs))
        splits=rows(out/'WDC/split_integrity.csv');matching=rows(out/'WDC/matching.csv');stress=rows(out/'WDC/robustness.csv')
        check('WDC_27_official_configurations',len(splits)==27 and all(int(r['test_pairs'])==4500 and r['all_official_pairs_retained']=='True' for r in splits))
        check('WDC_81_seed_cells_5_methods_6_stress_conditions',len([r for r in matching if r['stratum']=='official_all'])==405 and len(stress)==2430 and all(r['same_official_pairs']=='True' for r in stress))
        for freeze_path in (out/'WDC').glob('*cc_*/seed_*/freeze.json'):
            folder=freeze_path.parent;freeze=read(freeze_path);prefix=folder.parent.name;hardness=int(prefix.split('cc_')[0]);size=prefix.split('cc_')[1];seed=int(folder.name.split('_')[1])
            check(f'WDC_{hardness}_{size}_{seed}_frozen_model',digest(folder/'frozen_models.joblib')==freeze['model_sha256'])
            frozen=joblib.load(folder/'frozen_models.joblib')
            for unseen in [0,50,100]:
                pred=read(folder/f'predictions_{unseen:03d}un.json')
                with zipfile.ZipFile(ROOT/f'data/raw/wdc-products-{hardness}pair.zip') as archive:
                    official=read_member(archive,f'wdcproducts{hardness}cc{100-hardness}rnd{unseen:03d}un_gs.json.gz')
                check(f'WDC_{hardness}_{size}_{seed}_{unseen}_official_labels_and_pair_ids',official['pair_ids']==[r['pair_id'] for r in pred] and np.array_equal(official['y'],[r['label'] for r in pred]))
                replay,_=wdc_predictions(frozen['models'],attributes(official,frozen['feature'],embedding))
                difference=max(float(np.max(np.abs(values-np.array([r[name] for r in pred])))) for name,values in replay.items())
                check(f'WDC_{hardness}_{size}_{seed}_{unseen}_frozen_replay',difference<=1e-12,{'max_difference':difference})
                for row in matching:
                    if int(row['corner_cases'])==hardness and row['development_size']==size and int(row['seed'])==seed and int(row['unseen_percent'])==unseen and row['stratum']=='official_all':
                        metric(f'WDC_{hardness}_{size}_{seed}_{unseen}_{row["model"]}',[r['label'] for r in pred],[r[row['model']] for r in pred],freeze['accepts'][row['model']],row)
        wscale=rows(out/'WDC/e5/scale_trials.csv')
        check('WDC_three_scale_points_three_trials',len(wscale)==9 and {int(r['source_records']) for r in wscale}=={1000,10000,100000} and all(r['status']=='completed' for r in wscale))
        for r in wscale:check(f'WDC_scale_{r["source_records"]}_{r["trial"]}_recall',abs(float(r['blocking_recall'])-int(r['retained_true_pairs'])/max(1,int(r['true_pair_universe'])))<1e-12)
        supplemental=rows(out/'supplemental_e5/scorer_trials.csv');identity_costs=rows(out/'supplemental_e5/canonicalization_trials.csv')
        check('E5_supplemental_scorer_and_canonicalization_costs',len(supplemental)==135 and len(identity_costs)==36 and all(float(r['scoring_seconds'])>=0 for r in supplemental) and all(float(r['canonicalization_seconds'])>=0 for r in identity_costs))
        for dataset in ['WWI','Olist','DataCo','BPI2019','WDC']:
            for target in [1000,10000,100000]:
                subset=[r for r in supplemental if r['dataset']==dataset and int(r['target'])==target]
                check(f'{dataset}_{target}_same_candidate_scorer_comparison',len(subset)==9 and len({r['candidate_count'] for r in subset})==1 and {int(r['trial']) for r in subset}=={0,1,2})
        typed=rows(out/'wwi_entity_type_metrics.csv')
        check('WWI_supplier_stock_item_breakdowns_complete',len(typed)==156 and {r['entity_type'] for r in typed}=={'Supplier','StockItem'})
        for seed in [17,29,43]:
            pred=read(out/'WWI'/f'seed_{seed}/predictions.json');freeze=read(out/'WWI'/f'seed_{seed}/freeze.json')
            for row in [r for r in typed if int(r['seed'])==seed]:
                subset=[r for r in pred if r['left_source_uid'].split(':')[1]==row['entity_type'] and (row['stratum']=='all' or r['ambiguous'])]
                metric(f'WWI_{seed}_{row["entity_type"]}_{row["stratum"]}_{row["model"]}_metric',[r['label'] for r in subset],[r[row['model']] for r in subset],freeze['accepts'][row['model']],row)
    finally:embedding.close()
    result={'status':'passed','run_id':run_id,'tests':tests,'checks_passed':len(checks),'human_study_excluded':True,
        'verification':'independent scikit-learn metric recomputation, frozen model replay on all graph seeds and all 81 WDC seed/configuration cells, new query execution, source/hash/policy/experiment integrity',
        'not_a_fresh_retraining_replication':True,'checks':checks}
    write_json(out/'verification.json',result);print(json.dumps({k:result[k] for k in ['status','tests','checks_passed']}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',default='all-datasets-v1');verify(parser.parse_args().run_id)
