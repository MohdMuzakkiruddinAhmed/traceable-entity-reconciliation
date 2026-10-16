"""Supplement E5 with matched-candidate scorer and canonicalization costs."""
import argparse
import gc
import hashlib
import shutil
import time
import zipfile
import numpy as np
import joblib
import yaml
from kgerp.io import ROOT, read_json, write_json, write_rows, digest
from kgerp.expanded.embeddings import FrozenEmbeddings
from kgerp.expanded.experiments import matrix, blocker, learner_x
from kgerp.expanded.scaling import workload
from kgerp.expanded.wdc_sweep import read_member,attributes
from kgerp.research_plus.blocking import bounded_candidates
from continue_all_datasets import frozen_bundle


def timed(call, repetitions=3):
    call()  # Explicit unmeasured warm-up.
    measurements=[]
    for _ in range(repetitions):
        started=time.perf_counter();call();measurements.append(time.perf_counter()-started)
    return measurements


def canonicalize(pairs, probabilities, accept):
    used=set();accepted=[]
    for i in sorted(range(len(pairs)),key=lambda i:(-float(probabilities[i]),pairs[i])):
        a,b=pairs[i]
        if probabilities[i]>=accept and a not in used and b not in used:
            used.update([a,b]);accepted.append((a,b))
    return accepted


def main(run_id):
    out=ROOT/'results'/run_id
    manifest=read_json(out/'manifest.json')
    if manifest['status']!='completed':raise ValueError('Complete the core run first')
    config=yaml.safe_load((out/'config_snapshot.yaml').read_text())
    embedding=FrozenEmbeddings();rows=[];identities=[]
    try:
        for dataset in ['WWI','Olist','DataCo','BPI2019']:
            bundle=frozen_bundle(dataset,43,config,out/dataset/'seed_43')
            models=bundle['models']
            for target in config['scale']['node_targets']:
                graph,records,gold,mapping=workload(bundle,target)
                pairs,_=bounded_candidates(records,blocker(config),models['retrieval_k'])
                m=matrix(records,graph,bundle['attribute'],embedding,config,pairs=pairs)
                calls={
                    'B2Attributes':lambda:models['calibration']['attribute'].transform(models['attribute'].predict_proba(m['ax'])[:,1]),
                    'B4Flat':lambda:np.where(m['flat'][:,-1]>0,models['calibration']['weighted'].transform(models['weighted'].predict_proba(np.column_stack([m['ax'],m['flat']]))[:,1]),models['calibration']['attribute'].transform(models['attribute'].predict_proba(m['ax'])[:,1])),
                    'B5Graph':lambda:np.where(m['weighted'][:,-1]>0,models['calibration']['weighted'].transform(models['weighted'].predict_proba(learner_x(m,'weighted'))[:,1]),models['calibration']['attribute'].transform(models['attribute'].predict_proba(m['ax'])[:,1]))}
                p=calls['B5Graph']()
                if not np.allclose(p,calls['B4Flat'](),atol=1e-12,rtol=0):raise ValueError('Flat/graph scoring parity')
                for name,call in calls.items():
                    rows.extend(dict(dataset=dataset,target=target,actual_nodes=len(graph),candidate_count=len(pairs),model=name,trial=trial,scoring_seconds=seconds)
                                for trial,seconds in enumerate(timed(call)))
                call=lambda:canonicalize(pairs,p,models['accepts']['B5Graph'])
                accepted=call()
                identities.extend(dict(dataset=dataset,target=target,candidate_count=len(pairs),trial=trial,canonicalization_seconds=seconds,
                    accepted_links=len(accepted),false_merges=sum(gold[a]['canonical']!=gold[b]['canonical'] for a,b in accepted))
                    for trial,seconds in enumerate(timed(call)))
                print(f'{dataset} E5 individual scorers/canonicalization at {target:,} complete',flush=True)
                del graph,records,m,p;gc.collect()
            del bundle;gc.collect()
        frozen=joblib.load(out/'WDC/50cc_small/seed_43/frozen_models.joblib');models=frozen['models']
        with zipfile.ZipFile(ROOT/'data/raw/wdc-products-50pair.zip') as archive:
            official=read_member(archive,'wdcproducts50cc50rnd100un_gs.json.gz')
        observed=attributes(official,frozen['feature'],embedding)
        for target in config['scale']['node_targets']:
            # This supplementary WDC experiment isolates model scoring at
            # pair-volume points. Main E5 already measures offer remapping,
            # blocking, full feature extraction and scoring, three trials.
            index=np.arange(target)%len(official['pairs'])
            ax=observed['ax'][index];em=observed['em'][index]
            inputs={'attribute':ax,'embedding':em,'augmented':np.column_stack([ax,em])}
            for name,key in [('B2Attributes','attribute'),('B3Embedding','embedding'),('B2EmbeddingAugmented','augmented')]:
                call=lambda key=key:models['calibration'][key].transform(models[key].predict_proba(inputs[key])[:,1])
                rows.extend(dict(dataset='WDC',target=target,actual_nodes=len(official['records']),candidate_count=target,model=name,trial=trial,scoring_seconds=seconds)
                    for trial,seconds in enumerate(timed(call)))
            print(f'WDC supplemental scorers at {target:,} observed-pair rows complete',flush=True)
            del ax,em,inputs;gc.collect()
    finally:embedding.close()
    folder=out/'supplemental_e5';folder.mkdir(exist_ok=True)
    write_rows(folder/'scorer_trials.csv',rows);write_rows(folder/'canonicalization_trials.csv',identities)
    shutil.copy2(__file__,folder/'profile_driver.py')
    write_json(folder/'protocol.json',{
        'status':'completed','same_candidates_within_each_dataset_and_target':True,
        'models':'Frozen seed 43; WDC frozen 50cc/small/seed43; no refitting or changed thresholds.',
        'size_semantics':'Four graph datasets reuse main E5 source-view node targets. Supplemental WDC scorer points are 1k/10k/100k pair rows from the official fully unseen test, extended by exact feature-row replicas; actual_nodes is the independent offer pool. Main WDC E5 retains all nine full offer-volume/blocking/feature/scoring trials.',
        'timing':'One warm-up then three measured repeats of each scorer, calibration and attribute fallback included. Shared feature construction is outside these scorer-only timings and remains separately measured in the main E5.',
        'canonicalization':'Greedy descending-probability one-to-one acceptance on identical bounded pairs; rejected/abstained endpoints retain source identity. Measured separately from oracle-controlled query timing.',
        'limits':'Supplementary cost comparison; model-only timing ratios are not end-to-end graph/attribute overhead ratios. WDC canonicalization/ERP queries are inapplicable.'})
    # Record exact staged/view inputs as well as acquired raw bytes. Embedding
    # caches are derived, mutable accelerators and are deliberately excluded.
    inputs=list((ROOT/'data/staged').glob('*.json'))
    for dataset in ['Olist','DataCo','BPI2019']:
        inputs.extend(ROOT/'data/expanded'/dataset/name for name in ['source.sqlite','source_manifest.json','base_views.json'])
    for seed in [17,29,43]:
        inputs.extend([ROOT/f'data/generated/seed_{seed}/records.json',ROOT/f'data/generated/seed_{seed}/graph.pkl',ROOT/f'data/gold/seed_{seed}/truth.json'])
    amendment=ROOT/'data/expanded/Olist/view_adapter_amendment.json'
    if amendment.exists():inputs.append(amendment)
    manifest['input_files_sha256']={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in inputs}
    for p in folder.iterdir():manifest['result_sha256'][str(p.relative_to(out)).replace('\\','/')]=digest(p)
    manifest.setdefault('supplemental_experiments',[]).append('E5_equal_candidate_individual_scorers_and_canonicalization')
    write_json(out/'manifest.json',manifest)
    print('Supplemental E5 profiling completed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',default='all-datasets-v1');main(parser.parse_args().run_id)
