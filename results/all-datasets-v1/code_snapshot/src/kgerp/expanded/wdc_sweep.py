"""All 27 official WDC pairwise configurations, three training seeds each."""
from collections import defaultdict
import copy
import gzip
import hashlib
import json
import random
import time
import zipfile
import joblib
import numpy as np
from ..io import ROOT,write_json,write_rows,digest,utc_now
from ..metrics import binary,paired_bootstrap
from ..research_plus.learning import select_model,SigmoidCalibration,prior_weights,choose_threshold,reliability,coverage_risk
from ..research_plus.protocol import role_masks,ROLES
from .attributes import FastAttributes

FIELDS={'title':'name','description':'description','brand':'brand','price':'price','priceCurrency':'currency'}
METHODS=['B0Exact','B1Fuzzy','B2Attributes','B3Embedding','B2EmbeddingAugmented']


def read_member(archive,member):
    payload=gzip.decompress(archive.read(member))
    rows=[json.loads(line) for line in payload.decode('utf-8').splitlines() if line.strip()]
    records={};pairs=[];groups={};offer_text={}
    for row in rows:
        pair=[]
        for side in ['left','right']:
            uid='wdc:'+str(row['id_'+side]);record={'source_uid':uid,'entity_type':'StockItem','silo':side,'category':''}
            record.update({target:'' if row.get(field+'_'+side) is None else str(row[field+'_'+side]) for field,target in FIELDS.items()})
            if uid in records and any(record[k]!=records[uid][k] for k in FIELDS.values()):raise ValueError('Inconsistent WDC offer attributes')
            records[uid]=record;groups[uid]=str(row['cluster_id_'+side]);pair.append(uid)
            offer_text[uid]=hashlib.sha256(json.dumps({k:record[k] for k in FIELDS.values()},sort_keys=True).encode()).hexdigest()
        pairs.append(tuple(pair))
    if len({r['pair_id'] for r in rows})!=len(rows):raise ValueError('Duplicate official WDC pairs')
    return {'records':records,'pairs':pairs,'groups':groups,'y':np.array([int(r['label']) for r in rows]),
            'hard':np.array([bool(r.get('is_hard_negative',False)) or bool(r['label']) for r in rows]),
            'pair_ids':[r['pair_id'] for r in rows],'offer_text':offer_text,'decompressed_sha256':hashlib.sha256(payload).hexdigest(),'member':member}


def x_for(m,key):
    return m['ax'] if key=='attribute' else m['em'] if key=='embedding' else np.column_stack([m['ax'],m['em']])


def predictions(models,m):
    raw={k:models[k].predict_proba(x_for(m,k))[:,1] for k in ['attribute','embedding','augmented']}
    return {'B0Exact':((m['ax'][:,0]==1)&(m['ax'][:,9]==0)).astype(float),'B1Fuzzy':m['ax'][:,1],
            'B2Attributes':models['calibration']['attribute'].transform(raw['attribute']),
            'B3Embedding':models['calibration']['embedding'].transform(raw['embedding']),
            'B2EmbeddingAugmented':models['calibration']['augmented'].transform(raw['augmented'])},raw


def attributes(data,feature,embedding):
    return {**data,'ax':feature.transform(data['pairs'],data['records']),'em':embedding.similarities(data['pairs'],data['records'])}


def sweep(config,embedding,out):
    out=out/'WDC';out.mkdir(parents=True,exist_ok=True)
    all_metrics=[];all_robust=[];all_cal=[];split_audits=[];timing=[]
    for hardness in config['wdc']['corner_cases']:
        with zipfile.ZipFile(ROOT/f'data/raw/wdc-products-{hardness}pair.zip') as archive:
            prefix=f'wdcproducts{hardness}cc{100-hardness}rnd'
            tests={unseen:read_member(archive,prefix+f'{unseen:03d}un_gs.json.gz') for unseen in config['wdc']['unseen_percent']}
            for size in config['wdc']['development_sizes']:
                train=read_member(archive,prefix+'000un_train_'+size+'.json.gz')
                validation=read_member(archive,prefix+'000un_valid_'+size+'.json.gz')
                feature=FastAttributes().fit([train['records'][u] for pair in train['pairs'] for u in pair])
                t=attributes(train,feature,embedding);v=attributes(validation,feature,embedding)
                groups=sorted(set(validation['groups'].values()));random.Random(config['split_seed']+101).shuffle(groups)
                group_roles={g:ROLES[i%3] for i,g in enumerate(groups)}
                assignment={u:group_roles[g] for u,g in validation['groups'].items()};masks=role_masks(v['pairs'],assignment)
                if any(len(set(v['y'][m]))!=2 for m in masks.values()):raise ValueError('WDC development role lacks both classes')
                selection,calibration,policy=(masks[r] for r in ROLES);prior=float(v['y'].mean())
                # Features for all test conditions are label-blind. Cache the
                # common conditions across model seeds; never train on them.
                test_matrices={u:attributes(d,feature,embedding) for u,d in tests.items()}
                stress_matrices={}
                for unseen,test in tests.items():
                    scenarios=[('clean',{})]+[(f'optional_missingness_{int(p*100)}',{'missingness':p}) for p in [.1,.3,.5]]+[(f'attribute_noise_{int(p*100)}',{'noise':p}) for p in [.3,.5]]
                    stress_matrices[unseen]=[]
                    for name,changes in scenarios:
                        if name=='clean':m=test_matrices[unseen]
                        else:
                            altered=copy.deepcopy(test);rng=random.Random(811)
                            for r in altered['records'].values():
                                if rng.random()<changes.get('noise',0):r['name']=' '.join(w[:max(1,len(w)//2)] for w in r['name'].split())
                                for field in ['brand','description','price']:
                                    if r.get(field) and rng.random()<changes.get('missingness',0):r[field]=''
                            m=attributes(altered,feature,embedding)
                        stress_matrices[unseen].append((name,m))
                    overlaps={name:len(set(data['records'])&set(test['records'])) for name,data in [('train',train),('validation',validation)]}
                    overlap_uids=set(train['records'])|set(validation['records'])
                    disjoint=np.array([a not in overlap_uids and b not in overlap_uids for a,b in test['pairs']],dtype=bool)
                    test_matrices[unseen]['offer_disjoint']=disjoint
                    split_audits.append({'corner_cases':hardness,'development_size':size,'unseen_percent':unseen,'train_pairs':len(train['pairs']),
                        'validation_pairs':len(validation['pairs']),'test_pairs':len(test['pairs']),'train_test_offer_overlap':overlaps['train'],
                        'validation_test_offer_overlap':overlaps['validation'],'offer_disjoint_test_pairs':int(disjoint.sum()),
                        'train_validation_offer_overlap':len(set(train['records'])&set(validation['records'])),
                        'train_decompressed_sha256':train['decompressed_sha256'],'validation_decompressed_sha256':validation['decompressed_sha256'],'test_decompressed_sha256':test['decompressed_sha256'],
                        'all_official_pairs_retained':True})
                for seed in config['seeds']:
                    folder=out/f'{hardness}cc_{size}/seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
                    models={'calibration':{}};choices={};started=time.perf_counter()
                    for key in ['attribute','embedding','augmented']:
                        model,choice=select_model(x_for(t,key),t['y'],x_for(v,key)[selection],v['y'][selection],{**config['models'],'target_prior':prior},seed)
                        models[key]=model;choices[key]=choice
                        models['calibration'][key]=SigmoidCalibration().fit(model.predict_proba(x_for(v,key)[calibration])[:,1],v['y'][calibration],prior_weights(v['y'][calibration],prior))
                    vp,_=predictions(models,v)
                    accepts={name:choose_threshold(p[policy],v['y'][policy],prior) for name,p in vp.items()};accepts['B0Exact']=1.
                    models['accepts']=accepts
                    joblib.dump({'feature':feature,'models':models},folder/'frozen_models.joblib')
                    write_json(folder/'freeze.json',{'frozen_at_utc':utc_now(),'model_sha256':digest(folder/'frozen_models.joblib'),'accepts':accepts,'choices':choices,
                        'role_counts':{r:{'groups':sum(x==r for x in group_roles.values()),'pairs':int(m.sum()),'positives':int(v['y'][m].sum())} for r,m in masks.items()},
                        'official_train_test_pairs_unchanged':True,'development_prior':prior,'class_prior_correction':'full validation prevalence; no test labels',
                        'cross_role_pairs_omitted':int(len(v['y'])-sum(m.sum() for m in masks.values())),'group_disjoint_development_roles':True,
                        'calibration_and_policy_protocol':'same trained model/policy reused across all three official unseen configurations'})
                    timing.append({'corner_cases':hardness,'size':size,'seed':seed,'fit_seconds':time.perf_counter()-started})
                    for unseen,test in test_matrices.items():
                        p,raw=predictions(models,test);y=test['y'];context={'corner_cases':hardness,'development_size':size,'unseen_percent':unseen,'seed':seed}
                        conditions=[('official_all',np.ones(len(y),dtype=bool)),('hard_challenge',test['hard']),('offer_disjoint_subset',test['offer_disjoint'])]
                        for name,values in p.items():
                            for stratum,mask in conditions:
                                if mask.any():all_metrics.append({**context,'model':name,'stratum':stratum,**binary(y[mask],values[mask]>=accepts[name],values[mask])})
                        write_json(folder/f'predictions_{unseen:03d}un.json',[{'pair_id':test['pair_ids'][i],'label':int(y[i]),'offer_disjoint':bool(test['offer_disjoint'][i]),**{k:float(v[i]) for k,v in p.items()}} for i in range(len(y))])
                        reliability_bins=[]
                        for name,key in [('B2Attributes','attribute'),('B3Embedding','embedding'),('B2EmbeddingAugmented','augmented')]:
                            for mode,values in [('raw',raw[key]),('calibrated',p[name])]:
                                metric,bins=reliability(y,values);all_cal.append({**context,'model':name,'mode':mode,**metric})
                                reliability_bins.extend({'model':name,'mode':mode,**b} for b in bins)
                        write_rows(folder/f'reliability_bins_{unseen:03d}un.csv',reliability_bins)
                        truth={u:{'canonical':g} for u,g in test['groups'].items()}
                        intervals=[]
                        for name in ['B3Embedding','B2EmbeddingAugmented']:
                            low,high=paired_bootstrap(y,p['B2Attributes']>=accepts['B2Attributes'],p[name]>=accepts[name],test['pairs'],truth,seed,config['bootstrap_replicates'])
                            intervals.append({'contrast':name+'_minus_B2Attributes','ci_low':low,'ci_high':high,'bootstrap_unit':'product groups with endpoint multiplicities'})
                        write_rows(folder/f'paired_ci_{unseen:03d}un.csv',intervals)
                        risk=[]
                        for name,values in p.items():
                            for accept in sorted({accepts[name],.5,.6,.7,.8,.9,.95,.99,1.000001}):risk.append({'model':name,'accept':accept,**coverage_risk(y,values,accept,.1)})
                        write_rows(folder/f'coverage_risk_{unseen:03d}un.csv',risk)
                        for scenario_name,m in stress_matrices[unseen]:
                            changed,_=predictions(models,m)
                            for name,values in changed.items():all_robust.append({**context,'scenario':scenario_name,'model':name,'same_official_pairs':True,**binary(y,values>=accepts[name],values)})
                    print(f'WDC {hardness}%/{size}/seed {seed}: all three official unseen configurations + attribute robustness complete',flush=True)
                    write_rows(out/'matching.csv',all_metrics);write_rows(out/'robustness.csv',all_robust);write_rows(out/'calibration.csv',all_cal)
                write_rows(out/'split_integrity.csv',split_audits);write_rows(out/'training_costs.csv',timing)
    write_json(out/'experiment_scope.json',{'official_configurations':27,'training_seeds':3,'attribute_methods':METHODS,
        'test_pairs_per_configuration':4500,'graph_context_E3_ERP_queries':'not applicable: no ERP transaction relationships; never constructed from benchmark gold',
        'overlap_policy':'all official pairs retained; any supplied offer overlap disclosed and separately evaluated on offer-disjoint subset; overlap cells not clean held-out evidence',
        'E1':'offer/schema/count/pair-uniqueness/provenance and split-integrity audit','E2':'all official pairwise configurations','E4':'attribute noise/missingness, hardness/unseen, calibration and coverage-risk',
        'E5':'training cost by official small/medium/large sizes; additional candidate/index/scoring microbenchmarks in scale outputs',
        'not_claimed':'reproduction of Ditto/HierGAT/R-SupCon training or a multiclass task; those are literature comparators, not planned mandatory experiments'})
    return all_metrics
