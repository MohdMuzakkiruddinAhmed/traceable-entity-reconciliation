"""Domain-adapted E2/E3/E4 with common pairs and frozen development policies."""
from collections import Counter,defaultdict
import hashlib
import json
import time
import joblib
import numpy as np
from ..io import write_json,write_rows,digest,utc_now
from ..blocking import normalize,pair_id,candidates
from ..oracle import labels,true_pair_count
from ..metrics import binary,set_metrics,paired_bootstrap
from ..research_plus.blocking import bounded_candidates
from ..research_plus.learning import select_model,SigmoidCalibration,prior_weights,choose_threshold,reliability,coverage_risk
from ..research_plus.protocol import partition_validation,role_masks,ROLES
from .attributes import FastAttributes
from .features import graph_profiles,flat_profiles,context,NAMES
from .protocol import build_views,perturb
from .queries import PreparedQueries,SQLQueries,scoped_graph,catalog

METHODS=['B0Exact','B1Fuzzy','B2Attributes','B3Embedding','B2EmbeddingAugmented','B4Flat','B5Graph','NoGate','NoSupport','UnitWeights','NoContext','NoTransactions','OneHop']
LEARNERS=['attribute','embedding','augmented','weighted','no_support','no_transactions','one_hop']


def blocker(config):
    c=config['blocking']
    return {'token_prefix':c['token_prefix'],'include_category':True,'postings_per_k':c['postings_multiplier'],
            'max_keys':c['max_keys_per_record'],'max_candidates':c['max_candidates']}


def matrix(records,graph,attribute,embedding,config,pairs=None,k=None,check_flat=True,no_anchors=False):
    wwi=any(r['entity_type']=='StockItem' for r in records)
    if pairs is None:
        if wwi:pairs,_=candidates(records,{'token_prefix':4,'include_category':True,'max_candidates':2000000})
        else:pairs,_=bounded_candidates(records,blocker(config),k or max(config['blocking']['k_grid']))
    by_uid={r['source_uid']:r for r in records}
    ax=attribute.transform(pairs,by_uid)
    em=embedding.similarities(pairs,by_uid)
    profiles=graph_profiles(graph,records)
    cx,evidence=context(pairs,profiles,config['models']['min_anchors'])
    if check_flat:
        flat,_=context(pairs,flat_profiles(graph,records),config['models']['min_anchors'])
        if not np.allclose(cx,flat,atol=1e-12,rtol=0):raise ValueError('Graph/flat context extraction differs')
    else:flat=cx.copy()
    unit,_=context(pairs,graph_profiles(graph,records,weighted=False),config['models']['min_anchors'])
    nosupport,_=context(pairs,profiles,config['models']['min_anchors'],support_features=False)
    notx,_=context(pairs,profiles,config['models']['min_anchors'],transactions=False)
    onehop,one_evidence=context(pairs,graph_profiles(graph,records,one_hop=True),config['models']['min_anchors'],one_hop=True)
    near=Counter(a for i,(a,b) in enumerate(pairs) if ax[i,1]>=.65)
    ambiguous=np.array([ax[i,0]<.95 or near[a]>1 for i,(a,b) in enumerate(pairs)],dtype=bool)
    result={'records':records,'by_uid':by_uid,'pairs':pairs,'ax':ax,'em':em,'weighted':cx,'flat':flat,'unit':unit,
            'no_support':nosupport,'no_transactions':notx,'one_hop':onehop,'evidence':evidence,'one_evidence':one_evidence,'profiles':profiles,'ambiguous':ambiguous}
    result['context_names']=NAMES
    if wwi:
        from .features_wwi import combined
        result.update(combined(pairs,records,graph,profiles,flat_profiles(graph,records),graph_profiles(graph,records,weighted=False),graph_profiles(graph,records,one_hop=True),config,no_anchors))
        extra=[]
        for a,b in pairs:
            x,y=by_uid[a].get('quantity_per_outer'),by_uid[b].get('quantity_per_outer')
            present=x is not None and y is not None
            extra.append([float(present and x==y),min(x,y)/max(1,x,y) if present else 0.,float(not present)])
        result['ax']=np.column_stack([ax,np.array(extra)])
    return result


def learner_x(m,key):
    if key=='attribute':return m['ax']
    if key=='embedding':return m['em']
    if key=='augmented':return np.column_stack([m['ax'],m['em']])
    return np.column_stack([m['ax'],m[key]])


def score(models,m):
    raw={};cal={}
    for key in LEARNERS:
        raw[key]=models[key].predict_proba(learner_x(m,key))[:,1]
        cal[key]=models['calibration'][key].transform(raw[key])
    attr=cal['attribute'];available=m['weighted'][:,-1]>0
    scores={'B0Exact':((m['ax'][:,0]==1)&(m['ax'][:,4]==1)&(m['ax'][:,9]==0)).astype(float),
            'B1Fuzzy':m['ax'][:,1], 'B2Attributes':attr,'B3Embedding':cal['embedding'],'B2EmbeddingAugmented':cal['augmented'],
            'B5Graph':np.where(available,cal['weighted'],attr),'NoGate':cal['weighted'],
            'NoSupport':np.where(available,cal['no_support'],attr),'NoContext':attr.copy(),
            'NoTransactions':np.where(available,cal['no_transactions'],attr),
            'OneHop':np.where(m['one_hop'][:,-1]>0,cal['one_hop'],attr)}
    p=models['weighted'].predict_proba(np.column_stack([m['ax'],m['flat']]))[:,1]
    scores['B4Flat']=np.where(m['flat'][:,-1]>0,models['calibration']['weighted'].transform(p),attr)
    p=models['weighted'].predict_proba(np.column_stack([m['ax'],m['unit']]))[:,1]
    scores['UnitWeights']=np.where(m['unit'][:,-1]>0,models['calibration']['weighted'].transform(p),attr)
    if not np.allclose(scores['B4Flat'],scores['B5Graph'],atol=1e-12,rtol=0):raise ValueError('Graph/flat scoring parity')
    return scores,{'B2Attributes':raw['attribute'],'B5Graph':np.where(available,raw['weighted'],raw['attribute'])}


def fit(records,graph,gold,seed,config,embedding,out):
    train=[r for r in records if r['split']=='train'];validation=[r for r in records if r['split']=='validation']
    attrs=FastAttributes().fit(train)
    t=matrix(train,graph,attrs,embedding,config);v=matrix(validation,graph,attrs,embedding,config)
    assignment,groups=partition_validation(validation,gold,config['split_seed']+101)
    masks=role_masks(v['pairs'],assignment);ty,vy=labels(t['pairs'],gold),labels(v['pairs'],gold)
    if any(len(set(vy[mask]))!=2 for mask in masks.values()):raise ValueError('Development role lacks both classes')
    selection,calibration,policy=(masks[role] for role in ROLES)
    prior=float(vy.mean());models={'calibration':{}};choices={}
    for key in LEARNERS:
        tx=learner_x(t,key);vx=learner_x(v,key)
        model,choice=select_model(tx,ty,vx[selection],vy[selection],{**config['models'],'target_prior':prior},seed)
        models[key]=model;choices[key]=choice
        models['calibration'][key]=SigmoidCalibration().fit(model.predict_proba(vx[calibration])[:,1],vy[calibration],prior_weights(vy[calibration],prior))
    p,_=score(models,v)
    accepts={name:choose_threshold(s[policy],vy[policy],prior) for name,s in p.items()}
    accepts['B0Exact']=1.;accepts['NoContext']=accepts['B2Attributes'];accepts['B4Flat']=accepts['B5Graph']
    accepts['UnitWeights']=accepts['B5Graph'] # same trained model/policy; isolated provenance-weight contrast
    models['accepts']=accepts
    trials=[]
    for k in config['blocking']['k_grid']:
        pairs,stats=bounded_candidates(validation,blocker(config),k)
        trials.append({'k':k,'candidate_count':len(pairs),'blocking_recall':int(labels(pairs,gold).sum())/true_pair_count(validation,gold)})
    eligible=[r for r in trials if r['blocking_recall']>=config['blocking']['target_development_recall']]
    chosen=min(eligible,key=lambda r:r['k']) if eligible else max(trials,key=lambda r:(r['blocking_recall'],-r['k']))
    models['retrieval_k']=chosen['k']
    out.mkdir(parents=True,exist_ok=True)
    joblib.dump({'attributes':attrs,'models':models},out/'frozen_models.joblib')
    write_json(out/'freeze.json',{'frozen_at_utc':utc_now(),'model_sha256':digest(out/'frozen_models.joblib'),'accepts':accepts,'choices':choices,
        'role_counts':{role:{'groups':len(groups[role]),'pairs':int(mask.sum()),'positives':int(vy[mask].sum())} for role,mask in masks.items()},
        'cross_role_pairs_omitted':int(len(vy)-sum(mask.sum() for mask in masks.values())), 'development_prior':prior,
        'group_disjoint_selection_calibration_policy':True,'class_prior_weights':'restore full-development class prevalence using validation only; not a test-prior correction',
        'common_candidate_k':max(config['blocking']['k_grid']),'retrieval_trials':trials,'chosen_retrieval_k':chosen['k'],
        'development_recall_target_met':bool(eligible),'classification':config['classification'],'context_feature_names':v['context_names'],
        'context_semantics':'typed local transaction/category summaries; category support, not gold or copied master/entity identifiers'})
    return attrs,models


def identities(records,graph,m,scores,accept,reject,out,model_hash):
    mapping={r['source_uid']:r['source_uid'] for r in records};used=set();decisions=[];by_uid=m['by_uid']
    for i in sorted(range(len(scores)),key=lambda i:(-float(scores[i]),m['pairs'][i])):
        a,b=m['pairs'][i];p=float(scores[i]);status='rejected' if p<=min(reject,accept) else 'abstained'
        if p>=accept:
            if by_uid[a]['entity_type']!=by_uid[b]['entity_type']:status='type_conflict'
            elif a in used or b in used:status='one_to_one_conflict'
            else:
                status='accepted';used.update([a,b]);cid='canonical:'+hashlib.sha256((a+'|'+b).encode()).hexdigest()[:24];mapping[a]=mapping[b]=cid
        decisions.append({'candidate_id':pair_id(a,b),'left_source_uid':a,'right_source_uid':b,'score':p,'accept':accept,'reject':min(reject,accept),
            'status':status,'method_version':model_hash,'context_used':bool(m['weighted'][i,-1]),'context_evidence':m['evidence'][i],
            'attribute_values':m['ax'][i].tolist(),'context_values':m['weighted'][i].tolist() if m['weighted'][i,-1] else [],
            'decision_trigger':'same_type_one_to_one_frozen_development_policy'})
    write_json(out/'decisions.json',decisions)
    return mapping,decisions


def query_evaluation(dataset,records,graph,gold,mapping,out,seed):
    test=[r for r in records if r['split']=='test'];scope=scoped_graph(graph,test)
    oracle={r['source_uid']:'oracle:'+gold[r['source_uid']]['canonical'] for r in test}
    expected=SQLQueries(scope,oracle);oracle_answers=expected.execute();expected.close()
    if any(not v for v in oracle_answers.values()):raise ValueError('Empty expanded query oracle')
    query_rows=[]
    for name,identity in [('Unreconciled',{r['source_uid']:r['source_uid'] for r in test}),('Predicted',mapping),('Oracle',oracle)]:
        sql=SQLQueries(scope,identity);actual=PreparedQueries(scope,identity).execute()
        if actual!=sql.execute():raise ValueError('Expanded query SQL/graph mismatch')
        for q in actual:query_rows.append({'dataset':dataset,'seed':seed,'identity':name,'query':q,'SQL_KG_exact':True,**set_metrics(oracle_answers[q],actual[q])})
        sql.close()
    write_rows(out/'query_accuracy.csv',query_rows);write_json(out/'query_catalog.json',catalog(dataset))
    write_json(out/'oracle_query_denominators.json',{q:len(a) for q,a in oracle_answers.items()})
    return scope


def evaluate_seed(dataset,seed,config,embedding,out):
    out=out/f'seed_{seed}';out.mkdir(parents=True,exist_ok=True)
    records,graph,gold,view_manifest=build_views(dataset,seed,config)
    write_json(out/'view_manifest.json',view_manifest)
    started=time.perf_counter();attrs,models=fit(records,graph,gold,seed,config,embedding,out)
    training_seconds=time.perf_counter()-started
    # Test labels are used only after frozen model/policy artifacts exist.
    test=[r for r in records if r['split']=='test'];m=matrix(test,graph,attrs,embedding,config)
    probabilities,raw=score(models,m);y=labels(m['pairs'],gold);total=true_pair_count(test,gold)
    rows=[]
    for name,p in probabilities.items():
        for stratum,mask in [('all',np.ones(len(y),dtype=bool)),('ambiguous',m['ambiguous'])]:
            rows.append({'dataset':dataset,'seed':seed,'model':name,'stratum':stratum,'independent_groups':total,
                **binary(y[mask],p[mask]>=models['accepts'][name],p[mask],total if stratum=='all' else None)})
    write_rows(out/'matching.csv',rows)
    write_json(out/'predictions.json',[{'candidate_id':pair_id(a,b),'left_source_uid':a,'right_source_uid':b,'label':int(y[i]),'ambiguous':bool(m['ambiguous'][i]),
        'context_supported':bool(m['weighted'][i,-1]),**{name:float(p[i]) for name,p in probabilities.items()}} for i,(a,b) in enumerate(m['pairs'])])
    ci_rows=[]
    mask=m['ambiguous'];pairs=[p for p,keep in zip(m['pairs'],mask) if keep]
    for name in ['B5Graph','B2EmbeddingAugmented','OneHop','NoTransactions']:
        low,high=paired_bootstrap(y[mask],probabilities['B2Attributes'][mask]>=models['accepts']['B2Attributes'],probabilities[name][mask]>=models['accepts'][name],pairs,gold,seed,config['bootstrap_replicates'])
        ci_rows.append({'contrast':name+'_minus_B2Attributes','stratum':'ambiguous','ci_low':low,'ci_high':high})
    write_rows(out/'paired_ci.csv',ci_rows)
    calibration=[];bins=[];risk=[]
    for name in ['B2Attributes','B5Graph']:
        for mode,p in [('raw',raw[name]),('calibrated',probabilities[name])]:
            summary,table=reliability(y,p);calibration.append({'dataset':dataset,'seed':seed,'model':name,'mode':mode,**summary})
            bins.extend({'model':name,'mode':mode,**r} for r in table)
        for accept in sorted({models['accepts'][name],.5,.6,.7,.8,.9,.95,.99,1.000001}):
            risk.append({'model':name,'accept':accept,**coverage_risk(y,probabilities[name],accept,config['models']['reject_probability'])})
    write_rows(out/'calibration.csv',calibration);write_rows(out/'reliability_bins.csv',bins);write_rows(out/'coverage_risk.csv',risk)
    mapping,decisions=identities(records,graph,m,probabilities['B5Graph'],models['accepts']['B5Graph'],config['models']['reject_probability'],out,digest(out/'frozen_models.joblib'))
    accepted=[d for d in decisions if d['status']=='accepted']
    write_json(out/'identity_audit.json',{'accepted_links':len(accepted),'false_merges':sum(gold[d['left_source_uid']]['canonical']!=gold[d['right_source_uid']]['canonical'] for d in accepted),
        'raw_source_graph_unchanged':True,'one_to_one':len({d['left_source_uid'] for d in accepted})==len(accepted) and len({d['right_source_uid'] for d in accepted})==len(accepted)})
    scope=query_evaluation(dataset,records,graph,gold,mapping,out,seed)
    robust=[];logs=[]
    scenarios=[('clean',{}),('no_anchors',{'no_anchors':True})]
    for argument,levels in config['robustness'].items():scenarios.extend((argument+'_'+str(int(level*100)),{argument:level}) for level in levels)
    for scenario_name,parameters in scenarios:
        changed,view,log=perturb(test,graph,seed+701,**parameters)
        changed_matrix=matrix(changed,view,attrs,embedding,config,pairs=m['pairs'],no_anchors=parameters.get('no_anchors',False))
        p,_=score(models,changed_matrix)
        for name,values in p.items():
            metric=binary(y,values>=models['accepts'][name],values,total)
            robust.append({'dataset':dataset,'seed':seed,'scenario':scenario_name,'model':name,'same_frozen_pairs':True,
                **metric,**coverage_risk(y,values,models['accepts'][name],config['models']['reject_probability'])})
        logs.append({'scenario':scenario_name,**log})
    write_rows(out/'robustness.csv',robust);write_rows(out/'perturbation_counts.csv',logs)
    subgroup=[]
    degree={u:p['degree'] for u,p in m['profiles'].items()}
    masks={'supported':m['weighted'][:,-1]>0,'unsupported':m['weighted'][:,-1]==0,
           'sparse_0_to_5':np.array([degree[a]<=5 for a,b in m['pairs']]),'dense_over_5':np.array([degree[a]>5 for a,b in m['pairs']]),
           'hard_challenge':(y==1)|(m['ax'][:,1]>=.8)}
    for stratum,mask in masks.items():
        if mask.any():
            for name,p in probabilities.items():subgroup.append({'dataset':dataset,'seed':seed,'stratum':stratum,'model':name,**binary(y[mask],p[mask]>=models['accepts'][name],p[mask])})
    write_rows(out/'subgroups.csv',subgroup)
    pairs,stats=bounded_candidates(test,blocker(config),models['retrieval_k'])
    positions={p:i for i,p in enumerate(m['pairs'])};index=np.array([positions[p] for p in pairs],dtype=int)
    write_rows(out/'bounded_pipeline.csv',[{'model':name,**binary(labels(pairs,gold),p[index]>=models['accepts'][name],p[index],total)} for name,p in probabilities.items()])
    write_json(out/'costs.json',{'training_seconds':training_seconds,'matching_test_records':len(test),'candidate_pairs':len(y),'true_pairs':total,
        'common_blocking_recall':int(y.sum())/total,'retained_true_pairs_bounded':int(labels(pairs,gold).sum()),'bounded':stats})
    print(f'{dataset} seed {seed}: E2/E3/E4 complete; {len(test)} test records / {len(y)} candidates / blocking recall {int(y.sum())/total:.3f}',flush=True)
    return {'dataset':dataset,'records':records,'graph':graph,'gold':gold,'test_records':test,'scope':scope,'mapping':mapping,
            'attribute':attrs,'models':models,'matching':rows,'robustness':robust}
