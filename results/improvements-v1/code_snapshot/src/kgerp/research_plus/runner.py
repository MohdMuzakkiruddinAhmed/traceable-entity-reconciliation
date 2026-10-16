"""Evaluator for a versioned exploratory extension; never rewrites primary runs."""
from collections import defaultdict
import copy
import gc
import gzip
import json
from pathlib import Path
import pickle
import random
import time
import zipfile
import joblib
import numpy as np
import psutil
import yaml
from ..io import ROOT, read_json, write_json, write_rows, digest, utc_now
from ..blocking import candidates, pair_id
from ..features_context import infer_anchors, graph_neighbors, context_features
from ..matchers import score as original_score, acceptance, threshold
from ..metrics import binary, paired_bootstrap, set_metrics
from ..oracle import labels, true_pair_count, oracle_mapping
from ..identity import resolve
from ..queries import sql_database, sql_answers, graph_answers, SQL
from ..evaluations import environment, replica_workload
from .features import EnhancedAttributes, ATTRIBUTE_NAMES, CONTEXT_NAMES, weighted_graph_neighbors, weighted_flat_neighbors, weighted_context
from .learning import select_model, SigmoidCalibration, reliability, coverage_risk, prior_weights, choose_threshold
from .blocking import bounded_candidates
from .protocol import partition_validation, role_masks, scenario, ROLES
from .queries import PreparedGraphQueries

METHODS = ['OriginalB2','OriginalB5','AttrPlus','GraphWeighted','GraphUngated','GraphUnweighted','GraphNoSupport','GraphNoContext']

def load_seed(seed):
    records = read_json(ROOT / f'data/generated/seed_{seed}/records.json')
    gold = read_json(ROOT / f'data/gold/seed_{seed}/truth.json')
    with (ROOT / f'data/generated/seed_{seed}/graph.pkl').open('rb') as handle:
        graph = pickle.load(handle)
    return records, graph, gold

def matrices(records, graph, features, blocker, model_config, fixed_pairs=None):
    by_uid = {r['source_uid']:r for r in records}
    pairs, stats = candidates(records, blocker) if fixed_pairs is None else (fixed_pairs, {'candidate_count':len(fixed_pairs)})
    anchors = infer_anchors(records)
    gn, fn = weighted_graph_neighbors(graph, records), weighted_flat_neighbors(graph, records)
    if gn != fn:
        raise ValueError('Weighted graph/SQL neighborhoods disagree')
    kwargs = {'min_anchors':model_config['min_anchors'], 'min_weighted_support':model_config['min_weighted_support']}
    cx, evidence = weighted_context(pairs, by_uid, gn, anchors, **kwargs)
    flat, _ = weighted_context(pairs, by_uid, fn, anchors, **kwargs)
    if not np.array_equal(cx,flat):
        raise ValueError('Weighted graph/SQL feature discrepancy')
    cu, _ = weighted_context(pairs, by_uid, gn, anchors, weighted=False, **kwargs)
    cs, _ = weighted_context(pairs, by_uid, gn, anchors, support_features=False, **kwargs)
    return {'records':records, 'by_uid':by_uid, 'pairs':pairs, 'stats':stats, 'ax':features.transform(pairs,by_uid),
            'weighted':cx, 'unweighted':cu, 'no_support':cs, 'evidence':evidence, 'anchors':anchors}

def method_scores(models, matrix):
    ax = matrix['ax']
    attr_raw = models['attr'].predict_proba(ax)[:,1]
    attr = models['calibrators']['attr'].transform(attr_raw)
    output = {'AttrPlus':attr, 'GraphNoContext':attr.copy()}
    raw = {'AttrPlus':attr_raw}
    for name, key, estimator in [('GraphWeighted','weighted','weighted'), ('GraphUnweighted','unweighted','unweighted'), ('GraphNoSupport','no_support','no_support')]:
        p = models[estimator].predict_proba(np.column_stack([ax, matrix[key]]))[:,1]
        calibrated = models['calibrators'][estimator].transform(p)
        supported = matrix['weighted'][:,-1]>0 if key=='no_support' else matrix[key][:,-1]>0
        output[name] = np.where(supported, calibrated, attr)
        raw[name] = np.where(supported, p, attr_raw)
        if name == 'GraphWeighted':
            output['GraphUngated'] = calibrated
    return output, raw

def fit_protocol(records, graph, gold, seed, config, base_config, out):
    train_records = [r for r in records if r['split']=='train']
    validation_records = [r for r in records if r['split']=='validation']
    feature = EnhancedAttributes().fit(train_records)
    t = matrices(train_records,graph,feature,base_config['blocker'],config['models'])
    v = matrices(validation_records,graph,feature,base_config['blocker'],config['models'])
    assignment, groups = partition_validation(validation_records,gold,config['development_partition_seed'])
    masks = role_masks(v['pairs'],assignment)
    ty, vy = labels(t['pairs'],gold),labels(v['pairs'],gold)
    development_prior=float(vy.mean())
    selection_config={**config['models'],'target_prior':development_prior}
    for role, mask in masks.items():
        if len(set(vy[mask])) != 2:
            raise ValueError(f'{role} lacks both classes; do not silently reuse another partition')
    role_summary = {role:{'canonical_groups':len(groups[role]),'pairs':int(mask.sum()),'positives':int(vy[mask].sum())} for role,mask in masks.items()}
    selection, calibration, policy = (masks[r] for r in ROLES)
    models = {'calibrators':{}}; choices = {}
    for name, context_key in [('attr',None),('weighted','weighted'),('unweighted','unweighted'),('no_support','no_support')]:
        tx = t['ax'] if context_key is None else np.column_stack([t['ax'],t[context_key]])
        vx = v['ax'] if context_key is None else np.column_stack([v['ax'],v[context_key]])
        estimator, summary = select_model(tx,ty,vx[selection],vy[selection],selection_config,seed)
        models[name] = estimator;choices[name]=summary
        models['calibrators'][name] = SigmoidCalibration().fit(estimator.predict_proba(vx[calibration])[:,1],vy[calibration],
                                                              sample_weight=prior_weights(vy[calibration],development_prior))
    vp,_ = method_scores(models,v)
    models['thresholds'] = {name:choose_threshold(p[policy],vy[policy],development_prior) for name,p in vp.items()}
    models['thresholds']['GraphNoContext'] = models['thresholds']['AttrPlus']
    selection_trials = []
    for k in config['blocking']['k_grid']:
        pairs, stats = bounded_candidates(validation_records,config['blocking'],k)
        y = labels(pairs,gold)
        recall = int(y.sum()) / true_pair_count(validation_records,gold)
        selection_trials.append({'k':k, 'pairs':len(pairs), 'blocking_recall':recall})
    eligible = [r for r in selection_trials if r['blocking_recall'] >= config['blocking']['target_development_recall']]
    best = min(eligible,key=lambda r:r['k']) if eligible else max(selection_trials,key=lambda r:(r['blocking_recall'],-r['k']))
    models['block_k'] = best['k']
    path = out / f'wwi/seed_{seed}';path.mkdir(parents=True,exist_ok=True)
    joblib.dump({'features':feature,'models':models},path/'frozen_models.joblib')
    write_json(path/'freeze.json',{'frozen_at_utc':utc_now(), 'model_sha256':digest(path/'frozen_models.joblib'),
          'model_selection':choices,'thresholds':models['thresholds'], 'validation_roles':role_summary,
          'development_roles_group_disjoint':True,'pairs_between_roles_omitted':int(sum(~(selection|calibration|policy))),
          'blocker_development_trials':selection_trials,'chosen_k':best['k'],
          'blocker_target_met':bool(eligible),'attribute_feature_names':ATTRIBUTE_NAMES,'context_feature_names':CONTEXT_NAMES,
          'calibration':'logistic sigmoid on separate calibration groups; thresholds on policy groups only',
          'development_prior':development_prior,'class_prior_correction':'role partitions omit cross-role negatives; class weights restore full-validation prevalence using development labels only',
          'classification':'exploratory extension after prior held-out review'})
    return feature, models

def original_predictions(feature, models, pairs, records, graph, no_anchors=False):
    by_uid = {r['source_uid']:r for r in records}
    anchors = [] if no_anchors else infer_anchors(records)
    cx,_ = context_features(pairs,by_uid,graph_neighbors(graph,records),anchors)
    result = original_score(models,feature.transform(pairs,by_uid),cx)
    return {'OriginalB2':result['B2'],'OriginalB5':result['B5']}

def identity_decisions(graph, gold, all_records, matrix, probabilities, threshold_, reject, out, seed, model_hash):
    mapping, decisions = resolve(matrix['pairs'],probabilities,{r['source_uid']:r for r in all_records},matrix['evidence'],threshold_,reject,
                                  f'improved_seed_{seed}',model_hash)
    positions={pair_id(a,b):i for i,(a,b) in enumerate(matrix['pairs'])}
    for decision in decisions:
        i = positions[decision['candidate_id']]
        used = bool(matrix['weighted'][i,-1])
        decision['actual_context_used'] = used
        decision['supporting_feature_names'] = ATTRIBUTE_NAMES + (CONTEXT_NAMES if used else [])
        values = list(matrix['ax'][i]) + (list(matrix['weighted'][i]) if used else [])
        decision['feature_values'] = {k:float(v) for k,v in zip(decision['supporting_feature_names'],values)}
        decision['calibration_policy'] = 'disjoint_validation_calibration_groups_v3'
    write_json(out / f'wwi/seed_{seed}/decisions.json', decisions)
    # A physical, non-destructive identity view is exported only AFTER scoring.
    # No inferred identity node/edge is ever fed back into model features.
    identity_graph = graph.copy()
    canonical_groups = defaultdict(list)
    for source_uid, canonical in mapping.items():
        if canonical != source_uid:
            canonical_groups[canonical].append(source_uid)
    for canonical, sources in canonical_groups.items():
        kind = identity_graph.nodes[sources[0]]['entity_type']
        identity_graph.add_node(canonical,entity_type='CanonicalEntity',resolved_type=kind,canonical_id=canonical,run_id=f'improved_seed_{seed}')
        for source_uid in sources:
            identity_graph.add_edge(source_uid,canonical,key='REFERS_TO',relation='REFERS_TO',method_version=model_hash)
    accepted=[d for d in decisions if d['status']=='accepted']
    for d in accepted:
        identity_graph.add_edge(d['left_source_uid'],d['right_source_uid'],key='ACCEPTED_SAME_AS',relation='ACCEPTED_SAME_AS',
                                score=d['score'],method_version=model_hash,candidate_id=d['candidate_id'])
    with (out / f'wwi/seed_{seed}/canonical_graph.pkl').open('wb') as handle:
        pickle.dump(identity_graph,handle)
    write_json(out / f'wwi/seed_{seed}/identity_audit.json',{'accepted_links':len(accepted),
             'false_merges':sum(gold[d['left_source_uid']]['canonical']!=gold[d['right_source_uid']]['canonical'] for d in accepted),
             'canonical_nodes':len(canonical_groups),'refers_to_edges':2*len(canonical_groups),
             'raw_nodes_preserved':all(identity_graph.nodes[u]==graph.nodes[u] for u in graph),
             'raw_edges_preserved':all(identity_graph.has_edge(a,b,k) for a,b,k in graph.edges(keys=True)),
             'provenance_complete':all(d['feature_values'] and d['method_version'] and d['decision_trigger'] for d in accepted)})
    return mapping

def run_wwi(seed, config, base_config, source, out):
    records, graph, gold = load_seed(seed)
    feature, models = fit_protocol(records,graph,gold,seed,config,base_config,out)
    # Frozen files now exist. Only now evaluate labels of the test partition.
    test_records = [r for r in records if r['split']=='test']
    original = joblib.load(source / f'e2/seed_{seed}/frozen_models.joblib')
    matrix = matrices(test_records,graph,feature,base_config['blocker'],config['models'])
    probabilities,raw = method_scores(models,matrix)
    probabilities.update(original_predictions(original['features'],original['models'],matrix['pairs'],test_records,graph))
    thresholds = {**models['thresholds'],'OriginalB2':acceptance(original['models'],'B2'),'OriginalB5':acceptance(original['models'],'B5')}
    y=labels(matrix['pairs'],gold);total=true_pair_count(test_records,gold)
    old_pred = {p['candidate_id']:p for p in read_json(source / f'e2/seed_{seed}/predictions.json')}
    for i,(a,b) in enumerate(matrix['pairs']):
        p = old_pred[pair_id(a,b)]
        if abs(p['B2']-probabilities['OriginalB2'][i])>1e-12 or abs(p['B5']-probabilities['OriginalB5'][i])>1e-12:
            raise ValueError('Historical baseline replay differs')
    ambiguous=np.array([old_pred[pair_id(a,b)]['ambiguous'] for a,b in matrix['pairs']])
    metrics=[]
    for name in METHODS:
        for stratum,mask in [('all',np.ones(len(y),dtype=bool)),('ambiguous',ambiguous)]:
            p=probabilities[name];pred=p>=thresholds[name]
            metric=binary(y[mask],pred[mask],p[mask],total if stratum=='all' else None)
            metrics.append({'seed':seed,'model':name,'stratum':stratum,**metric})
    selected=[p for p,keep in zip(matrix['pairs'],ambiguous) if keep]
    ci=paired_bootstrap(y[ambiguous],probabilities['AttrPlus'][ambiguous]>=thresholds['AttrPlus'],
         probabilities['GraphWeighted'][ambiguous]>=thresholds['GraphWeighted'],selected,gold,seed,config['bootstrap_replicates'])
    write_json(out / f'wwi/seed_{seed}/paired_ci.json',{'contrast':'GraphWeighted_minus_AttrPlus','stratum':'ambiguous','ci_low':ci[0],'ci_high':ci[1]})
    write_rows(out / f'wwi/seed_{seed}/matching.csv',metrics)
    write_json(out / f'wwi/seed_{seed}/predictions.json',[{'candidate_id':pair_id(a,b),'left_source_uid':a,'right_source_uid':b,
        'label':int(y[i]),'ambiguous':bool(ambiguous[i]),'context_supported':bool(matrix['weighted'][i,-1]),
        **{name:float(p[i]) for name,p in probabilities.items()}} for i,(a,b) in enumerate(matrix['pairs'])])
    bounded,stats=bounded_candidates(test_records,config['blocking'],models['block_k'])
    write_json(out / f'wwi/seed_{seed}/blocking.json',{**stats,'blocking_recall':int(labels(bounded,gold).sum())/total,
               'missed_true_pairs':total-int(labels(bounded,gold).sum()),'denominator_true_pairs':total})
    positions={p:i for i,p in enumerate(matrix['pairs'])}
    selected_indices=np.array([positions[p] for p in bounded])
    bounded_y=labels(bounded,gold)
    write_rows(out / f'wwi/seed_{seed}/bounded_matching.csv',[{'seed':seed,'model':name,'blocker':'bounded',
          **binary(bounded_y,probabilities[name][selected_indices]>=thresholds[name],probabilities[name][selected_indices],total)} for name in METHODS])
    subgroups=[]
    degree={u:len(n) for u,n in graph_neighbors(graph,test_records).items()}
    masks={'Supplier':np.array([matrix['by_uid'][a]['entity_type']=='Supplier' for a,b in matrix['pairs']]),
           'StockItem':np.array([matrix['by_uid'][a]['entity_type']=='StockItem' for a,b in matrix['pairs']]),
           'supported':matrix['weighted'][:,-1]>0,'unsupported':matrix['weighted'][:,-1]==0,
           'sparse_left_degree_0_to_5':np.array([degree[a]<=5 for a,b in matrix['pairs']]),
           'dense_left_degree_over_5':np.array([degree[a]>5 for a,b in matrix['pairs']])}
    for stratum,mask in masks.items():
        if mask.any():
            subgroups.extend({'seed':seed,'model':name,'stratum':stratum,**binary(y[mask],probabilities[name][mask]>=thresholds[name])} for name in METHODS)
    write_rows(out / f'wwi/seed_{seed}/subgroups.csv',subgroups)
    calibration_rows=[];bins=[];risk=[]
    for name in ['AttrPlus','GraphWeighted']:
        for mode,p in [('uncalibrated',raw[name]),('calibrated',probabilities[name])]:
            summary,reliability_bins=reliability(y,p)
            calibration_rows.append({'seed':seed,'model':name,'mode':mode,**summary})
            bins.extend({'seed':seed,'model':name,'mode':mode,**r} for r in reliability_bins)
        for accept in sorted({thresholds[name], .50,.60,.70,.80,.90,.95,.99,1.000001}):
            risk.append({'seed':seed,'model':name,'accept':accept,'reject':config['models']['reject_probability'],
                         **coverage_risk(y,probabilities[name],accept,config['models']['reject_probability'])})
    write_rows(out / f'wwi/seed_{seed}/calibration.csv',calibration_rows)
    write_rows(out / f'wwi/seed_{seed}/reliability_bins.csv',bins)
    write_rows(out / f'wwi/seed_{seed}/coverage_risk.csv',risk)
    model_hash=digest(out / f'wwi/seed_{seed}/frozen_models.joblib')
    mapping=identity_decisions(graph,gold,records,matrix,probabilities['GraphWeighted'],thresholds['GraphWeighted'],config['models']['reject_probability'],out,seed,model_hash)
    with sql_database(graph,oracle_mapping(records,gold)) as sql:expected=sql_answers(sql)
    with sql_database(graph,mapping) as sql:actual_sql=sql_answers(sql)
    actual=PreparedGraphQueries(graph,mapping).execute()
    if actual!=actual_sql or actual!=graph_answers(graph,mapping):raise ValueError('Improved identity query parity failed')
    query_rows=[{'seed':seed,'query':q,**set_metrics(expected[q],actual[q]),'SQL_KG_equivalent':True} for q in SQL]
    write_rows(out / f'wwi/seed_{seed}/query_accuracy.csv',query_rows)
    robustness=[];perturbations=[]
    scenarios=[('clean',{}),('no_anchors',{})]
    for param,values in config['robustness'].items():
        argument={'attribute_noise':'attribute_noise','optional_missingness':'missingness','edge_removal':'edge_drop',
                  'misleading_edge_rewire':'rewire','provenance_missingness':'provenance_missingness'}[param]
        scenarios.extend((f'{param}_{int(v*100)}',{argument:v}) for v in values)
    for scenario_name,parameters in scenarios:
        changed,view,log=scenario(test_records,graph,seed+701,**parameters)
        changed_by_uid={r['source_uid']:r for r in changed}
        anchors=[] if scenario_name=='no_anchors' else infer_anchors(changed)
        neighborhood=weighted_graph_neighbors(view,changed)
        flat=weighted_flat_neighbors(view,changed)
        if neighborhood!=flat:raise ValueError('Perturbed graph/SQL evidence mismatch')
        kwargs={'min_anchors':config['models']['min_anchors'],'min_weighted_support':config['models']['min_weighted_support']}
        cx,ev=weighted_context(matrix['pairs'],changed_by_uid,neighborhood,anchors,**kwargs)
        cu,_=weighted_context(matrix['pairs'],changed_by_uid,neighborhood,anchors,weighted=False,**kwargs)
        cs,_=weighted_context(matrix['pairs'],changed_by_uid,neighborhood,anchors,support_features=False,**kwargs)
        cm={'ax':feature.transform(matrix['pairs'],changed_by_uid),'weighted':cx,'unweighted':cu,'no_support':cs}
        scored,_=method_scores(models,cm)
        scored.update(original_predictions(original['features'],original['models'],matrix['pairs'],changed,view,scenario_name=='no_anchors'))
        ci=paired_bootstrap(y,scored['AttrPlus']>=thresholds['AttrPlus'],scored['GraphWeighted']>=thresholds['GraphWeighted'],
                           matrix['pairs'],gold,seed,config['bootstrap_replicates'])
        for name in METHODS:
            p=scored[name];pred=p>=thresholds[name]
            robustness.append({'seed':seed,'scenario':scenario_name,'model':name,**binary(y,pred,p),
              **coverage_risk(y,p,thresholds[name],config['models']['reject_probability']),
              'supported_context_pairs':int(cx[:,-1].sum()),'same_frozen_candidates':True,
              'delta_f1_vs_AttrPlus':binary(y,pred)['f1']-binary(y,scored['AttrPlus']>=thresholds['AttrPlus'])['f1'],
              'delta_ci_low':ci[0] if name=='GraphWeighted' else None,'delta_ci_high':ci[1] if name=='GraphWeighted' else None})
        perturbations.append({'seed':seed,'scenario':scenario_name,**log})
    write_rows(out / f'wwi/seed_{seed}/robustness.csv',robustness)
    write_rows(out / f'wwi/seed_{seed}/perturbation_counts.csv',perturbations)
    print(f'WWI seed {seed}: richer attributes, disjoint calibration/policy, {len(scenarios)} robustness scenarios and physical identity view completed',flush=True)
    return {'records':records,'graph':graph,'gold':gold,'test':{'records':test_records},'mapping':mapping,
            'models':models,'features':feature,'matching':metrics,'robustness':robustness,'calibration':calibration_rows,'query':query_rows}

def run_wdc(config,base_config,source,out):
    archive=ROOT/'data/raw/wdc-products-50pair.zip'
    expected=read_json(source/'input_manifests/wdc_splits.json')
    if digest(archive)!=expected['archive_sha256']:raise ValueError('Official WDC archive checksum changed')
    data={split:read_json(ROOT/f'data/wdc/{split}.json') for split in ['train','validation','test']}
    validation_groups={}
    with zipfile.ZipFile(archive) as z:
        raw=gzip.decompress(z.read(base_config['wdc']['files']['validation']))
        import hashlib
        if hashlib.sha256(raw).hexdigest()!=expected['splits']['validation']['sha256_decompressed']:raise ValueError('WDC validation archive differs')
        for row in (json.loads(line) for line in raw.decode('utf-8').splitlines()):
            for side in ['left','right']:
                validation_groups['wdc:'+str(row['id_'+side])]=str(row['cluster_id_'+side])
    groups=sorted(set(validation_groups.values()));random.Random(config['development_partition_seed']).shuffle(groups)
    group_roles={g:ROLES[i%3] for i,g in enumerate(groups)}
    assignment={uid:group_roles[g] for uid,g in validation_groups.items()}
    feature=EnhancedAttributes().fit([r for pair in data['train'] for r in [pair['left'],pair['right']]])
    matrices_={}
    for split,rows in data.items():
        records={r['source_uid']:r for p in rows for r in [p['left'],p['right']]}
        pairs=[(p['left']['source_uid'],p['right']['source_uid']) for p in rows]
        matrices_[split]={'records':records,'pairs':pairs,'x':feature.transform(pairs,records),'y':np.array([p['label'] for p in rows])}
    t,v=matrices_['train'],matrices_['validation']
    masks=role_masks(v['pairs'],assignment)
    for role,mask in masks.items():
        if len(set(v['y'][mask]))!=2:raise ValueError('WDC development role lacks both classes: '+role)
    development_prior=float(v['y'].mean())
    estimator,selection=select_model(t['x'],t['y'],v['x'][masks['selection']],v['y'][masks['selection']],
                                    {**config['models'],'target_prior':development_prior},17)
    calibrator=SigmoidCalibration().fit(estimator.predict_proba(v['x'][masks['calibration']])[:,1],v['y'][masks['calibration']],
                                        sample_weight=prior_weights(v['y'][masks['calibration']],development_prior))
    policy=calibrator.transform(estimator.predict_proba(v['x'][masks['policy']])[:,1])
    accept=choose_threshold(policy,v['y'][masks['policy']],development_prior)
    wdcout=out/'wdc';wdcout.mkdir(parents=True,exist_ok=True)
    joblib.dump({'features':feature,'estimator':estimator,'calibrator':calibrator,'accept':accept},wdcout/'frozen_model.joblib')
    write_json(wdcout/'freeze.json',{'frozen_at_utc':utc_now(),'model_sha256':digest(wdcout/'frozen_model.joblib'),
      'selection':selection,'accept':accept,'official_split_counts':{s:len(rows) for s,rows in data.items()},
      'validation_roles':{role:{'groups':sum(r==role for r in group_roles.values()),'pairs':int(mask.sum()),'positives':int(v['y'][mask].sum())} for role,mask in masks.items()},
      'validation_cross_role_pairs_omitted':int(sum(~np.logical_or.reduce(list(masks.values())))),
      'official_train_and_test_pairs_unchanged':True,'calibration_and_policy_offers_group_disjoint':True,
      'calibration_roles':'internal division of official validation only; no benchmark train/test re-split',
      'development_prior':development_prior,'class_prior_correction':'weights restore full official-validation prevalence; no test-label prior correction',
      'attribute_only':True})
    test=matrices_['test'];y=test['y']
    raw=estimator.predict_proba(test['x'])[:,1];calibrated=calibrator.transform(raw)
    original=read_json(source/'e2/wdc/predictions.json')
    if [p['pair_id'] for p in data['test']]!=[p['pair_id'] for p in original]:raise ValueError('Official test pairs/order changed')
    previous=np.array([p['B2'] for p in original]);old_accept=read_json(source/'e2/wdc/freeze.json')['B2_threshold']
    result=[{'dataset':'WDC_Products','model':'OriginalB2',**binary(y,previous>=old_accept,previous)},
            {'dataset':'WDC_Products','model':'AttrPlus',**binary(y,calibrated>=accept,calibrated)}]
    write_rows(wdcout/'matching.csv',result)
    write_json(wdcout/'predictions.json',[{'pair_id':p['pair_id'],'label':int(y[i]),'OriginalB2':float(previous[i]),
             'AttrPlus':float(calibrated[i]),'AttrPlus_uncalibrated':float(raw[i])} for i,p in enumerate(data['test'])])
    calibration=[];bins=[]
    for mode,p in [('uncalibrated',raw),('calibrated',calibrated)]:
        metric,table=reliability(y,p);calibration.append({'mode':mode,**metric});bins.extend({'mode':mode,**r} for r in table)
    write_rows(wdcout/'calibration.csv',calibration);write_rows(wdcout/'reliability_bins.csv',bins)
    hard=np.array([bool(p['hard_negative']) or bool(p['label']) for p in data['test']])
    write_rows(wdcout/'hard_negative_challenge.csv',[{'model':'AttrPlus',**binary(y[hard],calibrated[hard]>=accept,calibrated[hard])}])
    # Independent attribute-only robustness on exactly the official test pairs.
    stress=[]
    for amount in [.1,.3,.5]:
        rng=random.Random(811)
        records=copy.deepcopy(test['records'])
        for record in records.values():
            for field in ['brand','description','price']:
                if record.get(field) and rng.random()<amount:record[field]=''
        p=calibrator.transform(estimator.predict_proba(feature.transform(test['pairs'],records))[:,1])
        stress.append({'scenario':f'optional_missingness_{int(amount*100)}',**binary(y,p>=accept,p)})
    write_rows(wdcout/'robustness.csv',stress)
    print('WDC: stronger attribute learner, disjoint validation roles, held-out calibration and missingness tests completed',flush=True)
    return result

def benchmark(bundle,config,out):
    blocking_rows=[];query_rows=[]
    k=bundle['models']['block_k']
    for factor in config['scale']['replica_factors']:
        for trial in range(config['scale']['repetitions']):
            gc.collect();start=time.perf_counter();graph,records=replica_workload(bundle,factor)
            build=time.perf_counter()-start
            expanded_gold={f"{r['source_uid']}:workload{i}":bundle['gold'][r['source_uid']] for i in range(factor) for r in bundle['test']['records']}
            positives=true_pair_count(records,expanded_gold)
            for method in ['original','bounded']:
                t=time.perf_counter()
                if method=='original':pairs,stats=candidates(records,{'token_prefix':config['blocking']['token_prefix'],
                    'include_category':True,'max_candidates':config['blocking']['max_candidates']})
                else:pairs,stats=bounded_candidates(records,config['blocking'],k)
                elapsed=time.perf_counter()-t
                retained=int(labels(pairs,expanded_gold).sum())
                score_timings={'attribute_feature_seconds':None,'AttrPlus_score_seconds':None,'GraphWeighted_score_seconds':None,'GraphUnweighted_score_seconds':None}
                if method=='bounded':
                    t=time.perf_counter()
                    m=matrices(records,graph,bundle['features'],config['blocking'],config['models'],fixed_pairs=pairs)
                    score_timings['attribute_feature_seconds']=time.perf_counter()-t
                    # Record comparable full probability paths on identical pairs.
                    for name in ['AttrPlus','GraphWeighted','GraphUnweighted']:
                        t=time.perf_counter();ax=m['ax'];attr=bundle['models']['calibrators']['attr'].transform(bundle['models']['attr'].predict_proba(ax)[:,1])
                        if name!='AttrPlus':
                            key='weighted' if name=='GraphWeighted' else 'unweighted'
                            context=bundle['models'][key].predict_proba(np.column_stack([ax,m[key]]))[:,1]
                            p=np.where(m[key][:,-1]>0,bundle['models']['calibrators'][key].transform(context),attr)
                        else:p=attr
                        score_timings[name+'_score_seconds']=time.perf_counter()-t
                blocking_rows.append({'factor':factor,'trial':trial,'blocker':method,'source_nodes':len(graph),
                    'master_records':len(records),'independent_original_groups':len(bundle['test']['records'])//2,
                    'candidate_count':len(pairs),'true_pair_universe':positives,'retained_true_pairs':retained,
                    'blocking_recall':retained/positives,'blocking_seconds':elapsed,'remap_build_seconds':build,
                    'per_left_limit':k if method=='bounded' else None,'postings_cap':stats.get('postings_cap'),
                    'cardinality_bound_respected':stats.get('cardinality_bound_respected'),
                    'resident_bytes_at_blocking':psutil.Process().memory_info().rss,**score_timings})
            # Fix the same namespace-disjoint identity decisions for all engines.
            # Replica namespace is used only in this query-workload construction,
            # never as a matching attribute or blocking feature.
            mapping={f'{u}:workload{i}':f'{canonical}:querycopy{i}' for i in range(factor) for u,canonical in bundle['mapping'].items()}
            t=time.perf_counter();prepared=PreparedGraphQueries(graph,mapping);graph_prepare=time.perf_counter()-t
            t=time.perf_counter();sql=sql_database(graph,mapping);sql_prepare=time.perf_counter()-t
            expected=sql_answers(sql)
            if prepared.execute()!=expected or graph_answers(graph,mapping)!=expected:raise ValueError('Prepared query parity failed')
            timings=defaultdict(list)
            for _ in range(config['scale']['warmed_query_repetitions']):
                for method,query in [('original_graph',lambda:graph_answers(graph,mapping)),('prepared_graph',prepared.execute),('indexed_sql',lambda:sql_answers(sql))]:
                    t=time.perf_counter();actual=query();elapsed=1000*(time.perf_counter()-t)
                    if actual!=expected:raise ValueError('Timed query parity failed')
                    timings[method].append(elapsed)
                # Also measure each prepared/SQL template separately.
                for q in SQL:
                    t=time.perf_counter();actual=prepared.query(q);elapsed=1000*(time.perf_counter()-t)
                    timings['prepared_'+q].append(elapsed)
                    t=time.perf_counter();actual_sql=set(sql.execute(SQL[q]).fetchall());elapsed=1000*(time.perf_counter()-t)
                    timings['sql_'+q].append(elapsed)
                    if actual!=actual_sql:raise ValueError('Per-query engine discrepancy')
            for method,elapsed in timings.items():
                query_rows.append({'factor':factor,'trial':trial,'method':method,'source_nodes':len(graph),
                    'latency_p50_ms':float(np.quantile(elapsed,.5)),'latency_p95_ms':float(np.quantile(elapsed,.95)),
                    'graph_prepare_seconds':graph_prepare,'sql_prepare_seconds':sql_prepare,
                    'same_identities':True,'warm_repetitions':len(elapsed)})
            sql.close();del prepared,graph;gc.collect()
            write_rows(out/'e5/blocking_scale.csv',blocking_rows)
            write_rows(out/'e5/query_scale.csv',query_rows)
            print(f'Extended E5 factor {factor}, trial {trial}: bounded retrieval recall measured; prepared graph/SQL query parity passed',flush=True)
    slopes={}
    for name in ['original','bounded']:
        data=[r for r in blocking_rows if r['blocker']==name]
        x=[np.mean([r['master_records'] for r in data if r['factor']==f]) for f in config['scale']['replica_factors']]
        y=[np.mean([r['candidate_count'] for r in data if r['factor']==f]) for f in config['scale']['replica_factors']]
        slopes[name]=float(np.polyfit(np.log(x),np.log(y),1)[0])
    write_json(out/'e5/summary.json',{'candidate_growth_exponents':slopes,'selected_k':k,
       'claim_limit':'bounded output cardinality is enforced by design, not proof of subquadratic total computation or unchanged recall',
       'memory_limit':'resident snapshots reported; original E5 remains the sampled peak-RSS reference',
       'cache':'one explicit warm-up; preparation separated from six-query batch and per-template timing',
       'replica_positive_universe':'all cross-silo aliases with the same original canonical group, including cross-replica positives',
       'independent_population_unchanged':True})
    return blocking_rows,query_rows

def charts(out,matching,robustness,calibration,blocking_rows,query_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    methods=['OriginalB2','OriginalB5','AttrPlus','GraphWeighted']
    values=[np.mean([r['f1'] for r in matching if r['model']==m and r['stratum']=='ambiguous']) for m in methods]
    axes[0,0].bar(range(4),values,color=['#9aa8b5','#728da1','#355f8d','#d76c31'])
    axes[0,0].set(xticks=range(4),xticklabels=['Old B2','Old B5','Attr+','Weighted KG'],ylim=(0,1.05),ylabel='Seed-mean F1',title='Exploratory ambiguous WWI comparison')
    for name in ['original','bounded']:
        scales=sorted({r['master_records'] for r in blocking_rows})
        values=[np.mean([r['candidate_count'] for r in blocking_rows if r['blocker']==name and r['master_records']==n]) for n in scales]
        axes[0,1].loglog(scales,values,'o-',label=name)
    axes[0,1].set(xlabel='Matching master records',ylabel='Candidates',title='Candidate output versus recall trade-off');axes[0,1].legend()
    for m in ['AttrPlus','GraphWeighted','GraphUnweighted']:
        scenarios=['clean','misleading_edge_rewire_15','misleading_edge_rewire_30','provenance_missingness_30','no_anchors']
        values=[np.mean([r['f1'] for r in robustness if r['model']==m and r['scenario']==s]) for s in scenarios]
        axes[1,0].plot(range(5),values,'o-',label=m)
    axes[1,0].set(xticks=range(5),xticklabels=['clean','rewire15','rewire30','lineage30','no anchors'],ylim=(0,1.05),ylabel='F1',title='Frozen models under misleading context');axes[1,0].legend(fontsize=8)
    for method in ['original_graph','prepared_graph','indexed_sql']:
        scales=sorted({r['source_nodes'] for r in query_rows})
        medians=[np.median([r['latency_p50_ms'] for r in query_rows if r['method']==method and r['source_nodes']==n]) for n in scales]
        axes[1,1].plot(scales,medians,'o-',label=method)
    axes[1,1].set(xlabel='Source nodes',ylabel='Six-query batch p50 (ms)',title='Same identities; preparation measured separately');axes[1,1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'improvements.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for seed in [17,29,43]:
        rows=read_json(out/f'wwi/seed_{seed}/predictions.json');y=np.array([r['label'] for r in rows])
        for i,method in enumerate(['AttrPlus','GraphWeighted']):
            p=np.array([r[method] for r in rows]);_,bins=reliability(y,p)
            axes[i].plot([r['mean_probability'] for r in bins],[r['positive_fraction'] for r in bins],'o-',label=f'seed {seed}')
    for ax,title in zip(axes,['Attribute reliability','Weighted-context reliability']):
        ax.plot([0,1],[0,1],'--',color='grey');ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability',ylabel='Observed positive fraction',title=title);ax.legend()
    fig.tight_layout();fig.savefig(out/'reliability.png',dpi=180);plt.close(fig)
    import csv
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    colors={'AttrPlus':'#355f8d','GraphWeighted':'#d76c31'}
    for seed in [17,29,43]:
        with (out/f'wwi/seed_{seed}/coverage_risk.csv').open(encoding='utf-8') as handle:rows=list(csv.DictReader(handle))
        for name,color in colors.items():
            selected=[r for r in rows if r['model']==name and r['selective_risk']]
            label=name if seed==17 else None
            axes[0].plot([float(r['coverage']) for r in selected],[float(r['selective_risk']) for r in selected],'o-',color=color,alpha=.6,label=label)
            selected=[r for r in rows if r['model']==name and r['accepted_error_rate']]
            axes[1].plot([float(r['accept']) for r in selected],[float(r['accepted_error_rate']) for r in selected],'o-',color=color,alpha=.6,label=label)
    axes[0].set(xlabel='Auto-decision coverage',ylabel='Error among auto-decisions',title='Selective risk (each line is one seed)')
    axes[1].set(xlabel='Acceptance threshold',ylabel='Error among accepted pairs',title='Accepted-pair error versus threshold')
    for ax in axes:ax.legend()
    fig.tight_layout();fig.savefig(out/'coverage_risk.png',dpi=180);plt.close(fig)

def report(out,matching,robustness,calibration,wdc,blocking_rows,query_rows):
    lines=['# Exploratory project improvements','',
      'The original `initial-core-v3` study and manuscript are preserved. These additions follow prior test review, so every new method/experiment is exploratory. All WWI scorers share the original candidate pairs; WDC remains attribute-only.','',
      '## Matching on the original ambiguous test pairs','',
      '| Method | Seed-mean F1 |','|---|---:|']
    for m in METHODS:
        mean=np.mean([r['f1'] for r in matching if r['model']==m and r['stratum']=='ambiguous'])
        lines.append(f'| {m} | {mean:.4f} |')
    lines+=['','Model selection, sigmoid calibration and threshold setting use disjoint canonical validation groups. Pairs crossing development roles are omitted from those operations. Test labels are evaluated only after models and policies are written to frozen artifacts.','',
       'Role filtering retains same-group positives more often than different-group negatives. Class weights restore full-validation prevalence for selection/calibration/policy using development labels only. This correction cannot restore omitted conditional comparisons; small-role uncertainty and development-to-test prior shift still limit interpretation.','',
       '## External WDC attribute results','','| Method | Precision | Recall | F1 |','|---|---:|---:|---:|']
    lines += [f"| {r['model']} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |" for r in wdc]
    lines += ['','Official train/test pairs are unchanged. The official validation partition is internally divided by product group for selection, calibration and policy; exact denominators are in `wdc/freeze.json`. This is a different exploratory development protocol from the original model, so attribute/model/calibration effects are combined.','',
      '## Robustness, provenance and abstention','',
      'The extension includes optional missingness, additional edge removal, type-safe misleading-edge rewiring, missing lineage metadata, no-anchor fallback, no-support-feature, no-context, no-gate and unit-weight ablations. Random perturbation logs are evaluator-only; injected edge labels never become model features. Original candidate pairs and trained thresholds remain frozen for every stress condition.','',
      '| Model | Raw mean ECE | After sigmoid mean ECE |','|---|---:|---:|']
    for m in ['AttrPlus','GraphWeighted']:
        raw=np.mean([r['ece'] for r in calibration if r['model']==m and r['mode']=='uncalibrated'])
        calibrated=np.mean([r['ece'] for r in calibration if r['model']==m and r['mode']=='calibrated'])
        lines.append(f'| {m} | {raw:.4f} | {calibrated:.4f} |')
    lines+=['','ECE uses ten uniform bins and is descriptive on a small, imbalanced sample. Brier/log-loss include discrimination and uncertainty; neither alone proves calibration. The calibrated and uncalibrated scores are retained. Coverage-risk and accepted-error rates are available alongside false-positive/negative counts.','',
      '## Bounded candidate retrieval and query preparation','',
      '| Matching records | Original candidates | Bounded candidates | Original blocking recall | Bounded blocking recall |','|---|---:|---:|---:|---:|']
    for n in sorted({r['master_records'] for r in blocking_rows}):
        old=next(r for r in blocking_rows if r['master_records']==n and r['blocker']=='original')
        new=next(r for r in blocking_rows if r['master_records']==n and r['blocker']=='bounded')
        lines.append(f"| {n} | {old['candidate_count']} | {new['candidate_count']} | {old['blocking_recall']:.4f} | {new['blocking_recall']:.4f} |")
    lines+=['','The per-left top-k and per-posting budgets enforce bounded output size. This is a designed trade-off, not proof of unchanged recall or subquadratic total compute. Recall counts **all** positive cross-silo aliases, including cross-replica pairs. Replica count does not increase independent population diversity.','',
      '| Source nodes | Original graph p50 ms | Prepared graph p50 ms | Indexed SQL p50 ms |','|---|---:|---:|---:|']
    for n in sorted({r['source_nodes'] for r in query_rows}):
        values=[np.median([r['latency_p50_ms'] for r in query_rows if r['source_nodes']==n and r['method']==m]) for m in ['original_graph','prepared_graph','indexed_sql']]
        lines.append(f"| {n} | {' | '.join(f'{v:.3f}' for v in values)} |")
    lines+=['','Preparation time is recorded separately for graph adjacency and SQL indexes. All methods return identical answers under the same identity map. Per-template p50/p95 values are saved, as well as six-template batch values. Timing describes these Python/SQLite implementations on the saved hardware; it is not an intrinsic database superiority claim.','',
       '## What remains incomplete','',
       'The core study does not yet include transaction date/quantity summaries in matching context, independent dirty ERP sources, embedding/neural comparisons, a Neo4j/Cypher deployment, extensive external robustness, a human review study, or final author/distribution/submission sign-off. Some are optional extensions; the plan coverage audit distinguishes these from required work. No positive hypothesis is required to count an honestly evaluated experiment as complete.','',
       '![Measured exploratory improvements](improvements.png)','', '![Held-out reliability diagrams](reliability.png)','',
       '![Selective risk and accepted-pair error](coverage_risk.png)']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def run(config_path,run_id):
    if Path(run_id).name!=run_id or run_id in {'.','..'}:raise ValueError('Invalid run ID')
    config=yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    source=ROOT/'results'/config['source_run']
    original=read_json(source/'run_manifest.json')
    for name,sha in original['code_sha256'].items():
        if digest(ROOT/name)!=sha:raise ValueError('Original frozen code changed: '+name)
    for name,sha in original['result_sha256'].items():
        if digest(source/name)!=sha:raise ValueError('Original frozen result changed: '+name)
    out=ROOT/'results'/run_id;out.mkdir(parents=True,exist_ok=False)
    (out/'config_snapshot.yaml').write_text(Path(config_path).read_text(encoding='utf-8'),encoding='utf-8')
    code={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'src/kgerp/research_plus').glob('*.py')}
    write_json(out/'manifest.json',{'status':'running','classification':config['classification'],'source_run':config['source_run'],'code_sha256':code})
    try:
        base_config=yaml.safe_load((source/'config_snapshot.yaml').read_text(encoding='utf-8'))
        matching=[];robustness=[];calibration=[];queries=[]
        for seed in config['seeds']:
            bundle=run_wwi(seed,config,base_config,source,out)
            matching.extend(bundle['matching']);robustness.extend(bundle['robustness']);calibration.extend(bundle['calibration']);queries.extend(bundle['query'])
        wdc=run_wdc(config,base_config,source,out)
        blocking_rows,query_rows=benchmark(bundle,config,out)
        write_rows(out/'wwi_matching.csv',matching);write_rows(out/'wwi_robustness.csv',robustness)
        write_rows(out/'wwi_calibration.csv',calibration);write_rows(out/'wwi_query_accuracy.csv',queries)
        charts(out,matching,robustness,calibration,blocking_rows,query_rows)
        report(out,matching,robustness,calibration,wdc,blocking_rows,query_rows)
        write_json(out/'manifest.json',{'status':'completed','classification':config['classification'],'source_run':config['source_run'],
          'original_results_unchanged':all(digest(source/name)==sha for name,sha in original['result_sha256'].items()),
          'original_code_unchanged':all(digest(ROOT/name)==sha for name,sha in original['code_sha256'].items()),
          'code_sha256':code,'config_sha256':digest(config_path),'hardware':environment(),
          'completed':['enhanced_WWI_and_WDC_attribute_baselines','disjoint_calibration_policy','weighted_context_and_ablations',
             'misleading_edges_missingness_provenance_no_anchors','coverage_risk_reliability','bounded_blocking_with_recall','prepared_queries_and_physical_canonical_view'],
          'result_sha256':{str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file() and p.name!='manifest.json'}})
        write_json(ROOT/'results/latest_improvements.json',{'run_id':run_id,'status':'completed'})
        print(f'Exploratory improvements completed: {out}',flush=True)
    except Exception as exc:
        write_json(out/'manifest.json',{'status':'failed','error':str(exc),'source_run':config['source_run'],'code_sha256':code})
        raise
    return out
