from collections import defaultdict
import copy
import gc
import importlib.metadata
import json
import platform
import random
import time
import threading
import tracemalloc
import joblib
import networkx as nx
import numpy as np
import psutil
from .io import ROOT, digest, utc_now, write_json, write_rows, read_json
from .schema import TABLES, reference_edges, uid
from .graph_loader import build_graph, save_graph
from .silo_generator import generate,save
from .blocking import candidates,pair_id,normalize
from .features_attribute import AttributeFeatures,NAMES as ATTRIBUTE_NAMES
from .features_context import assert_visible,infer_anchors,graph_neighbors,flat_neighbors,context_features,NAMES as CONTEXT_NAMES
from .matchers import train,score,acceptance,fit_learned,threshold
from .oracle import labels,true_pair_count,oracle_mapping
from .metrics import binary,set_metrics,paired_bootstrap
from .identity import resolve
from .queries import SQL,sql_database,sql_answers,graph_answers
from .wdc import import_splits

def environment():
    return {'timestamp_utc':utc_now(),'python':platform.python_version(),'platform':platform.platform(),
            'processor':platform.processor(),'logical_cpus':psutil.cpu_count(),
            'ram_bytes':psutil.virtual_memory().total,'backend':'NetworkX MultiDiGraph / indexed SQLite in-memory',
            'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','networkx','rapidfuzz','psutil']}}

def e1(tables,out):
    start=time.perf_counter();graph=build_graph(tables);elapsed=time.perf_counter()-start
    expected,invalid=reference_edges(tables)
    actual={(a,k,b) for a,b,k in graph.edges(keys=True)}
    expected_nodes={uid(t,r[TABLES[t][1]]) for t,rows in tables.items() for r in rows}
    before=(graph.number_of_nodes(),graph.number_of_edges())
    build_graph(tables,graph)
    idempotent=before==(graph.number_of_nodes(),graph.number_of_edges())
    lineage=sum(all(r.get(k) is not None for k in ['source_uid','source_table','source_record_id','source_batch','mapper_version']) for _,r in graph.nodes(data=True))
    rows=[{'table':t,'expected_nodes':len(records),'actual_nodes':sum(r['source_table']==t for _,r in graph.nodes(data=True))} for t,records in tables.items()]
    write_rows(out/'e1/fidelity.csv',rows)
    summary={'expected_nodes':len(expected_nodes),'actual_nodes':graph.number_of_nodes(),
             'node_precision':len(expected_nodes&set(graph))/max(1,len(graph)),'node_recall':len(expected_nodes&set(graph))/max(1,len(expected_nodes)),
             'expected_edges':len(expected),'actual_edges':len(actual),'edge_precision':len(expected&actual)/max(1,len(actual)),
             'edge_recall':len(expected&actual)/max(1,len(expected)),'orphan_count':len(invalid),
             'lineage_coverage':lineage/max(1,len(graph)),'idempotent':idempotent,'build_seconds':elapsed}
    write_json(out/'e1/summary.json',summary)
    write_json(out/'e1/mapping_errors.json',{'missing':sorted(expected-actual),'spurious':sorted(actual-expected)})
    sql_controls=[]
    for reference in read_json(ROOT/'data/manifests/sql_reference_counts.json'):
        prefix='wwi:'+reference['source_table']+':'
        count=sum(a.startswith(prefix) and k==reference['relation'] for a,k,b in actual)
        sql_controls.append({**reference,'graph_edge_count':count,'equal':count==reference['valid_source_join_rows']})
    write_rows(out/'e1/sql_join_controls.csv',sql_controls)
    if not all(r['equal'] for r in sql_controls):raise ValueError('Independent SQL reference count disagreement')
    rng=random.Random(17)
    audit=[]
    for u in rng.sample(sorted(graph),min(100,len(graph))):
        r=graph.nodes[u];pk=TABLES[r['source_table']][1]
        row=next(row for row in tables[r['source_table']] if str(row[pk])==r['source_record_id'])
        audit.append({'source_uid':u,'table':r['source_table'],'row_id':r['source_record_id'],'attributes_equal':row==r['attributes']})
    write_rows(out/'e1/provenance_audit.csv',audit)
    if expected!=actual or set(graph)!=expected_nodes or not idempotent or lineage!=len(graph):
        raise ValueError('E1 integrity blocker; downstream experiments stopped')
    save_graph(graph)
    del graph;gc.collect()
    print(f'E1 passed: {len(expected_nodes):,} nodes, {len(expected):,} relationships',flush=True)
    return summary

def development(records,graph,gold,config):
    assert_visible(records)
    features=AttributeFeatures().fit([r for r in records if r['split']=='train'])
    matrices={}
    for split in ('train','validation','test'):
        selected=[r for r in records if r['split']==split]
        by_uid={r['source_uid']:r for r in selected}
        pairs,stats=candidates(selected,config['blocker'])
        anchors=infer_anchors(selected)
        gn,fn=graph_neighbors(graph,selected),flat_neighbors(graph,selected)
        if gn!=fn:raise ValueError('Flat SQL and graph context neighborhoods differ')
        context,evidence=context_features(pairs,by_uid,gn,anchors,config['models']['min_anchors'])
        flat,_=context_features(pairs,by_uid,fn,anchors,config['models']['min_anchors'])
        if not np.array_equal(context,flat):raise ValueError('Flat/graph feature mismatch')
        matrices[split]={'records':selected,'by_uid':by_uid,'pairs':pairs,'stats':stats,'anchors':anchors,
                         'ax':features.transform(pairs,by_uid),'cx':context,'evidence':evidence}
    # Only development partitions read labels before freezing the model.
    t,v=matrices['train'],matrices['validation']
    models=train((t['ax'],t['cx'],labels(t['pairs'],gold)),(v['ax'],v['cx'],labels(v['pairs'],gold)),config['models'])
    return features,models,matrices

def e2_seed(tables,config,seed,out):
    records,graph,gold=generate(tables,seed,config['corruption'],config['group_split_seed'])
    save(records,graph,gold,seed)
    # Query non-vacuity is a dataset-construction gate, checked before fitting.
    with sql_database(graph,oracle_mapping(records,gold)) as sql:preflight=sql_answers(sql)
    empty=[q for q,answers in preflight.items() if not answers]
    if empty:raise ValueError(f'Empty query oracle in dataset preflight: {empty}')
    write_json(out/f'e3/seed_{seed}/oracle_preflight.json',{q:len(a) for q,a in preflight.items()})
    features,models,m=development(records,graph,gold,config)
    model_dir=out/f'e2/seed_{seed}'
    model_dir.mkdir(parents=True,exist_ok=True)
    joblib.dump({'features':features,'models':models},model_dir/'frozen_models.joblib')
    model_hash=digest(model_dir/'frozen_models.joblib')
    write_json(model_dir/'freeze.json',{'frozen_at_utc':utc_now(),'model_sha256':model_hash,
              'thresholds':{name:acceptance(models,name) for name in ['B0','B1','B2','B4','B5']},
              'attribute_features':ATTRIBUTE_NAMES,'context_features':CONTEXT_NAMES,
              'validation_f1':{'B2':models['B2']['validation_f1'],'context':models['context']['validation_f1']}})
    test=m['test'];pairs=test['pairs'];y=labels(pairs,gold)
    scores=score(models,test['ax'],test['cx'])
    # Predeclared ambiguous subset uses attributes alone: name ratio < .95 OR
    # more than one candidate with token-sort similarity >= .65 per left record.
    similar=defaultdict(int)
    for i,(a,b) in enumerate(pairs):similar[a]+=int(test['ax'][i,1]>=.65)
    ambiguous=np.array([test['ax'][i,0]<.95 or similar[a]>1 for i,(a,b) in enumerate(pairs)])
    total=true_pair_count(test['records'],gold)
    rows=[]
    for name in ['B0','B1','B2','B4','B5']:
        pred=scores[name]>=acceptance(models,name)
        for stratum,mask in [('all',np.ones(len(y),dtype=bool)),('ambiguous',ambiguous)]:
            metrics=binary(y[mask],pred[mask],scores[name][mask],total if stratum=='all' else None)
            selected_pairs=[p for p,keep in zip(pairs,mask) if keep]
            ci=paired_bootstrap(y[mask],scores['B2'][mask]>=acceptance(models,'B2'),pred[mask],selected_pairs,gold,seed,config['bootstrap_replicates'])
            base=binary(y[mask],scores['B2'][mask]>=acceptance(models,'B2'))['f1']
            rows.append({'dataset':'WWI_silo_simulation','split':'test','seed':seed,'model':name,'stratum':stratum,
                         'n_entities':len({gold[r['source_uid']]['canonical'] for r in test['records']}),**metrics,
                         'delta_f1_vs_B2':metrics['f1']-base,'ci_low':ci[0],'ci_high':ci[1]})
    write_rows(model_dir/'metrics.csv',rows)
    write_json(model_dir/'candidate_statistics.json',{**test['stats'],'eligible_true_pairs':total,
               'retained_true_pairs':int(y.sum()),'same_candidates_all_models':True})
    predictions=[]
    for i,(a,b) in enumerate(pairs):
        predictions.append({'candidate_id':pair_id(a,b),'left_source_uid':a,'right_source_uid':b,'label':int(y[i]),
                            'ambiguous':bool(ambiguous[i]),'context_supported':bool(test['cx'][i,-1]),
                            **{name:float(values[i]) for name,values in scores.items()}})
    write_json(model_dir/'predictions.json',predictions)
    write_json(model_dir/'anchors.json',[{'left_source_uid':a,'right_source_uid':b,'rule':'unique_normalized_exact_name_same_type_compatible_category'} for a,b in test['anchors']])
    all_by_uid={r['source_uid']:r for r in records}
    mapping,decisions=resolve(pairs,scores['B5'],all_by_uid,test['evidence'],acceptance(models,'B5'),models['reject'],f'seed_{seed}',model_hash)
    write_json(model_dir/'decisions.json',decisions)
    accepted=[d for d in decisions if d['status']=='accepted']
    provenance={'accepted_links':len(accepted),'provenance_complete':all(all(k in d for k in ['candidate_id','method_version','evidence_source_uids','threshold_policy','run_id','decision_trigger']) for d in accepted),
                'accepted_false_merges':sum(gold[d['left_source_uid']]['canonical']!=gold[d['right_source_uid']]['canonical'] for d in accepted)}
    write_json(model_dir/'link_audit.json',provenance)
    print(f'E2 seed {seed}: {len(pairs):,} common test candidates; {len(accepted)} accepted links',flush=True)
    return {'records':records,'graph':graph,'gold':gold,'models':models,'features':features,'test':test,
            'scores':scores,'mapping':mapping,'metrics':rows}

def e3(bundle,seed,out):
    graph,gold,records=bundle['graph'],bundle['gold'],bundle['records']
    predicted=bundle['mapping']
    oracle=oracle_mapping(records,gold)
    raw={r['source_uid']:r['source_uid'] for r in records}
    with sql_database(graph,oracle) as sql:expected=sql_answers(sql)
    if any(not answers for answers in expected.values()):raise ValueError('E3 requires nonempty oracle tasks')
    variants={}
    with sql_database(graph,raw) as sql:variants['R0_unreconciled_SQL']=sql_answers(sql)
    variants['R1_predicted_KG']=graph_answers(graph,predicted)
    with sql_database(graph,predicted) as sql:variants['R2_same_identity_SQL']=sql_answers(sql)
    if graph_answers(graph,oracle)!=expected:raise ValueError('Oracle SQL/graph disagreement')
    rows=[]
    for name,answers in variants.items():
        for q,actual in answers.items():
            rows.append({'seed':seed,'query':q,'variant':name,**set_metrics(expected[q],actual),
                         'KG_SQL_equivalent':variants['R1_predicted_KG'][q]==variants['R2_same_identity_SQL'][q]})
    if variants['R1_predicted_KG']!=variants['R2_same_identity_SQL']:raise ValueError('E3 canonical SQL/KG discrepancy')
    write_rows(out/f'e3/seed_{seed}/query_accuracy.csv',rows)
    write_json(out/f'e3/seed_{seed}/answers.json',{'oracle':{q:sorted(a) for q,a in expected.items()},
               **{v:{q:sorted(a) for q,a in ans.items()} for v,ans in variants.items()}})
    print(f'E3 seed {seed}: six oracle queries, SQL/KG equivalence passed',flush=True)
    return rows

def e4(bundle,seed,out,config):
    test=bundle['test'];pairs=test['pairs'];gold=bundle['gold'];models=bundle['models'];features=bundle['features']
    y=labels(pairs,gold);rows=[]
    scenarios=[('clean',0.,0.,False),('attribute_noise_30',.3,0.,False),('attribute_noise_50',.5,0.,False),
               ('edge_drop_15',0.,.15,False),('edge_drop_30',0.,.3,False),('no_anchors',0.,0.,True)]
    for name,noise,drop,no_anchors in scenarios:
        rng=random.Random(seed+101)
        records=copy.deepcopy(test['records'])
        for r in records:
            if rng.random()<noise:
                words=r['name'].split();r['name']=' '.join(w[:max(1,len(w)//2)] for w in words)
                r['brand']='';r['category']='';r['description']=''
        by_uid={r['source_uid']:r for r in records}
        graph=bundle['graph'].copy() if drop else bundle['graph']
        if drop:
            for a,b,k in sorted(graph.edges(keys=True)):
                if k=='FROM_SUPPLIER' and rng.random()<drop:graph.remove_edge(a,b,k)
        anchors=[] if no_anchors else infer_anchors(records)
        cx,evidence=context_features(pairs,by_uid,graph_neighbors(graph,records),anchors,config['models']['min_anchors'])
        ax=features.transform(pairs,by_uid);scores=score(models,ax,cx)
        for method in ['B2','B4','B5','B5_no_context','B5_ungated']:
            threshold_=acceptance(models,method);s=scores[method]
            pred=s>=threshold_
            decided=(s>=threshold_)|(s<=min(models['reject'],threshold_))
            error=(pred!=y.astype(bool))
            rows.append({'seed':seed,'scenario':name,'model':method,**binary(y,pred,s),
                         'decision_coverage':float(decided.mean()),'selective_risk':float(error[decided].mean()) if decided.any() else 0,
                         'supported_context_pairs':int(cx[:,-1].sum()),'same_clean_candidates':True})
    write_rows(out/f'e4/seed_{seed}/robustness.csv',rows)
    print(f'E4 seed {seed}: six noise/anchor scenarios and two ablations completed',flush=True)
    return rows

def wdc_evaluate(config,out):
    data=import_splits(config)
    features=AttributeFeatures().fit([r for p in data['train'] for r in (p['left'],p['right'])])
    matrices={}
    for split,rows in data.items():
        records={r['source_uid']:r for p in rows for r in (p['left'],p['right'])}
        pairs=[(p['left']['source_uid'],p['right']['source_uid']) for p in rows]
        matrices[split]=(features.transform(pairs,records),np.array([p['label'] for p in rows]))
    tx,ty=matrices['train'];vx,vy=matrices['validation']
    model=fit_learned(tx,ty,vx,vy,[.1,1.,10.])
    fuzzy=threshold(vx[:,1],vy,[.55,.65,.75,.85,.95])
    outdir=out/'e2/wdc';outdir.mkdir(parents=True,exist_ok=True)
    joblib.dump({'features':features,'model':model},outdir/'frozen_attribute_model.joblib')
    write_json(outdir/'freeze.json',{'model_sha256':digest(outdir/'frozen_attribute_model.joblib'),'B2_threshold':model['accept'],'B1_threshold':fuzzy,'config':config,'frozen_at_utc':utc_now()})
    test_x,y=matrices['test'];scores={'B0':((test_x[:,0]==1)&(test_x[:,9]==0)).astype(float),'B1':test_x[:,1],'B2':model['estimator'].predict_proba(test_x)[:,1]}
    metrics=[]
    for name,s in scores.items():
        accept=1 if name=='B0' else fuzzy if name=='B1' else model['accept']
        metrics.append({'dataset':f"WDC_Products_50_small_{config['unseen_products']}unseen",'model':name,'split':'test',**binary(y,s>=accept,s)})
    write_rows(outdir/'wdc_attribute_only.csv',metrics)
    write_json(outdir/'predictions.json',[{'pair_id':p['pair_id'],'label':p['label'],'hard_negative':p['hard_negative'],**{name:float(s[i]) for name,s in scores.items()}} for i,p in enumerate(data['test'])])
    hard=np.array([p['hard_negative'] for p in data['test']])|y.astype(bool)
    write_rows(outdir/'hard_negative_challenge.csv',[{'model':name,**binary(y[hard],s[hard]>=(1 if name=='B0' else fuzzy if name=='B1' else model['accept']),s[hard])} for name,s in scores.items()])
    print(f'WDC: official 2500/2500/4500 splits evaluated attribute-only',flush=True)
    return metrics

def replica_workload(bundle,factor):
    base=bundle['graph'];graph=nx.MultiDiGraph();records=[]
    for replica in range(factor):
        mapping={u:f'{u}:workload{replica}' for u in base}
        for u,attrs in base.nodes(data=True):
            r=dict(attrs);r['source_uid']=mapping[u]
            graph.add_node(mapping[u],**r)
        for a,b,k in base.edges(keys=True):graph.add_edge(mapping[a],mapping[b],key=k,relation=k)
        for r in bundle['test']['records']:
            value=dict(r);value['source_uid']=mapping[r['source_uid']];records.append(value)
    return graph,records

def e5(bundle,out,config):
    rows=[];latencies=[]
    for factor in config['scale']['replicas']:
        for trial in range(config['scale']['repetitions']):
            gc.collect()
            process=psutil.Process();rss_start=process.memory_info().rss;peak=[rss_start]
            stop=threading.Event()
            def sample():
                while not stop.wait(.05):peak[0]=max(peak[0],process.memory_info().rss)
            thread=threading.Thread(target=sample,daemon=True);thread.start()
            started=time.perf_counter()
            row={'factor':factor,'trial':trial,'status':'completed','source_records':0,'master_records':0,'graph_edges':0,
                 'unique_original_test_entities':len(bundle['test']['records'])//2,'candidate_count':0,'possible_pairs':0,
                 'reduction_ratio':0.,'max_index_block':0,'remap_build_seconds':0.,'blocking_seconds':0.,
                 'attribute_feature_seconds':0.,'graph_context_seconds':0.,'flat_context_seconds':0.,
                 'B2_score_seconds':0.,'B4_score_seconds':0.,'B5_score_seconds':0.,'identity_seconds':0.,
                 'sql_index_seconds':0.,'sql_batch_p50_ms':0.,'sql_batch_p95_ms':0.,'graph_batch_p50_ms':0.,
                 'graph_batch_p95_ms':0.,'peak_rss_bytes':0,'rss_increment_bytes':0,'total_seconds':0.,'failure':''}
            graph=None;sql=None
            try:
                graph,records=replica_workload(bundle,factor)
                row.update(source_records=len(graph),master_records=len(records),graph_edges=graph.number_of_edges(),remap_build_seconds=time.perf_counter()-started)
                if process.memory_info().rss>config['scale']['memory_limit_gb']*1024**3:raise RuntimeError('Measured RSS budget exceeded')
                t=time.perf_counter();pairs,stats=candidates(records,{**config['blocker'],'max_candidates':config['scale']['max_candidates']})
                row.update(stats);row['blocking_seconds']=time.perf_counter()-t
                by_uid={r['source_uid']:r for r in records}
                t=time.perf_counter();ax=bundle['features'].transform(pairs,by_uid);row['attribute_feature_seconds']=time.perf_counter()-t
                anchors=infer_anchors(records)
                t=time.perf_counter();gn=graph_neighbors(graph,records);cx,evidence=context_features(pairs,by_uid,gn,anchors,config['models']['min_anchors']);row['graph_context_seconds']=time.perf_counter()-t
                t=time.perf_counter();fn=flat_neighbors(graph,records);fx,_=context_features(pairs,by_uid,fn,anchors,config['models']['min_anchors']);row['flat_context_seconds']=time.perf_counter()-t
                if not np.array_equal(cx,fx):raise ValueError('E5 graph/flat feature discrepancy')
                models=bundle['models']
                t=time.perf_counter();b2=models['B2']['estimator'].predict_proba(ax)[:,1];row['B2_score_seconds']=time.perf_counter()-t
                t=time.perf_counter();b4=models['context']['estimator'].predict_proba(np.column_stack([ax,fx]))[:,1];row['B4_score_seconds']=time.perf_counter()-t
                t=time.perf_counter();raw=models['context']['estimator'].predict_proba(np.column_stack([ax,cx]))[:,1];b5=np.where(cx[:,-1]>0,raw,b2);row['B5_score_seconds']=time.perf_counter()-t
                t=time.perf_counter();mapping,_=resolve(pairs,b5,by_uid,evidence,acceptance(models,'B5'),models['reject'],'scale','frozen');row['identity_seconds']=time.perf_counter()-t
                # Include all source masters in the canonical query view.
                all_mapping={u:u for u,r in graph.nodes(data=True) if r['entity_type'] in {'Supplier','StockItem'}}
                all_mapping.update(mapping)
                t=time.perf_counter();sql=sql_database(graph,all_mapping);row['sql_index_seconds']=time.perf_counter()-t
                sql_answers(sql);graph_answers(graph,all_mapping)  # explicit warm-up
                sql_times,graph_times=[],[]
                for _ in range(5):
                    t=time.perf_counter();sq=sql_answers(sql);sql_times.append(1000*(time.perf_counter()-t))
                    t=time.perf_counter();gq=graph_answers(graph,all_mapping);graph_times.append(1000*(time.perf_counter()-t))
                    if sq!=gq:raise ValueError('E5 SQL/graph query discrepancy')
                row.update(sql_batch_p50_ms=float(np.quantile(sql_times,.5)),sql_batch_p95_ms=float(np.quantile(sql_times,.95)),
                           graph_batch_p50_ms=float(np.quantile(graph_times,.5)),graph_batch_p95_ms=float(np.quantile(graph_times,.95)))
            except (MemoryError,RuntimeError) as exc:
                row['status']='resource_limited';row['failure']=str(exc)
            finally:
                if sql is not None:sql.close()
                stop.set();thread.join();peak[0]=max(peak[0],process.memory_info().rss)
                row.update(peak_rss_bytes=peak[0],rss_increment_bytes=max(0,peak[0]-rss_start),total_seconds=time.perf_counter()-started)
                rows.append(row)
                write_rows(out/'e5/scale_trials.csv',rows)
                print(f"E5 factor {factor}, trial {trial}: {row['source_records']:,} nodes, {row['candidate_count']:,} candidates ({row['status']})",flush=True)
                del graph;gc.collect()
    valid=[r for r in rows if r['status']=='completed']
    scales=sorted({r['factor'] for r in valid})
    slope=None
    if len(scales)>=3:
        x=[np.mean([r['master_records'] for r in valid if r['factor']==f]) for f in scales]
        y=[np.mean([r['candidate_count'] for r in valid if r['factor']==f]) for f in scales]
        slope=float(np.polyfit(np.log(x),np.log(y),1)[0])
    write_json(out/'e5/summary.json',{'completed_scale_points':len(scales),'candidate_growth_exponent':slope,
               'hardware':environment(),'cache':'five warmed six-query batches after one explicit warm-up',
               'workload':'namespace-disjoint replicas; full generated transaction graph, test-master matching workload',
               'independent_diversity':'unchanged; replicas duplicate descriptive attributes and do not add independent entities',
               'anchor_behavior':'unique-name anchors naturally disappear as identical replicas increase',
               'memory':'process RSS sampled every 50ms; absolute peak and incremental peak, allocator retention possible',
               'B5_timing':'context inference + cached B2 fallback; sum feature times for end-to-end overhead'})
    return rows

def plots(out,e2rows,e3rows,e4rows,e5rows,wdcrows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    models=['B0','B1','B2','B4','B5']
    for i,name in enumerate(models):
        values=[r['f1'] for r in e2rows if r['model']==name and r['stratum']=='ambiguous']
        axes[0,0].bar(i,np.mean(values),yerr=np.std(values),color='#286c8e')
    axes[0,0].set(xticks=range(5),xticklabels=models,ylim=(0,1.05),title='WWI ambiguous test F1 (seed mean ± SD)',ylabel='F1')
    queries=list(SQL)
    for name,offset,color in [('R0_unreconciled_SQL',-.2,'#a9b9c5'),('R1_predicted_KG',.2,'#286c8e')]:
        values=[np.mean([r['f1'] for r in e3rows if r['variant']==name and r['query']==q]) for q in queries]
        axes[0,1].bar(np.arange(6)+offset,values,width=.4,label=name,color=color)
    axes[0,1].set(xticks=range(6),xticklabels=queries,ylim=(0,1.05),title='Six query templates against oracle',ylabel='F1');axes[0,1].legend(fontsize=8)
    scenarios=['clean','attribute_noise_30','attribute_noise_50','edge_drop_15','edge_drop_30','no_anchors']
    for name in ['B2','B5','B5_ungated']:
        values=[np.mean([r['f1'] for r in e4rows if r['scenario']==s and r['model']==name]) for s in scenarios]
        axes[1,0].plot(range(6),values,marker='o',label=name)
    axes[1,0].set(xticks=range(6),xticklabels=['clean','attr30','attr50','edge15','edge30','no anchor'],ylim=(0,1.05),title='Frozen models under corruption',ylabel='F1');axes[1,0].legend()
    valid=[r for r in e5rows if r['status']=='completed']
    if valid:
        axes[1,1].scatter([r['master_records'] for r in valid],[r['candidate_count'] for r in valid],color='#286c8e')
        axes[1,1].set(xscale='log',yscale='log',xlabel='Source test-master records',ylabel='Candidates',title='Measured candidate growth')
    fig.tight_layout();fig.savefig(out/'evaluation_summary.png',dpi=180);plt.close(fig)
    write_rows(out/'e2/wwi_pairwise_primary.csv',e2rows)
    write_rows(out/'e3/query_accuracy.csv',e3rows)
    write_rows(out/'e4/robustness.csv',e4rows)
    write_rows(out/'e2/wdc_attribute_only.csv',wdcrows)
