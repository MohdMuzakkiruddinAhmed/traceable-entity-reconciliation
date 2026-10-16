"""Three source-node scale targets, blocking recall, peak RSS and equal queries."""
from collections import Counter,defaultdict
import gc
import threading
import time
import networkx as nx
import numpy as np
import psutil
from ..io import write_json,write_rows
from ..oracle import labels,true_pair_count
from ..metrics import binary
from ..research_plus.blocking import bounded_candidates
from .experiments import blocker,matrix,score
from .queries import PreparedQueries,SQLQueries,SQL


class MemorySample:
    def __enter__(self):
        self.process=psutil.Process();self.start=self.process.memory_info().rss;self.peak=self.start
        self.stop=threading.Event()
        def poll():
            while not self.stop.wait(.05):self.peak=max(self.peak,self.process.memory_info().rss)
        self.thread=threading.Thread(target=poll,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.stop.set();self.thread.join();self.peak=max(self.peak,self.process.memory_info().rss)


def workload(bundle,target):
    base=bundle['scope'];by_uid={r['source_uid']:r for r in bundle['test_records']}
    groups=defaultdict(list)
    for uid in by_uid:groups[bundle['gold'][uid]['canonical']].append(uid)
    units=[]
    for canonical,masters in sorted(groups.items()):
        nodes=set(masters)
        for uid in masters:
            for _,tx,k in base.out_edges(uid,keys=True):
                if k=='HAS_TRANSACTION':nodes.add(tx);nodes.update(t for _,t,key in base.out_edges(tx,keys=True) if key=='HAS_CATEGORY')
        units.append((canonical,masters,base.subgraph(nodes)))
    graph=nx.MultiDiGraph();records=[];gold={};identity={};copy_number=0;done=False
    while not done:
        for canonical,masters,unit in units:
            names={u:u+':scale'+str(copy_number) for u in unit}
            for u,a in unit.nodes(data=True):graph.add_node(names[u],**{**a,'source_uid':names[u]})
            for a,b,k in unit.edges(keys=True):graph.add_edge(names[a],names[b],key=k,relation=k)
            for uid in masters:
                records.append({**by_uid[uid],'source_uid':names[uid]})
                gold[names[uid]]={'canonical':canonical}
                # Query identity is deliberately per-copy oracle control. It
                # never enters matching features or candidate generation.
                identity[names[uid]]='query:'+canonical+':copy'+str(copy_number)
            if len(graph)>=target:done=True;break
        # Preserve permitted native master-to-master relations in a copy after
        # both endpoint units have been included (needed for WWI anchor context).
        for a,b,k in base.edges(keys=True):
            if k not in {'HAS_TRANSACTION','HAS_CATEGORY'}:
                left=a+':scale'+str(copy_number);right=b+':scale'+str(copy_number)
                if left in graph and right in graph:graph.add_edge(left,right,key=k,relation=k)
        copy_number+=1
    return graph,records,gold,identity


def benchmark(bundle,config,embedding,out):
    out=out/'e5';out.mkdir(parents=True,exist_ok=True)
    trials=[];queries=[];matching=[]
    for target in config['scale']['node_targets']:
        for trial in range(config['scale']['repetitions']):
            gc.collect();started=time.perf_counter()
            with MemorySample() as memory:
                graph,records,gold,identity=workload(bundle,target);build=time.perf_counter()-started
                t=time.perf_counter();pairs,stats=bounded_candidates(records,blocker(config),bundle['models']['retrieval_k']);blocking=time.perf_counter()-t
                if psutil.Process().memory_info().rss>16*1024**3:raise RuntimeError('Expanded E5 16 GiB process-RSS budget exceeded')
                t=time.perf_counter();m=matrix(records,graph,bundle['attribute'],embedding,config,pairs=pairs);features=time.perf_counter()-t
                t=time.perf_counter();probabilities,_=score(bundle['models'],m);scoring=time.perf_counter()-t
                y=labels(pairs,gold);positive_universe=true_pair_count(records,gold)
                for name,p in probabilities.items():matching.append({'dataset':bundle['dataset'],'target_nodes':target,'trial':trial,'model':name,
                    **binary(y,p>=bundle['models']['accepts'][name],p,positive_universe)})
                t=time.perf_counter();prepared=PreparedQueries(graph,identity);graph_prepare=time.perf_counter()-t
                t=time.perf_counter();sql=SQLQueries(graph,identity);sql_prepare=time.perf_counter()-t
                expected=sql.execute()
                if prepared.execute()!=expected:raise ValueError('Expanded E5 query parity')
                times=defaultdict(list)
                for repetition in range(config['scale']['warm_repetitions']):
                    for method,query in [('graph_batch',prepared.execute),('sql_batch',sql.execute)]:
                        t=time.perf_counter();actual=query();times[method].append(1000*(time.perf_counter()-t))
                        if actual!=expected:raise ValueError('Timed E5 parity')
                    for q in SQL:
                        for method,query in [('graph_'+q,lambda q=q:prepared.query(q)),('sql_'+q,lambda q=q:sql.query(q))]:
                            t=time.perf_counter();actual=query();times[method].append(1000*(time.perf_counter()-t))
                            if actual!=expected[q]:raise ValueError('Per-template E5 parity')
                for method,latencies in times.items():queries.append({'dataset':bundle['dataset'],'target_nodes':target,'actual_nodes':len(graph),'trial':trial,'method':method,
                    'p50_ms':float(np.quantile(latencies,.5)),'p95_ms':float(np.quantile(latencies,.95)),'warm_repetitions':len(latencies),
                    'graph_prepare_seconds':graph_prepare,'sql_prepare_seconds':sql_prepare,'same_oracle_identity_map':True})
                sql.close()
            trials.append({'dataset':bundle['dataset'],'target_nodes':target,'actual_nodes':len(graph),'trial':trial,'edges':graph.number_of_edges(),
                'matching_records':len(records),'independent_original_groups':len({g['canonical'] for g in gold.values()}),'true_positive_pair_universe':positive_universe,
                'retained_true_pairs':int(y.sum()),'blocking_recall':int(y.sum())/positive_universe,'candidate_count':len(pairs),
                'build_seconds':build,'blocking_seconds':blocking,'full_feature_seconds':features,'all_scorers_seconds':scoring,
                'per_left_candidate_cap':bundle['models']['retrieval_k'],'sampled_peak_rss_bytes':memory.peak,'rss_increment_bytes':max(0,memory.peak-memory.start),
                'total_seconds':time.perf_counter()-started,'status':'completed'})
            write_rows(out/'scale_trials.csv',trials);write_rows(out/'query_latencies.csv',queries);write_rows(out/'matching_at_scale.csv',matching)
            print(f'{bundle["dataset"]} E5 {target:,} target/trial {trial}: {len(graph):,} nodes, {len(pairs):,} candidates, recall {int(y.sum())/positive_universe:.3f}',flush=True)
            del graph,records,m,prepared,probabilities;gc.collect()
    write_json(out/'protocol.json',{'targets':config['scale']['node_targets'],'trials_per_target':3,'memory':'process RSS sampled every 50ms; allocator retention can influence absolute baseline',
        'workload':'complete source-view entity/transaction/category units; grow through independent test groups, then namespace-disjoint replicas',
        'positive_universe':'all cross-silo aliases of the same original entity, including cross-replica positives; no inferred copy IDs in features',
        'query_control':'identical per-copy oracle identities for graph and SQL; answers and per-template latency verified',
        'preparation':'separate graph adjacency preparation and SQL loading/indexing; one warm-up before five measured repeats',
        'limits':'bounded output is by design; lower recall is reported; this is not subquadratic-computation or independent-enterprise evidence'})
    return trials
