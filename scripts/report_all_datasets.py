"""Generate a five-dataset research report directly from measured artifacts."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from kgerp.io import ROOT,write_json,write_rows,digest
from kgerp.metrics import binary
from kgerp.oracle import true_pair_count


def main(run_id):
    out=ROOT/'results'/run_id
    manifest=json.loads((out/'manifest.json').read_text())
    if manifest['status']!='completed':raise ValueError('Report only a completed run')
    frames=[];robust=[];scale=[];coverage=[];identity=[]
    for dataset in ['WWI','Olist','DataCo','BPI2019']:
        for seed in [17,29,43]:
            folder=out/dataset/f'seed_{seed}'
            frames.append(pd.read_csv(folder/'matching.csv'))
            robust.append(pd.read_csv(folder/'robustness.csv'))
            audit=json.loads((folder/'identity_audit.json').read_text());identity.append({'dataset':dataset,'seed':seed,**audit})
        scale.append(pd.read_csv(out/dataset/'e5/scale_trials.csv'))
        for study in ['E1','E2','E3','E4','E5']:coverage.append({'dataset':dataset,'experiment':study,'status':'executed','evidence':dataset+'/e1.json' if study=='E1' else dataset+'/e5/scale_trials.csv' if study=='E5' else dataset+'/seed_{17,29,43}/'+{'E2':'matching.csv','E3':'query_accuracy.csv','E4':'robustness.csv'}[study],
            'scope':'native plus controlled source views; WWI native Q1-Q8 saved separately' if dataset=='WWI' else 'natural repeat customers' if dataset=='Olist' else 'controlled source-view identity task, not dirty-identity ground truth'})
    wdc=pd.read_csv(out/'WDC/matching.csv');audit=pd.read_csv(out/'WDC/split_integrity.csv')
    for study in ['E1','E2','E3','E4','E5']:coverage.append({'dataset':'WDC','experiment':study,'status':'not_applicable' if study=='E3' else 'executed',
        'evidence':'WDC/'+{'E1':'e1.json','E2':'matching.csv','E3':'experiment_scope.json','E4':'robustness.csv','E5':'e5/scale_trials.csv'}[study],
        'scope':'No ERP transaction graph: ERP query/context tests would require invented relationships' if study=='E3' else 'official offer-only pairwise benchmark; 27 configurations, three seeds'})
    write_rows(out/'experiment_coverage.csv',coverage);write_rows(out/'identity_summary.csv',identity)
    typed=[]
    for seed in [17,29,43]:
        pred=json.loads((out/'WWI'/f'seed_{seed}/predictions.json').read_text())
        freeze=json.loads((out/'WWI'/f'seed_{seed}/freeze.json').read_text())
        records=json.loads((ROOT/f'data/generated/seed_{seed}/records.json').read_text())
        truth=json.loads((ROOT/f'data/gold/seed_{seed}/truth.json').read_text())
        for kind in ['Supplier','StockItem']:
            test=[r for r in records if r['split']=='test' and r['entity_type']==kind]
            total=true_pair_count(test,truth)
            for stratum in ['all','ambiguous']:
                subset=[r for r in pred if r['left_source_uid'].split(':')[1]==kind and (stratum=='all' or r['ambiguous'])]
                for name,threshold in freeze['accepts'].items():
                    y=np.array([r['label'] for r in subset],dtype=int);p=np.array([r[name] for r in subset])
                    typed.append({'dataset':'WWI','seed':seed,'entity_type':kind,'stratum':stratum,'model':name,'full_true_pair_universe':total,
                                  **binary(y,p>=threshold,p,total if stratum=='all' else None)})
    write_rows(out/'wwi_entity_type_metrics.csv',typed)
    matching=pd.concat(frames,ignore_index=True);robustness=pd.concat(robust,ignore_index=True);scaling=pd.concat(scale,ignore_index=True)
    matching.to_csv(out/'matching_all_graph_datasets.csv',index=False);robustness.to_csv(out/'robustness_all_graph_datasets.csv',index=False);scaling.to_csv(out/'scale_all_graph_datasets.csv',index=False)
    lines=['# Five-dataset non-human experiment report','',
       '**Exploratory scope expansion requested by the user.** The original WWI/WDC primary study and improvements-v2 remain unchanged. E6 human study is excluded. All applicable E1-E5 studies are executed; WDC ERP query/graph-context tests are not applicable because the supplied benchmark has no transaction relationships.','',
       '## Input roles and full-source fidelity','','| Dataset | Full-source scope | Matching truth and scope |','|---|---|---|']
    for dataset in ['WWI','Olist','DataCo','BPI2019']:
        e1=json.loads((out/dataset/'e1.json').read_text())
        nodes=e1.get('nodes',e1.get('actual_nodes'));edges=e1.get('edges',e1.get('actual_edges'))
        view=json.loads((out/dataset/'seed_17/view_manifest.json').read_text())
        scope=f"{nodes:,} nodes / {edges:,} edges"
        truth={'WWI':'240 original masters; preserved 49-group test split; richer typed transaction summaries',
               'Olist':f"{view['selected_groups']:,} eligible natural repeat customers; two distinct observed accounts with histories",
               'DataCo':f"{view['selected_groups']:,} stable-hash sampled customers from {view['eligible_source_groups']:,} eligible; controlled aliases, disjoint orders",
               'BPI2019':f"{view['selected_groups']:,} eligible anonymized vendors; controlled descriptive views, disjoint purchase cases"}[dataset]
        lines.append(f'| {dataset} | {scope} | {truth} |')
    w1=json.loads((out/'WDC/e1.json').read_text())
    lines.append(f"| WDC | {w1['unique_offers']:,} offers, {w1['official_split_files']} official split files | 27 pairwise configurations x 3 seeds; 4,500 official test pairs per cell |")
    lines+=['','Full-source Olist includes all nine files plus derived postal-prefix grouping nodes. DataCo retains all 180,519 row observations and audits normalization conflicts. BPI imports all 251,734 cases / 1,595,923 events and validates eight structural relationship sets. Native relational joins define E1 expected edge sets; lineage, dangling edges and repeated edge import are checked. WWI full-source E1 is reused from its preserved verified run.','',
       'DataCo publisher downloads returned HTTP 403. Its structured CSV is from the documented mirror with recorded SHA-256, 53 columns and 180,519 rows; publisher-byte equality is not claimed. BPI uses the publisher-linked legacy Figshare file and verifies the publisher MD5. Olist is official Kaggle version 2. Acquisition metadata is in `data/manifests/expanded_downloads.json`.','',
       '## Matching comparisons','','Seed means below are **conditional F1 on identical candidate pairs**, on the attribute-defined ambiguous stratum. End-to-end recall/F1, retained positives, candidate counts and subgroup denominators are saved separately. The retrieval cap can miss true pairs; a small conditional F1 gain cannot establish a full-pipeline gain.','',
       '| Method | WWI | Olist | DataCo | BPI2019 |','|---|---:|---:|---:|---:|']
    methods=matching['model'].drop_duplicates().tolist()
    for method in methods:
        values=[matching.loc[(matching.dataset==d)&(matching.model==method)&(matching.stratum=='ambiguous'),'f1'].mean() for d in ['WWI','Olist','DataCo','BPI2019']]
        lines.append('| '+method+' | '+' | '.join(f'{v:.4f}' for v in values)+' |')
    effects=[]
    lines+=['','| Dataset | Seed | B5 minus B2 F1 | Paired group-bootstrap 95% interval |','|---|---:|---:|---|']
    for dataset in ['WWI','Olist','DataCo','BPI2019']:
        for seed in [17,29,43]:
            intervals=pd.read_csv(out/dataset/f'seed_{seed}/paired_ci.csv')
            interval=intervals[intervals.contrast=='B5Graph_minus_B2Attributes'].iloc[0]
            selected=matching[(matching.dataset==dataset)&(matching.seed==seed)&(matching.stratum=='ambiguous')]
            delta=float(selected.loc[selected.model=='B5Graph','f1'].iloc[0]-selected.loc[selected.model=='B2Attributes','f1'].iloc[0])
            effects.append({'dataset':dataset,'seed':seed,'contrast':interval.contrast,'delta_f1':delta,'ci_low':interval.ci_low,'ci_high':interval.ci_high})
            lines.append(f'| {dataset} | {seed} | {delta:+.4f} | [{interval.ci_low:+.4f}, {interval.ci_high:+.4f}] |')
    write_rows(out/'context_effects.csv',effects)
    lines+=['','B5Graph versus B2Attributes is the controlled addition-of-context contrast. B4Flat uses independently SQL-extracted identical evidence and the same trained policy; exact scores agree within 1e-12. B2EmbeddingAugmented and B3Embedding are additional attribute references; they do not imply reproduction of literature fine-tuned neural models.','',
       'All learned families share the same five-choice LR/RF selection catalog. Training fits TF-IDF and estimators; disjoint validation canonical groups handle selection, sigmoid calibration and threshold policy. Cross-role pairs are omitted, with development-only prevalence correction. Models/policies are persisted before test-label metrics. This does not eliminate small-role uncertainty, conditional-selection bias or development-to-test prior shift.','',
       'WWI exposes authentic QuantityPerOuter compatibility to attribute baselines and uses it for stock-unit conversion in contextual purchasing quantities. Stock movements remain signed stock units. Supplier purchasing outers are kept separate from quantities that finance does not provide. BPI quantity is absent; its amount is the first-event anonymized cumulative net worth per purchase case, not an invoice-payment sum.','',
       'Olist has sparse histories and lacks customer names. Its descriptive attributes are city/state/postal prefix. Transaction context includes authentic product-category and seller-state paths. A minimal diagnostic fixture lacked sufficient shared support; that pre-training source-adapter amendment and the minimal fixture are retained. BPI vendor names/anonymized vendor IDs and company/source IDs are excluded from matching text. Its baseline descriptive fields are modal spend-area/document-type/classification descriptors from each disjoint case history, rather than literal vendor master attributes. Its comparison measures additional detailed history summaries beyond those coarse descriptors.','',
       'Adapter semantics matter: WWI retains its independently inferred master-anchor gate; the other graph datasets use support from two distinct projected category tags, which does not establish two independently matched entities. Prefix-only tags represent categorical missingness and can contribute to this heuristic support; available context does not guarantee two known informative categories. OneHop refers to the normalized entity/transaction/category projection, rather than native purchase/event graph depth, and uses local activity support. NoTransactions removes numerical/date/unit transaction summaries but retains category, activity and degree information. UnitWeights applies the frozen weighted model and policy to unit-weight inputs; this is an input-weight stress test, not a separately retrained ablation. The prior improvements-v2 study contains a separately trained unweighted WWI comparator.','',
       'The interrupted run was continued without refitting completed WWI models. Q1 SQL was optimized by deduplicating local entity/category incidence before joining identities; answer semantics are unchanged and parity is checked throughout. All WWI E5 trials were rerun with this query implementation. The prior six measurements and previous source are preserved under e5-before-Q1-optimization and code_amendments; manifest amendments record both code hashes.','',
       'A provenance review corrected Olist transaction projections that inherited their owner customer filename. Their native row origin is now the order-item file; date/status and category/seller fields retain their documented joined origins. All three Olist query-control evaluations and nine scaling trials were rerun, with previous measurements retained. Matching estimators and thresholds were preserved; every frozen score is independently replayed after the repair.','',
       '## WDC: all official pairwise configurations','','Mean test F1 across three training seeds. Original official train/test pairs are unchanged. Validation alone is internally divided by product group for development roles.','',
       '| Corner cases | Development | Unseen | B2 | B3 embedding | B2 + embedding | Overlapping train/test offers |','|---:|---|---:|---:|---:|---:|---:|']
    for (hardness,size,unseen),data in wdc[wdc.stratum=='official_all'].groupby(['corner_cases','development_size','unseen_percent'],sort=True):
        values=[data.loc[data.model==m,'f1'].mean() for m in ['B2Attributes','B3Embedding','B2EmbeddingAugmented']]
        overlap=int(audit.loc[(audit.corner_cases==hardness)&(audit.development_size==size)&(audit.unseen_percent==unseen),'train_test_offer_overlap'].iloc[0])
        lines.append(f'| {hardness}% | {size} | {unseen}% | '+ ' | '.join(f'{v:.4f}' for v in values)+f' | {overlap} |')
    lines+=['','Some supplied seen/half-seen files contain offer overlaps despite the benchmark documentation describing offer-disjoint splits. All official cells are still measured and explicitly flagged; contaminated cells are not presented as clean held-out evidence. `split_integrity.csv` includes train/validation and validation/test overlap as well. A separately reported offer-disjoint subset retains exact denominators without altering official pairs or fitting on test. No unseen configuration is selected from its test performance.','',
       '## Queries, robustness and calibration','',
       'Each graph dataset runs eight nonempty source-UID query tasks under unreconciled, predicted and oracle identity maps. SQL and graph answers agree exactly under each map. Source-local controls Q2/Q4 are distinguished from identity-sensitive category paths, totals, positive-value transaction links, identity pairs and provenance. Monetary aggregation uses cent-rounded source units and is not described as actual paid spend when the schema does not support it. WWI additionally preserves original native Q1-Q6 and executes planned native Q7/Q8 category/lineage tasks.','',
       'E4 evaluates clean/no-anchor conditions; attribute noise 30/50%; optional missingness 10/30/50%; relation removal 15/30%; typed misleading-edge rewiring 15/30%; visible-lineage missingness 30%. Methods include no-context/no-support/no-gate/unit-weight/no-transaction/one-hop ablations. All conditions use frozen pairs, estimators and policies. Perturbation logs never become correctness features. WDC has attribute stress only, plus all hardness/unseen/size variants.','',
       'Per-seed paired group-bootstrap intervals, sparse/dense and supported/unsupported subgroups, hard challenges, raw/calibrated Brier/log-loss/ECE, reliability bins and coverage-risk curves are saved. WWI supplier/stock-item breakdowns are in wwi_entity_type_metrics.csv, with only three independent held-out suppliers per seed. Calibration metrics on small imbalanced samples remain descriptive. Accepted false merges after one-to-one constraints are audited separately from pair-classifier false positives.','',
       '| Dataset | Accepted B5 links, three seeds | False merges |','|---|---:|---:|']
    ident=pd.DataFrame(identity)
    for d in ['WWI','Olist','DataCo','BPI2019']:
        data=ident[ident.dataset==d];lines.append(f'| {d} | {int(data.accepted_links.sum())} | {int(data.false_merges.sum())} |')
    lines+=['','## Three-point scaling','',
       'Every dataset has 1,000 / 10,000 / 100,000 source-view node or offer-record targets, three trials each. Graph targets include complete entity/transaction/category units and may slightly exceed the target. Independent groups grow only until the original test pool is exhausted; later aliases are replicas. Matching recall includes every same-original-entity cross-silo positive, including cross-replica pairs. Query workloads use the same per-copy oracle identities for both representations, independently of matching features.','',
       '| Dataset | Target | Actual nodes, trial 0 | Matching records | Candidates | Blocking recall |','|---|---:|---:|---:|---:|---:|']
    for _,r in scaling[scaling.trial==0].iterrows():lines.append(f'| {r.dataset} | {int(r.target_nodes):,} | {int(r.actual_nodes):,} | {int(r.matching_records):,} | {int(r.candidate_count):,} | {r.blocking_recall:.4f} |')
    ws=pd.read_csv(out/'WDC/e5/scale_trials.csv')
    for _,r in ws[ws.trial==0].iterrows():lines.append(f'| WDC | {int(r.target_source_records):,} | {int(r.source_records):,} offers | {int(r.source_records):,} | {int(r.candidate_count):,} | {r.blocking_recall:.4f} |')
    lines+=['','ETL/remapping, blocking, complete feature construction, scoring, graph preparation and SQL loading/indexing are timed separately. Graph/SQL batch and individual-template p50/p95 are measured after explicit warm-up. Process peak RSS is sampled every 50 ms, including incremental baseline; allocator retention affects the absolute process peak. Output cardinality is bounded by design, with measured recall loss. No subquadratic total-computation, universal graph-engine superiority, independent-enterprise diversity or production-readiness claim follows. WDC has offer indexing/scoring throughput only, because enterprise queries are inapplicable.','',
       '## Completion and limits','',
       'All applicable non-human E1-E5 experiments and planned optional embedding/one-hop/query extensions are executed. E6 is excluded by instruction. Twenty-seven WDC pairwise configurations are tested; multiclass WDC, literature-model reproductions, GNN training, production deployments, RDF/SHACL and a Neo4j UI are separate tasks rather than experiments in this plan.','',
       'Dataset-specific adapters, history sampling (up to 40 transactions per view for new graph datasets), controlled aliases for DataCo/BPI, Olist repeat-customer eligibility and source mirror limitations constrain generalization. A positive result is not necessary for completion. The original paper remains a preserved study draft; incorporating this expansion, publication rights, author review and submission are separate unfinished publication work.','',
       'See `experiment_coverage.csv`, `verification.json`, `config_snapshot.yaml`, the exact `code_snapshot/`, per-seed frozen models/decisions, source hashes and the acquired-model manifest. No human participants were recruited or contacted.','',
       '![Matching across four graph datasets](matching_comparison.png)','', '![All WDC configurations](wdc_configurations.png)','', '![Budgeted retrieval and recall](blocking_tradeoffs.png)']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    selected=['B2Attributes','B3Embedding','B2EmbeddingAugmented','B5Graph','NoGate','NoTransactions','OneHop']
    for ax,d in zip(axes.flat,['WWI','Olist','DataCo','BPI2019']):
        values=[matching.loc[(matching.dataset==d)&(matching.model==m)&(matching.stratum=='ambiguous'),'f1'].mean() for m in selected]
        ax.bar(range(len(selected)),values,color=['#335f8d','#8eacbc','#588b8b','#df793d','#b09a73','#8573a1','#939da5'])
        ax.set(xticks=range(len(selected)),xticklabels=['Attr','Emb','Attr+Emb','Graph','No gate','No numeric tx','1 hop'],ylim=(0,1),ylabel='Seed-mean conditional F1',title=d)
        ax.tick_params(axis='x',rotation=20)
    fig.suptitle('Exploratory matching on common ambiguous candidates');fig.tight_layout();fig.savefig(out/'matching_comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,5))
    order=[(h,s) for h in [20,50,80] for s in ['small','medium','large']]
    for ax,method in zip(axes,['B2Attributes','B3Embedding','B2EmbeddingAugmented']):
        image=np.array([[wdc.loc[(wdc.model==method)&(wdc.stratum=='official_all')&(wdc.corner_cases==h)&(wdc.development_size==s)&(wdc.unseen_percent==u),'f1'].mean() for u in [0,50,100]] for h,s in order])
        ax.imshow(image,vmin=0,vmax=1,cmap='Blues',aspect='auto')
        for i in range(9):
            for j in range(3):ax.text(j,i,f'{image[i,j]:.2f}',ha='center',va='center',color='white' if image[i,j]>.6 else 'black',fontsize=9)
        ax.set(xticks=[0,1,2],xticklabels=['0%','50%','100%'],yticks=range(9),yticklabels=[f'{h}% {s}' for h,s in order],xlabel='Unseen products',title=method)
    fig.suptitle('All 27 WDC official pairwise cells; overlaps audited separately');fig.tight_layout();fig.savefig(out/'wdc_configurations.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for d in ['WWI','Olist','DataCo','BPI2019']:
        data=scaling[(scaling.dataset==d)&(scaling.trial==0)]
        axes[0].plot(data.actual_nodes,data.candidate_count,'o-',label=d);axes[1].plot(data.actual_nodes,data.blocking_recall,'o-',label=d)
    axes[0].plot(ws.loc[ws.trial==0,'source_records'],ws.loc[ws.trial==0,'candidate_count'],'o-',label='WDC');axes[1].plot(ws.loc[ws.trial==0,'source_records'],ws.loc[ws.trial==0,'blocking_recall'],'o-',label='WDC')
    for ax in axes:ax.set_xscale('log');ax.set_xlabel('Generated view nodes / WDC offer records');ax.legend(fontsize=8)
    axes[0].set_yscale('log');axes[0].set_ylabel('Candidate output');axes[1].set(ylim=(0,1.05),ylabel='Full positive-universe blocking recall')
    fig.tight_layout();fig.savefig(out/'blocking_tradeoffs.png',dpi=180);plt.close(fig)
    # Stress curves use every seed and display seed range, not a sampling CI.
    selected=['B2Attributes','B2EmbeddingAugmented','B5Graph','NoGate']
    fig,axes=plt.subplots(2,4,figsize=(15,7),sharey=True)
    robustness=pd.concat(robust,ignore_index=True)
    for column,dataset in enumerate(['WWI','Olist','DataCo','BPI2019']):
        for row,axis in enumerate(['optional_missingness','edge_removal']):
            ax=axes[row,column]
            conditions=['clean']+[axis+'_'+str(level) for level in ([10,30,50] if row==0 else [15,30])]
            x=[0]+([10,30,50] if row==0 else [15,30])
            for name in selected:
                points=[robustness.loc[(robustness.dataset==dataset)&(robustness.model==name)&(robustness.scenario==condition),'f1'] for condition in conditions]
                ax.plot(x,[s.mean() for s in points],'o-',markersize=3,label=name)
                ax.fill_between(x,[s.min() for s in points],[s.max() for s in points],alpha=.10)
            ax.set(ylim=(0,1),xlabel=('Optional missingness (%)' if row==0 else 'Removed edges (%)'),title=dataset if row==0 else '')
            if column==0:ax.set_ylabel('Conditional F1; mean and seed range')
    axes[0,0].legend(fontsize=7);fig.suptitle('Frozen-policy robustness: three seeds per graph dataset');fig.tight_layout();fig.savefig(out/'robustness_curves.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(14,4),sharey=True)
    for ax,dataset in zip(axes,['WWI','Olist','DataCo','BPI2019']):
        ax.plot([0,1],[0,1],'--',color='#888888',linewidth=1)
        # Reliability bins contain their own denominators. Show seed 43 rather
        # than averaging bins whose occupied intervals differ across seeds.
        bins=pd.read_csv(out/dataset/'seed_43/reliability_bins.csv')
        for name in ['B2Attributes','B5Graph']:
            data=bins[(bins.model==name)&(bins['mode']=='calibrated')]
            ax.plot(data.mean_probability,data.positive_fraction,'o-',label=name)
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability',title=dataset)
    axes[0].set_ylabel('Observed positive fraction');axes[0].legend(fontsize=8)
    fig.suptitle('Calibration on common test candidates, seed 43; descriptive');fig.tight_layout();fig.savefig(out/'calibration_reliability.png',dpi=180);plt.close(fig)
    # Empirical candidate slopes describe bounded output, not total complexity.
    slope_rows=[]
    for dataset in ['WWI','Olist','DataCo','BPI2019','WDC']:
        data=ws[ws.trial==0] if dataset=='WDC' else scaling[(scaling.dataset==dataset)&(scaling.trial==0)]
        x=data.source_records if dataset=='WDC' else data.matching_records
        slope_rows.append({'dataset':dataset,'size_variable':'matching_source_records',
            'points':len(data),'candidate_loglog_slope':float(np.polyfit(np.log(x),np.log(data.candidate_count),1)[0]),
            'interpretation':'Output budget imposed; report blocking recall; not total-computation complexity or independent population diversity.'})
    write_rows(out/'candidate_slopes.csv',slope_rows)
    lines.extend(['','![Frozen-policy robustness](robustness_curves.png)','',
                  '![Descriptive calibration](calibration_reliability.png)','',
                  'Robustness plots show three-seed means and ranges, not confidence intervals. Reliability plots show seed 43 with exact per-bin denominators in the corresponding CSV. Candidate slopes use matching source-record counts and are available in `candidate_slopes.csv`; the imposed output cap prevents interpreting these as computational-complexity bounds.'])
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(out/'report_manifest.json',{'run_id':run_id,'generated_from_measured_artifacts':True,'files_sha256':{p.name:digest(p) for p in out.glob('*') if p.suffix in {'.md','.png','.csv'}},
         'human_study_excluded':True,'WDC_ERP_queries_not_applicable':True})
    print('Generated five-dataset report and measured figures: '+str(out),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',default='all-datasets-v1');main(parser.parse_args().run_id)
