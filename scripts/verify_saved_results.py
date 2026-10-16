"""Recompute released matching metrics without raw data or model fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support

ROOT=Path(__file__).resolve().parents[1]

def verify(run):
    checks=[]
    def check(name,condition):
        if not condition:raise AssertionError(name)
        checks.append(name)
    def compare(name,records,model,threshold,row,total=None):
        y=np.array([x['label'] for x in records],dtype=int)
        p=np.array([x[model] for x in records],dtype=float)
        check(name+'_finite',bool(np.isfinite(p).all()))
        if len(y):
            tn,fp,fn,tp=confusion_matrix(y,p>=threshold,labels=[0,1]).ravel()
            precision,recall,f1,_=precision_recall_fscore_support(y,p>=threshold,average='binary',zero_division=0)
            actual={'n_pairs':len(y),'positives':int(y.sum()),'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':precision,'recall':recall,'f1':f1,
                    'average_precision':average_precision_score(y,p) if y.sum() else 0.}
        else:actual={k:0 for k in ['n_pairs','positives','tp','fp','fn','tn','precision','recall','f1','average_precision']}
        if total is not None:
            actual.update(blocking_recall=int(y.sum())/total,end_to_end_recall=actual['tp']/total,
                end_to_end_f1=2*actual['tp']/(actual['tp']+actual['fp']+total))
        check(name+'_metrics',all(abs(float(row[k])-float(v))<1e-12 for k,v in actual.items()))
    graph_rows=0
    for d in ['WWI','Olist','DataCo','BPI2019']:
      for seed in [17,29,43]:
        folder=run/d/f'seed_{seed}';pred=json.loads((folder/'predictions.json').read_text())
        freeze=json.loads((folder/'freeze.json').read_text())
        for row in pd.read_csv(folder/'matching.csv').to_dict('records'):
            selected=pred if row['stratum']=='all' else [x for x in pred if x['ambiguous']]
            compare(f'{d}_{seed}_{row["model"]}_{row["stratum"]}',selected,row['model'],freeze['accepts'][row['model']],row,
                    int(row['independent_groups']) if row['stratum']=='all' else None);graph_rows+=1
        dec=json.loads((folder/'decisions.json').read_text());accepted=[x for x in dec if x['status']=='accepted']
        labels={x['candidate_id']:x['label'] for x in pred};audit=json.loads((folder/'identity_audit.json').read_text())
        check(f'{d}_{seed}_accepted_audit',len(accepted)==audit['accepted_links'] and sum(labels[x['candidate_id']]==0 for x in accepted)==audit['false_merges'])
        check(f'{d}_{seed}_one_to_one',len({x['left_source_uid'] for x in accepted})==len(accepted)==len({x['right_source_uid'] for x in accepted}))
        if (folder/'frozen_models.joblib').exists():
            check(f'{d}_{seed}_fitted_model_checksum',hashlib.file_digest((folder/'frozen_models.joblib').open('rb'),'sha256').hexdigest()==freeze['model_sha256'])
    wdc_rows=0;wdc=pd.read_csv(run/'WDC/matching.csv')
    for hardness in [20,50,80]:
      for size in ['small','medium','large']:
       for seed in [17,29,43]:
        folder=run/'WDC'/f'{hardness}cc_{size}'/f'seed_{seed}';freeze=json.loads((folder/'freeze.json').read_text())
        for unseen in [0,50,100]:
            pred=json.loads((folder/f'predictions_{unseen:03d}un.json').read_text())
            rows=wdc[(wdc.corner_cases==hardness)&(wdc.development_size==size)&(wdc.seed==seed)&(wdc.unseen_percent==unseen)]
            for row in rows[rows.stratum.isin(['official_all','offer_disjoint_subset'])].to_dict('records'):
                selected=pred if row['stratum']=='official_all' else [x for x in pred if x['offer_disjoint']]
                compare(f'WDC_{hardness}_{size}_{seed}_{unseen}_{row["model"]}_{row["stratum"]}',selected,row['model'],freeze['accepts'][row['model']],row);wdc_rows+=1
        if (folder/'frozen_models.joblib').exists():
            check(f'WDC_{hardness}_{size}_{seed}_fitted_model_checksum',hashlib.file_digest((folder/'frozen_models.joblib').open('rb'),'sha256').hexdigest()==freeze['model_sha256'])
    check('312_graph_matching_rows',graph_rows==312)
    check('810_WDC_official_and_disjoint_rows',wdc_rows==810)
    return {'status':'passed','checks_passed':len(checks),'graph_matching_rows':graph_rows,'wdc_matching_rows':wdc_rows,
       'tolerance':1e-12,'new_model_fitting':False,'raw_input_or_model_score_replay':False,
       'scope':'Saved-score metrics, final accepted-link audits, one-to-one constraints and available fitted-model checksums. Hard-challenge flags require official raw WDC; robustness scores were not recomputed.',
       'checks':checks}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',default='all-datasets-v1');parser.add_argument('--output',default='verification/saved-results.json')
    args=parser.parse_args();report=verify(ROOT/'results'/args.run_id)
    path=ROOT/args.output;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))

if __name__=='__main__':main()
