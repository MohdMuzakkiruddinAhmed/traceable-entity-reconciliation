import numpy as np
from sklearn.metrics import average_precision_score

def binary(y,pred,scores=None,total_positives=None):
    y,pred = np.asarray(y,dtype=int),np.asarray(pred,dtype=bool)
    tp,fp,fn,tn = (int(((y==1)&pred).sum()),int(((y==0)&pred).sum()),
                  int(((y==1)&~pred).sum()),int(((y==0)&~pred).sum()))
    precision,recall = tp/max(1,tp+fp),tp/max(1,tp+fn)
    f1 = 2*tp/max(1,2*tp+fp+fn)
    out = {'n_pairs':len(y),'positives':int(y.sum()),'prevalence':float(y.mean()) if len(y) else 0,
           'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':precision,'recall':recall,'f1':f1}
    if scores is not None:
        out['average_precision']=float(average_precision_score(y,scores)) if y.sum() else 0
    if total_positives is not None:
        out['blocking_recall']=int(y.sum())/max(1,total_positives)
        out['end_to_end_recall']=tp/max(1,total_positives)
        out['end_to_end_f1']=2*tp/max(1,tp+fp+total_positives)
    return out

def set_metrics(expected,actual):
    tp=len(expected&actual)
    return {'expected_answers':len(expected),'actual_answers':len(actual),'tp':tp,
            'precision':tp/len(actual) if actual else float(not expected),
            'recall':tp/len(expected) if expected else float(not actual),
            'f1':2*tp/(len(actual)+len(expected)) if actual or expected else 1.0}

def paired_bootstrap(y,a,b,pairs,gold,seed,repetitions=1000):
    """Entity multiplicities applied to both endpoints; correlated pair resampling."""
    groups = sorted({gold[u]['canonical'] for pair in pairs for u in pair})
    if not groups or not len(y): return None,None
    lookup = {g:i for i,g in enumerate(groups)}
    left = np.array([lookup[gold[u]['canonical']] for u,v in pairs])
    right = np.array([lookup[gold[v]['canonical']] for u,v in pairs])
    rng = np.random.default_rng(seed)
    differences=[]
    def f1(pred,weights):
        tp = weights[(y==1)&pred].sum()
        fp = weights[(y==0)&pred].sum()
        fn = weights[(y==1)&~pred].sum()
        return 2*tp/max(1,2*tp+fp+fn)
    for _ in range(repetitions):
        counts = np.bincount(rng.integers(0,len(groups),len(groups)),minlength=len(groups))
        weights = np.where(left==right,counts[left],counts[left]*counts[right])
        differences.append(f1(b,weights)-f1(a,weights))
    return tuple(map(float,np.quantile(differences,[.025,.975])))
