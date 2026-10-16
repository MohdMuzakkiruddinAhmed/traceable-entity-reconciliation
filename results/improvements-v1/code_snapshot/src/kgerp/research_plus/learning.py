"""Model selection, calibration and policy have disjoint development roles."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, brier_score_loss, log_loss
from ..matchers import threshold

def prior_weights(labels, target_prior):
    y=np.asarray(labels,dtype=int)
    observed=float(y.mean())
    if not 0<observed<1 or not 0<target_prior<1:
        raise ValueError('Prior correction requires both classes and a valid development prior')
    return np.where(y==1,target_prior/observed,(1-target_prior)/(1-observed))

def choose_threshold(probabilities, labels, target_prior=None):
    p=np.asarray(probabilities,dtype=float)
    weights=prior_weights(labels,target_prior) if target_prior is not None else None
    grid=sorted(set([.5,1.000001]+p.tolist()))
    return float(max(grid,key=lambda t:(f1_score(labels,p>=t,sample_weight=weights,zero_division=0),t)))

def select_model(train_x, train_y, selection_x, selection_y, config, seed):
    if len(set(train_y)) != 2 or len(set(selection_y)) != 2:
        raise ValueError('Both classes required for training/model selection')
    trials = []
    for c in config['logistic_C']:
        estimator = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight='balanced', max_iter=1500, random_state=seed))
        trials.append((f'logistic_C_{c}', estimator))
    for depth in config['forest_depths']:
        estimator = RandomForestClassifier(n_estimators=config['forest_trees'], max_depth=depth, min_samples_leaf=2,
                           class_weight='balanced', random_state=seed, n_jobs=2)
        trials.append((f'forest_depth_{depth}', estimator))
    summaries, fitted = [], []
    for name, estimator in trials:
        estimator.fit(train_x, train_y)
        p = estimator.predict_proba(selection_x)[:, 1]
        accept = choose_threshold(p, selection_y, config.get('target_prior'))
        weights=prior_weights(selection_y,config['target_prior']) if 'target_prior' in config else None
        f1 = float(f1_score(selection_y, p >= accept, sample_weight=weights, zero_division=0))
        summaries.append({'candidate':name, 'selection_f1':f1, 'selection_threshold':accept})
        fitted.append(estimator)
    chosen = max(range(len(trials)), key=lambda i:(summaries[i]['selection_f1'], -i))
    return fitted[chosen], {'chosen':trials[chosen][0], 'trials':summaries}

def logit(probabilities):
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1-1e-6)
    return np.log(p/(1-p)).reshape(-1,1)

class SigmoidCalibration:
    def fit(self, probabilities, labels, sample_weight=None):
        if len(set(labels)) != 2:
            raise ValueError('Calibration requires both classes on held-out calibration groups')
        self.estimator = LogisticRegression(C=1e6, max_iter=1000, random_state=0)
        self.estimator.fit(logit(probabilities), labels, sample_weight=sample_weight)
        return self

    def transform(self, probabilities):
        return self.estimator.predict_proba(logit(probabilities))[:,1]

def reliability(labels, probabilities, bins=10):
    y = np.asarray(labels, dtype=int); p = np.asarray(probabilities,dtype=float)
    if not len(y) or not np.isfinite(p).all() or (p<0).any() or (p>1).any():
        raise ValueError('Invalid probability audit inputs')
    membership = np.minimum((p*bins).astype(int), bins-1)
    rows = []
    for b in range(bins):
        mask = membership == b
        if mask.any():
            rows.append({'bin':b, 'count':int(mask.sum()), 'mean_probability':float(p[mask].mean()),
                         'positive_fraction':float(y[mask].mean())})
    ece = sum(r['count']*abs(r['mean_probability']-r['positive_fraction']) for r in rows) / len(y)
    return {'brier':float(brier_score_loss(y,p)), 'log_loss':float(log_loss(y,p,labels=[0,1])),
            'ece':ece, 'n_pairs':len(y), 'positive_prevalence':float(y.mean())}, rows

def coverage_risk(labels, probabilities, accept, reject):
    y=np.asarray(labels,dtype=bool);p=np.asarray(probabilities,dtype=float)
    rejected=p<=min(reject,accept);accepted=p>=accept
    decided=accepted|rejected
    errors=(accepted!=y)
    return {'coverage':float(decided.mean()), 'selective_risk':float(errors[decided].mean()) if decided.any() else None,
            'accepted_count':int(accepted.sum()), 'accepted_error_rate':float((~y[accepted]).mean()) if accepted.any() else None,
            'abstained_count':int((~decided).sum())}
