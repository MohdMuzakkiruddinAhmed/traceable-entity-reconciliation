import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

def threshold(scores, labels, grid=None):
    grid = list(grid) if grid is not None else sorted(set([0.5,1.000001]+list(np.asarray(scores,dtype=float))))
    # Highest threshold breaks ties conservatively.
    return float(max(grid,key=lambda t:(f1_score(labels,np.asarray(scores)>=t,zero_division=0),t)))

def fit_learned(train_x, train_y, val_x, val_y, c_values):
    if len(set(train_y)) < 2 or not len(val_y):
        raise ValueError('Insufficient independent development groups/classes')
    options = []
    for c in c_values:
        model = make_pipeline(StandardScaler(),LogisticRegression(C=c,class_weight='balanced',max_iter=1000,random_state=0))
        model.fit(train_x,train_y)
        score = model.predict_proba(val_x)[:,1]
        accept = threshold(score,val_y)
        options.append((f1_score(val_y,score>=accept,zero_division=0),-c,model,accept))
    best = max(options,key=lambda x:(x[0],x[1]))
    return {'estimator':best[2],'accept':best[3],'validation_f1':float(best[0]),'C':float(-best[1])}

def train(train, validation, config):
    ax,cy,y = train
    vx,vc,vy = validation
    b2 = fit_learned(ax,y,vx,vy,config['logistic_C'])
    context = fit_learned(np.column_stack([ax,cy]),y,np.column_stack([vx,vc]),vy,config['logistic_C'])
    # B4 and ungated B5 share the learner to isolate extraction and gate effects.
    vs = context['estimator'].predict_proba(np.column_stack([vx,vc]))[:,1]
    fallback = b2['estimator'].predict_proba(vx)[:,1]
    gated = np.where(vc[:,-1]>0,vs,fallback)
    return {'B2':b2,'context':context,'B1_accept':threshold(vx[:,1],vy,config['fuzzy_threshold_grid']),
            'B5_accept':threshold(gated,vy),'reject':config['rejection_threshold']}

def score(models, ax, context):
    b2 = models['B2']['estimator'].predict_proba(ax)[:,1]
    raw = models['context']['estimator'].predict_proba(np.column_stack([ax,context]))[:,1]
    return {'B0':((ax[:,0]==1)&(ax[:,9]==0)).astype(float),'B1':ax[:,1],'B2':b2,'B4':raw,
            'B5':np.where(context[:,-1]>0,raw,b2),'B5_no_context':b2,'B5_ungated':raw}

def acceptance(models, model):
    if model=='B0': return 1.0
    if model=='B1': return models['B1_accept']
    if model in {'B2','B5_no_context'}: return models['B2']['accept']
    if model=='B5': return models['B5_accept']
    return models['context']['accept']
