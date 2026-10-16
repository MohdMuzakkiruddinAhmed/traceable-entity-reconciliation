import gzip
import json
import zipfile
import numpy as np
import pytest
from kgerp.features_attribute import AttributeFeatures,numeric
from kgerp.matchers import score
from kgerp import wdc

class Estimator:
    def __init__(self,positive):self.positive=positive
    def predict_proba(self,x):return np.tile([1-self.positive,self.positive],(len(x),1))

def test_no_anchor_gate_uses_identical_attribute_probability():
    models={'B2':{'estimator':Estimator(.2)},'context':{'estimator':Estimator(.9)}}
    ax=np.zeros((2,13));cx=np.zeros((2,6));cx[1,-1]=1
    predictions=score(models,ax,cx)
    assert predictions['B5'].tolist()==[.2,.9]
    assert predictions['B2'].tolist()==[.2,.2]
    assert predictions['B5_no_context'].tolist()==predictions['B2'].tolist()

def test_currency_and_missing_name_controls():
    a={'source_uid':'a','name':'tool','price':'10','currency':'USD'}
    b={'source_uid':'b','name':'tool','price':'10','currency':'EUR'}
    features=AttributeFeatures().fit([a,b]);x=features.transform([('a','b')],{'a':a,'b':b})
    assert x[0,7]==0  # no raw cross-currency numeric comparison
    assert numeric('NaN') is None and numeric('inf') is None
    a['name']=b['name']='';x=features.transform([('a','b')],{'a':a,'b':b})
    models={'B2':{'estimator':Estimator(.1)},'context':{'estimator':Estimator(.1)}}
    assert score(models,x,np.zeros((1,6)))['B0'][0]==0

def test_wdc_preserves_official_pairs_and_rejects_overlap(tmp_path,monkeypatch):
    monkeypatch.setattr(wdc,'ROOT',tmp_path)
    (tmp_path/'data/raw').mkdir(parents=True)
    config={'files':{},'expected_train':1,'expected_validation':1,'expected_test':1}
    def row(a,b):
        return {'id_left':a,'id_right':b,'title_left':'same tool','title_right':'same tool','label':1,'pair_id':f'{a}#{b}'}
    def archive(test):
        with zipfile.ZipFile(tmp_path/'data/raw/wdc-products-50pair.zip','w') as z:
            for split,pair in [('train',row(1,2)),('validation',row(3,4)),('test',test)]:
                member=split+'.json.gz';config['files'][split]=member
                z.writestr(member,gzip.compress((json.dumps(pair)+'\n').encode()))
    archive(row(5,6));loaded=wdc.import_splits(config)
    assert loaded['test'][0]['pair_id']=='5#6' and loaded['train'][0]['label']==1
    assert 'cluster_id' not in loaded['train'][0]['left']
    archive(row(1,6))
    with pytest.raises(ValueError,match='overlap'):wdc.import_splits(config)
