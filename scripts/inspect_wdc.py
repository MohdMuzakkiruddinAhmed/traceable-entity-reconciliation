import gzip
import json
import zipfile
from pathlib import Path
z=zipfile.ZipFile(Path(__file__).resolve().parents[1]/'data/raw/wdc-products-50pair.zip')
def records(name):
    rows=[json.loads(x) for x in gzip.decompress(z.read(name)).decode().splitlines()]
    offers={}
    for p in rows:
        for side in ('left','right'):
            offers[str(p['id_'+side])] = {k[:-len(side)-1]:v for k,v in p.items() if k.endswith('_'+side)}
    return rows,offers
t,train=records('wdcproducts50cc50rnd000un_train_small.json.gz')
v,val=records('wdcproducts50cc50rnd000un_valid_small.json.gz')
print('Columns:',sorted(t[0]))
print('Train sample:',{k:v for k,v in t[0].items() if 'description' not in k})
for name in ['wdcproducts50cc50rnd000un_gs.json.gz','wdcproducts50cc50rnd050un_gs.json.gz','wdcproducts50cc50rnd100un_gs.json.gz']:
    rows,test=records(name)
    overlap=sorted(train.keys()&test.keys())
    identical=sum(train[k]==test[k] for k in overlap)
    print(name, len(rows),'train ID overlap:',len(overlap),'identical rows:',identical)
    for k in overlap[:2]:
        print(k, 'train:', {a:b for a,b in train[k].items() if a!='description'}, 'test:',{a:b for a,b in test[k].items() if a!='description'})
