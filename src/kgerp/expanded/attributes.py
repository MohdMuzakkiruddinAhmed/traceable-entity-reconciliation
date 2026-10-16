"""Batch sparse cosine evaluation with the preserved 20-feature definition."""
import re
import numpy as np
from rapidfuzz.fuzz import ratio,token_sort_ratio
from ..blocking import normalize
from ..features_attribute import numeric
from ..research_plus.features import EnhancedAttributes,explicit_size,jaccard


class FastAttributes(EnhancedAttributes):
    def transform(self,pairs,records):
        items=list(records.values());ids={r['source_uid']:i for i,r in enumerate(items)}
        normalized=[normalize(r.get('name')) for r in items]
        tf=self.base.vectorizer.transform(normalized)
        left=np.array([ids[a] for a,b in pairs],dtype=int);right=np.array([ids[b] for a,b in pairs],dtype=int)
        cosines=np.empty(len(pairs),dtype=float)
        for start in range(0,len(pairs),32768):
            stop=start+32768
            cosines[start:stop]=np.asarray(tf[left[start:stop]].multiply(tf[right[start:stop]]).sum(axis=1)).ravel()
        result=[]
        for i,(a,b) in enumerate(pairs):
            x,y=normalized[ids[a]],normalized[ids[b]];ra,rb=records[a],records[b]
            xs,ys=set(x.split()),set(y.split());pa,pb=numeric(ra.get('price')),numeric(rb.get('price'))
            ca,cb=normalize(ra.get('currency')),normalize(rb.get('currency'))
            comparable=ca==cb and pa is not None and pb is not None
            price=min(abs(pa),abs(pb))/max(abs(pa),abs(pb),1e-9) if comparable else 0.
            def agree(key):
                av,bv=normalize(ra.get(key)),normalize(rb.get(key));return float(bool(av and bv) and av==bv)
            nx,ny=set(re.findall(r'\d+(?:\.\d+)?',x)),set(re.findall(r'\d+(?:\.\d+)?',y))
            ax={t for t in x.split() if any(c.isdigit() for c in t)};ay={t for t in y.split() if any(c.isdigit() for c in t)}
            sx,sy=explicit_size(x),explicit_size(y);both=bool(sx and sy)
            result.append([ratio(x,y)/100,token_sort_ratio(x,y)/100,len(xs&ys)/max(1,len(xs|ys)),cosines[i],agree('category'),agree('brand'),
                ratio(normalize(ra.get('description')),normalize(rb.get('description')))/100 if ra.get('description') and rb.get('description') else 0.,
                price,agree('currency'),float(not x or not y),float(not ra.get('category') or not rb.get('category')),
                float(not ra.get('brand') or not rb.get('brand')),float(pa is None or pb is None),jaccard(nx,ny),jaccard(ax,ay),
                float(both and sx==sy),float(both and sx!=sy),float(not both),min(len(x),len(y))/max(1,len(x),len(y)),float(not nx or not ny)])
        return np.array(result,dtype=float).reshape(-1,20)
