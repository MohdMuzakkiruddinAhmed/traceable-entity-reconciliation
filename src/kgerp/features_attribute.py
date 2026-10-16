import numpy as np
from rapidfuzz.fuzz import ratio, token_sort_ratio
from sklearn.feature_extraction.text import TfidfVectorizer
from .blocking import normalize

NAMES = ['name_ratio','token_sort','token_jaccard','char_tfidf','category_agreement','brand_agreement',
         'description_ratio','price_similarity','currency_agreement','name_missing','category_missing','brand_missing','price_missing']

def numeric(value):
    try:
        number = float(str(value).replace(',',''))
        return number if np.isfinite(number) else None
    except (ValueError,TypeError):
        return None

class AttributeFeatures:
    def fit(self, records):
        self.vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=1, max_features=30000)
        self.vectorizer.fit([normalize(r.get('name')) or 'missing' for r in records])
        return self

    def transform(self, pairs, records):
        items = list(records.values())
        ids = {r['source_uid']:i for i,r in enumerate(items)}
        tf = self.vectorizer.transform([normalize(r.get('name')) for r in items])
        result = []
        for left, right in pairs:
            a,b = records[left],records[right]
            x,y = normalize(a.get('name')), normalize(b.get('name'))
            xs,ys = set(x.split()),set(y.split())
            price_a,price_b = numeric(a.get('price')),numeric(b.get('price'))
            currency_a,currency_b=normalize(a.get('currency')),normalize(b.get('currency'))
            comparable = (currency_a==currency_b) and price_a is not None and price_b is not None
            price = min(abs(price_a),abs(price_b))/max(abs(price_a),abs(price_b),1e-9) if comparable else 0
            def agree(key):
                u,v = normalize(a.get(key)),normalize(b.get(key))
                return float(bool(u and v) and u==v)
            result.append([ratio(x,y)/100, token_sort_ratio(x,y)/100, len(xs&ys)/max(1,len(xs|ys)),
                           float(tf[ids[left]].multiply(tf[ids[right]]).sum()), agree('category'),agree('brand'),
                           ratio(normalize(a.get('description')),normalize(b.get('description')))/100 if a.get('description') and b.get('description') else 0,
                           price,agree('currency'),float(not x or not y),
                           float(not a.get('category') or not b.get('category')),
                           float(not a.get('brand') or not b.get('brand')),float(price_a is None or price_b is None)])
        return np.asarray(result,dtype=float).reshape(-1,len(NAMES))
