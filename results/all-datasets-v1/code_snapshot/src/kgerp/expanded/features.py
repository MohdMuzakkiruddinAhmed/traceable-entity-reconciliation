"""Typed transaction/category context; no labels or original entity keys."""
from collections import Counter
from datetime import datetime
from functools import lru_cache
import math
import sqlite3
import numpy as np
from ..research_plus.features import lineage_quality

NAMES=['category_cosine','activity_cosine','degree_similarity','distinct_shared_categories','support_fraction',
       'date_mean_similarity','date_span_similarity','quantity_mean_similarity','quantity_sd_similarity',
       'amount_mean_similarity','amount_sd_similarity','date_available','quantity_unit_compatible',
       'amount_unit_compatible','contradictory_category','context_available']


@lru_cache(maxsize=100000)
def day(value):
    if not value:return None
    try:
        if len(value)>9 and value[4]=='-':return float(datetime.fromisoformat(value.replace('Z','+00:00')).date().toordinal())
        return float(datetime.strptime(value,'%m/%d/%Y %H:%M').date().toordinal())
    except ValueError:return None


def aggregate(rows,master_quality,weighted=True):
    categories=Counter();activities=Counter();dates=[];quantities=[];amounts=[];quantity_units=set();amount_units=set()
    for row in rows:
        q=min(master_quality,row['quality']) if weighted else 1.
        for label in row['tags']:categories[label]+=q
        for label,count in row.get('activities',{row['activity']:1}).items():
            if label:activities[label]+=q*count
        d=day(row['date'])
        if d is not None:dates.append((d,q))
        for field,values,units in [('quantity',quantities,quantity_units),('amount',amounts,amount_units)]:
            value=row[field]
            if value is not None and math.isfinite(float(value)):
                values.append((float(value),q));units.add(row[field+'_unit'])
    def moments(values):
        if not values or sum(q for v,q in values)<=0:return None
        v=np.array([v for v,q in values]);w=np.array([q for v,q in values]);mean=float(np.average(v,weights=w))
        return (mean,float(np.sqrt(np.average((v-mean)**2,weights=w))),float(v.max()-v.min()))
    return {'categories':categories,'activities':activities,'degree':sum(min(master_quality,r['quality']) if weighted else 1. for r in rows),
            'dates':moments(dates),'quantity':moments(quantities),'amount':moments(amounts),
            'quantity_units':quantity_units,'amount_units':amount_units}


def graph_profiles(graph,records,weighted=True,one_hop=False):
    output={}
    for record in records:
        uid=record['source_uid'];rows=[]
        for _,tx,key in graph.out_edges(uid,keys=True):
            if key!='HAS_TRANSACTION':continue
            attrs=graph.nodes[tx]
            if attrs['silo']!=record['silo']:raise ValueError('Cross-silo transaction key')
            tags=[] if one_hop else [graph.nodes[t]['label'] for _,t,k in graph.out_edges(tx,keys=True) if k=='HAS_CATEGORY']
            rows.append({**attrs,'tags':tags,'quality':lineage_quality(attrs)})
        output[uid]=aggregate(rows,lineage_quality(record),weighted)
    return output


def flat_profiles(graph,records,weighted=True,one_hop=False):
    # Extraction uses relational joins rather than graph traversal. The same
    # aggregate definition is applied to independently retrieved local rows.
    with sqlite3.connect(':memory:') as sql:
        sql.executescript('CREATE TABLE own(master TEXT,tx TEXT); CREATE TABLE transactions(uid TEXT PRIMARY KEY,attrs TEXT,silo TEXT); CREATE TABLE tags(tx TEXT,label TEXT);')
        import json
        sql.executemany('INSERT INTO transactions VALUES (?,?,?)',[(u,json.dumps(a),a['silo']) for u,a in graph.nodes(data=True) if a['entity_type']=='Transaction'])
        sql.executemany('INSERT INTO own VALUES (?,?)',[(a,b) for a,b,k in graph.edges(keys=True) if k=='HAS_TRANSACTION'])
        sql.executemany('INSERT INTO tags VALUES (?,?)',[(a,graph.nodes[b]['label']) for a,b,k in graph.edges(keys=True) if k=='HAS_CATEGORY'])
        sql.executescript('CREATE INDEX own_master ON own(master); CREATE INDEX tags_tx ON tags(tx);')
        output={}
        for r in records:
            rows=[]
            for uid,attrs,silo in sql.execute('SELECT t.uid,t.attrs,t.silo FROM own o JOIN transactions t ON o.tx=t.uid WHERE o.master=?',(r['source_uid'],)):
                if silo!=r['silo']:raise ValueError('Cross-silo context key in SQL')
                attrs=json.loads(attrs);tags=[] if one_hop else [x[0] for x in sql.execute('SELECT label FROM tags WHERE tx=?',(uid,))]
                rows.append({**attrs,'tags':tags,'quality':lineage_quality(attrs)})
            output[r['source_uid']]=aggregate(rows,lineage_quality(r),weighted)
    return output


def cosine(a,b):
    return sum(v*b.get(k,0) for k,v in a.items())/max(1e-12,math.sqrt(sum(v*v for v in a.values())*sum(v*v for v in b.values()))) if a and b else 0.


def similarity(a,b):
    return max(0.,1-abs(a-b)/max(1.,abs(a),abs(b)))


def context(pairs,profiles,min_support=2,support_features=True,transactions=True,one_hop=False):
    values=[];evidence=[]
    for left,right in pairs:
        a,b=profiles[left],profiles[right]
        ca,cb=(a['activities'],b['activities']) if one_hop else (a['categories'],b['categories'])
        shared={k for k in ca.keys()&cb.keys() if min(ca[k],cb[k])>0}
        support=sum(min(ca[k],cb[k]) for k in shared)
        available=len(shared)>=min_support and support>=1.
        quantity=bool(a['quantity'] and b['quantity'] and a['quantity_units']==b['quantity_units'] and a['quantity_units'] and '' not in a['quantity_units'])
        amount=bool(a['amount'] and b['amount'] and a['amount_units']==b['amount_units'] and a['amount_units'] and '' not in a['amount_units'])
        dates=bool(a['dates'] and b['dates'])
        date_mean=max(0.,1-abs(a['dates'][0]-b['dates'][0])/365.) if dates else 0.
        row=[cosine(ca,cb),cosine(a['activities'],b['activities']),similarity(a['degree'],b['degree']),len(shared),
             support/max(1.,sum(ca.values())+sum(cb.values())),date_mean,
             similarity(a['dates'][2],b['dates'][2]) if dates else 0.,
             similarity(a['quantity'][0],b['quantity'][0]) if quantity else 0.,similarity(a['quantity'][1],b['quantity'][1]) if quantity else 0.,
             similarity(a['amount'][0],b['amount'][0]) if amount else 0.,similarity(a['amount'][1],b['amount'][1]) if amount else 0.,
             float(dates),float(quantity),float(amount),float(bool(ca and cb and not shared)),float(available)]
        if not support_features:
            for index in [3,4,15]:row[index]=0.
        if not transactions:
            for index in range(5,14):row[index]=0.
        values.append(row);evidence.append({'context_available':available,'supporting_categories':sorted(shared),'weighted_support':float(support),
            'quantity_units_compatible':quantity,'amount_units_compatible':amount})
    return np.array(values,dtype=float).reshape(-1,len(NAMES)),evidence
