"""Evaluator-side group splitting and independently remapped source views."""
import copy
import json
import random
import uuid
from collections import defaultdict
import networkx as nx
from ..io import ROOT, write_json
from ..silo_generator import corrupt_name
from .sources import stable


def build_views(dataset,seed,config):
    if dataset=='WWI':
        from .wwi import build
        return build(seed)
    bases=json.loads((ROOT/'data/expanded'/dataset/'base_views.json').read_text(encoding='utf-8'))
    total=len(bases)
    bases.sort(key=lambda row:stable(str(config['split_seed'])+'|'+row['canonical']))
    if dataset!='Olist':bases=bases[:config['max_groups']]
    rng=random.Random(seed);records=[];graph=nx.MultiDiGraph();gold={}
    tag_nodes={}
    for i,base in enumerate(bases):
        split='train' if i<int(len(bases)*.6) else 'validation' if i<int(len(bases)*.8) else 'test'
        for side,view in zip(['A','B'],base['views']):
            uid=f'{side}:{base["kind"]}:{uuid.UUID(int=rng.getrandbits(128)).hex}'
            attrs={k:str(view.get(k) or '') for k in ['name','category','brand','description','price','currency']}
            if dataset!='Olist':
                attrs['name']=corrupt_name(attrs['name'],rng,{'abbreviation_probability':.30,'typo_probability':.35})
            for k in ['brand','category','description']:
                if rng.random()<.15:attrs[k]=''
            record={'source_uid':uid,'entity_type':base['kind'],'silo':side,'split':split,**attrs,
                    'source_table':view['source_table'],'source_batch':f'{dataset}-views-seed-{seed}','mapper_version':'expanded-v4'}
            records.append(record);graph.add_node(uid,**record)
            gold[uid]={'canonical':dataset+':'+base['canonical'],'split':split}
            # Hash sampling is fixed independently of labels and feature quality.
            transactions=sorted(view['transactions'],key=lambda t:stable(t['source_row']))[:40]
            for original in transactions:
                tid=f'{side}:Transaction:{uuid.UUID(int=rng.getrandbits(128)).hex}'
                attrs=copy.deepcopy(original);attrs.pop('source_row')
                attrs.update({'source_uid':tid,'entity_type':'Transaction','silo':side,'source_table':view['source_table'],
                    'source_batch':record['source_batch'],'mapper_version':'expanded-v4',
                    'amount_unit':'BPI_anonymized_EUR_initial_case_value' if dataset=='BPI2019' else dataset+'_line_amount',
                    'quantity_unit':'Olist_item_count' if dataset=='Olist' else 'DataCo_item_quantity' if dataset=='DataCo' else '',
                    'source_row_digest':stable(original['source_row'])})
                tags=attrs.pop('tags')
                graph.add_node(tid,**attrs);graph.add_edge(uid,tid,key='HAS_TRANSACTION',relation='HAS_TRANSACTION')
                for label in tags:
                    key=(side,label)
                    if key not in tag_nodes:
                        tag_nodes[key]=f'{side}:Category:{uuid.UUID(int=rng.getrandbits(128)).hex}'
                        graph.add_node(tag_nodes[key],entity_type='ContextCategory',label=label,silo=side,
                            source_table='visible_category_vocabulary',source_batch=record['source_batch'],mapper_version='expanded-v4')
                    graph.add_edge(tid,tag_nodes[key],key='HAS_CATEGORY',relation='HAS_CATEGORY')
    seen={s:set() for s in ['A','B']}
    for u,a in graph.nodes(data=True):
        if a['entity_type']=='Transaction':seen[a['silo']].add(a['source_row_digest'])
    if seen['A'] & seen['B']:raise ValueError('Copied transaction keys across silos')
    path=ROOT/f'data/gold/expanded/{dataset}/seed_{seed}'
    write_json(path/'truth.json',gold)
    return records,graph,gold,{'dataset':dataset,'eligible_source_groups':total,'selected_groups':len(bases),
         'selection':'stable source-key hash before alias perturbation; all eligible Olist repeats, cap for other datasets',
         'transaction_sample_limit_per_view':40,'cross_silo_copied_transaction_rows':0,
         'group_counts':{s:len({g['canonical'] for g in gold.values() if g['split']==s}) for s in ['train','validation','test']},
         'identity_task':'natural repeat-customer observations' if dataset=='Olist' else 'controlled alias/source-view evaluation'}


def perturb(records,graph,seed,attribute_noise=0.,optional_missingness=0.,edge_removal=0.,misleading_edge_rewire=0.,provenance_missingness=0.,no_anchors=False):
    rng=random.Random(seed);changed=copy.deepcopy(records);view=graph.copy()
    allowed={r['source_uid'] for r in changed}
    logs={'attributes_changed':0,'optional_fields_removed':0,'edges_removed':0,'edges_rewired':0,'lineage_removed':0}
    for record in changed:
        if rng.random()<attribute_noise:
            record['name']=' '.join(w[:max(1,len(w)//2)] for w in record['name'].split());logs['attributes_changed']+=1
        for key in ['category','brand','description','price']:
            if record.get(key) and rng.random()<optional_missingness:record[key]='';logs['optional_fields_removed']+=1
        if rng.random()<provenance_missingness:
            for key in ['source_table','source_batch','mapper_version']:record[key]=''
            logs['lineage_removed']+=1
        view.nodes[record['source_uid']].update(record)
    masters=defaultdict(list)
    for r in changed:masters[(r['silo'],r['entity_type'],r['split'])].append(r['source_uid'])
    for a,b,k in sorted(graph.edges(keys=True)):
        if k=='HAS_CATEGORY' and no_anchors:
            view.remove_edge(a,b,k)
        elif k=='HAS_TRANSACTION':
            if a not in allowed:continue
            if rng.random()<edge_removal:
                view.remove_edge(a,b,k);logs['edges_removed']+=1
            elif rng.random()<misleading_edge_rewire:
                node=graph.nodes[a];options=sorted(u for u in masters[(node['silo'],node['entity_type'],node['split'])] if u!=a)
                if options:
                    target=rng.choice(options);view.remove_edge(a,b,k);view.add_edge(target,b,key=k,relation=k);logs['edges_rewired']+=1
        elif k=='FROM_SUPPLIER' and a in allowed and b in allowed:
            if rng.random()<edge_removal:
                view.remove_edge(a,b,k);logs['edges_removed']+=1
            elif rng.random()<misleading_edge_rewire:
                source=graph.nodes[a]
                options=sorted(u for u in masters[(source['silo'],'Supplier',source['split'])] if u!=b)
                if options:
                    target=rng.choice(options);view.remove_edge(a,b,k);view.add_edge(a,target,key=k,relation=k);logs['edges_rewired']+=1
    return changed,view,logs
