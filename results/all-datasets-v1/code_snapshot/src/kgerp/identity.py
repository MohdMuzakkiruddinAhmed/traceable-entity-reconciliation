import hashlib
from .blocking import pair_id

def resolve(pairs,scores,records,evidence,accept,reject,run_id,model_version):
    """Conservative one-to-one cross-silo links; never transitive union-find."""
    used,links,decisions = set(),[],[]
    for i in sorted(range(len(pairs)),key=lambda j:(-float(scores[j]),pairs[j])):
        left,right = pairs[i]
        same_type = records[left]['entity_type']==records[right]['entity_type']
        status = 'rejected' if scores[i]<=min(reject,accept) else 'abstained'
        if scores[i]>=accept:
            if not same_type:
                status='type_conflict'
            elif left in used or right in used:
                status='one_to_one_conflict'
            else:
                status='accepted';used.update((left,right));links.append((left,right))
        decisions.append({'candidate_id':pair_id(left,right),'left_source_uid':left,'right_source_uid':right,
                          'entity_type':records[left]['entity_type'],'score':float(scores[i]),
                          'threshold_policy':{'accept':accept,'reject':min(reject,accept)},'method_version':model_version,
                          'run_id':run_id,'status':status,'decision_trigger':'same_type_one_to_one_validation_threshold',
                          'supporting_feature_names':['name_similarity','char_tfidf','independent_typed_anchors'],
                          'evidence_source_uids':evidence[i]['neighbor_source_uids'],
                          'independent_anchor_pairs':evidence[i]['anchor_source_pairs']})
    mapping = {uid:uid for uid in records}
    for left,right in links:
        canonical = 'canonical:' + hashlib.sha256((left+'|'+right).encode()).hexdigest()[:24]
        mapping[left]=mapping[right]=canonical
    return mapping,decisions
