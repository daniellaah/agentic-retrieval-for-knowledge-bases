"""Family-paired P3 development decisions, raw provenance and unknown quality."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean
import sys

from arkb.evaluation.comparison import paired_family_bootstrap


def read(path):return json.loads(path.read_text())
def rows(path):return [json.loads(line) for line in path.read_text().splitlines()]


def compare(group, left, right, metric, families, stats):
    arms={left:defaultdict(list),right:defaultdict(list)}
    applicable=[]; excluded=[]
    for cid in sorted(families):
        selected=[r for r in group if r['case_id']==cid and r['arm'] in arms]
        values=[metric(r) for r in selected]
        if not values or any(v is None for v in values):
            excluded.append(cid);continue
        for row,value in zip(selected,values):arms[row['arm']][cid].append(value)
        applicable.append(cid)
    if not applicable or len({families[c] for c in applicable})<2:
        return {'status':'insufficient_defined_families','excluded_cases':excluded}
    result=paired_family_bootstrap(arms[left],arms[right],{c:families[c] for c in applicable},
                                   seed=stats['seed'],resamples=stats['resamples'])
    return {**result,'left':left,'right':right,'excluded_cases':excluded}


def main():
    bundle,output=map(Path,sys.argv[1:]);metadata=read(bundle/'experiment.json')
    if metadata['status']!='completed':raise ValueError('Cannot summarize incomplete run.')
    checksums=read(bundle/'checksums.json')
    for name in ('protocol.json','component-results.jsonl','agent-results.jsonl','candidate-inputs.jsonl','dataset/queries.jsonl','gate.json'):
        if hashlib.sha256((bundle/name).read_bytes()).hexdigest()!=checksums[name]:
            raise ValueError('Summary input checksum mismatch: '+name)
    protocol=read(bundle/'protocol.json');stats=protocol['statistics']
    component=rows(bundle/'component-results.jsonl');agent=rows(bundle/'agent-results.jsonl')
    cases={c['id']:c for c in rows(bundle/'dataset/queries.jsonl')}
    from arkb.evaluation.controlled import stopping_order
    selected={cid for cid,c in cases.items() if c['task_type'] not in protocol['component']['excluded_tasks']}
    expected={(cid,arm,0) for cid in selected for arm in protocol['component']['arms']}
    if (len(component)!=len(expected) or {(r['case_id'],r['arm'],r['trial']) for r in component}!=expected
            or [(r['case_id'],r['arm'],r['trial']) for r in agent]!=stopping_order(protocol['stopping'])
            or not read(bundle/'gate.json')['experiment_valid']):
        raise ValueError('Summary requires complete registered matrices and a valid run.')
    families={r['case_id']:cases[r['case_id']]['intent_family_id'] for r in component}
    retrieval={}
    for arm in protocol['component']['arms']:
        group=[r for r in component if r['arm']==arm]
        def finite_mean(fn):
            values=[fn(r) for r in group]; known=[v for v in values if v is not None]
            return {'case_mean':mean(known) if known else None,'defined_cases':len(known)}
        retrieval[arm]={'queries':len(group),'families':len(set(families.values())),
            'stage_coverage':{stage:finite_mean(lambda r:r['stage_evidence'][stage]['evidence_coverage'])
                for stage in ('candidate_union','fusion_pool','returned')},
            'source_ndcg_from_top10_chunks':finite_mean(lambda r:r['metrics']['source_metrics_from_top_k_chunks']['ndcg_exp@10']),
            'assessed_at10':finite_mean(lambda r:r['metrics']['source_metrics_from_top_k_chunks']['assessed@10']),
            'mean_candidate_union_count':mean(r['candidate_union_count'] for r in group),
            'mean_fusion_pool_count':mean(r['fusion_pool_count'] for r in group)}
    candidate_comparisons=[]
    for left,right in [protocol['component']['primary_comparison'],*protocol['component']['secondary_comparisons']]:
        evidence=compare(component,left,right,lambda r:r['stage_evidence']['returned']['evidence_coverage'],families,stats)
        target=protocol['component']['minimum_effect'];low,high=evidence['delta_ci95'];delta=evidence['delta_right_minus_left']
        if low>0 and delta>=target: decision='continue_dev_validation_required'
        elif high<target:decision='reject_for_this_dev_round'
        else:decision='evidence_insufficient'
        candidate_comparisons.append({'left':left,'right':right,'returned_evidence':evidence,
            'union_evidence':compare(component,left,right,lambda r:r['stage_evidence']['candidate_union']['evidence_coverage'],families,stats),
            'pool_evidence':compare(component,left,right,lambda r:r['stage_evidence']['fusion_pool']['evidence_coverage'],families,stats),
            'decision':decision,'production_adoption':False})
    stop_families={r['case_id']:cases[r['case_id']]['intent_family_id'] for r in agent}
    metric_fns={'normal_final_fraction':lambda r:float(r['stop_reason']=='final'),
        'evidence_delivery_proxy':lambda r:(None if r['metrics']['evidence']['evidence_coverage'] is None else
            r['metrics']['evidence']['evidence_coverage'] if r['stop_reason']=='final' and not r['constraint_failures'] else 0.),
        'executed_tools':lambda r:sum(e['executed'] for e in r['observation']['tools']),
        'provider_output_tokens':lambda r:r['observation']['usage']['eval_count']['total'],
        'calls_after_label_threshold':lambda r:r['stopping_diagnostics']['executed_calls_after_label_threshold'],
        'elapsed_ms':lambda r:r['elapsed_ms']}
    stopping={}
    for arm in protocol['stopping']['arms']:
        group=[r for r in agent if r['arm']==arm]
        stopping[arm]={'rows':len(group),'queries':len({r['case_id'] for r in group}),
            'families':len({stop_families[r['case_id']] for r in group}),
            'stops':dict(Counter(r['stop_reason'] for r in group)),
            'errors':sum(r['error'] is not None for r in group),
            'reminder_exposed_runs':sum(r['stopping_diagnostics']['reminders_submitted']>0 for r in group),
            'grounded_task_success':None,'reviewed_outputs':0,
            'metrics':{name:{'trial_mean':mean(known) if known else None,'defined_trials':len(known)}
                for name,fn in metric_fns.items() for known in [[fn(r) for r in group if fn(r) is not None]]}}
    left,right=protocol['stopping']['primary_comparison']
    stop_comparisons={name:compare(agent,left,right,fn,stop_families,stats) for name,fn in metric_fns.items()}
    raw=rows(bundle/'candidate-inputs.jsonl')
    result={'bundle':bundle.name,'protocol_sha256':hashlib.sha256((bundle/'protocol.json').read_bytes()).hexdigest(),
        'gate':read(bundle/'gate.json'),'component_arms':retrieval,'component_comparisons':candidate_comparisons,
        'stopping_arms':stopping,'stopping_comparisons':stop_comparisons,
        'stopping_decision':{'decision':'evidence_insufficient','production_adoption':False,
            'reason':'Independent reviewed quality is missing; diagnostic final/coverage cannot establish noninferiority.'},
        'component_timing':{'rerank_pairs':sum(len(r['rerank']['scores']) for r in raw),
            'rerank_total_ms':sum(r['rerank']['elapsed_ms'] for r in raw),'policy':'Shared frozen prefix derivation has no independent per-arm retrieval latency; scorer time is measured separately, uncontrolled cache/load.'},
        'facet_losses':[{'case_id':r['case_id'],'family':r['family'],'arm':r['arm'],'lost_facets':r['lost_facets']}
                        for r in component if any(r['lost_facets'].values())],
        'stop_per_trial':[{'case_id':r['case_id'],'family':stop_families[r['case_id']],'arm':r['arm'],'trial':r['trial'],
            'stop_reason':r['stop_reason'],'error':r['error'],**r['stopping_diagnostics']} for r in agent],
        'raw_sha256':{name:hashlib.sha256((bundle/name).read_bytes()).hexdigest() for name in ('candidate-inputs.jsonl','component-results.jsonl','agent-results.jsonl')},
        'limits':['Known provisional dev data; neither normal final nor evidence coverage measures answer correctness.',
            'Case means and family means use different weighting; repeated trials are not independent samples.',
            'Secondary percentile intervals are exploratory and unadjusted for multiple comparisons.',
            'Component exact retrieval prefixes do not measure ANN depth effects or real per-arm cost.',
            'Stopping changes only reminder insertion; default production behavior remains enabled.',
            'Evidence_delivery_proxy is coverage times valid final delivery, not grounded task success.']}
    with output.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps({'component_comparisons':[{k:v for k,v in c.items() if k in ('left','right','decision')} for c in candidate_comparisons],
                     'stopping':stopping},ensure_ascii=False))


if __name__=='__main__':main()
