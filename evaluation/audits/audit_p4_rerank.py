"""Measure actual retained reranker input truncation without loading model weights."""
import argparse
from collections import Counter
import json
from pathlib import Path
from arkb.evaluation.external import digest,load_external,read_jsonl,verify_checksums,write_json


def stage_attribution(data,rankings,candidates,token_rows):
    mapping=data.source_map();by_query={q:{r['arm']:r for r in rankings if r['qid']==q} for q in data.queries}
    zero_body=Counter(r['qid'] for r in token_rows if r['retained_body_tokens']==0)
    input_counts=Counter(r['qid'] for r in token_rows);rows=[]
    for raw in candidates:
        qid=raw['qid'];arms=by_query[qid];gold={d for d,v in data.qrels[qid].items() if v>0}
        legs={k:list(dict.fromkeys(mapping[h['source']] for h in hs)) for k,hs in raw['legs'].items()}
        union=set().union(*map(set,legs.values()))
        stages={'bm25_leg':legs.get('bm25',[]),'semantic_leg':legs.get('semantic',[]),
            'leg_union':union,'hybrid100':arms['hybrid']['ranking'],
            'pool20':[mapping[h['source']] for h in raw['rerank_inputs']],
            'hybrid10':arms['hybrid']['ranking'][:10],'reranked10':arms['hybrid-rerank20']['ranking'][:10]}
        values={}
        for name,ids in stages.items():
            seen=set(ids);aspects=data.aspects.get(qid,[]);total=sum(a['weight'] for a in aspects)
            values[name]={'unique_sources':len(seen),'known_positive_sources':sorted(gold & seen),
                'source_recall':len(gold & seen)/len(gold),
                'weighted_aspect_coverage':sum(a['weight'] for a in aspects if seen.intersection(a['supporting_docs']))/total if total else None}
        recall=lambda stage:values[stage]['source_recall']
        labels=[]
        if not gold & union:labels.append('no_known_positive_in_leg_union')
        if recall('hybrid100')<recall('leg_union'):labels.append('known_positive_lost_at_fusion100')
        if recall('pool20')<recall('hybrid100'):labels.append('known_positive_outside_pool20')
        if recall('reranked10')<recall('pool20'):labels.append('pool_positive_outside_reranked10')
        if recall('reranked10')<recall('hybrid10'):labels.append('top10_source_recall_degraded_by_reranking')
        if input_counts[qid] and zero_body[qid]==input_counts[qid]:labels.append('all_reranked_inputs_have_zero_body')
        before=values['hybrid10']['weighted_aspect_coverage'];after=values['reranked10']['weighted_aspect_coverage']
        if before is not None and after<before:labels.append('top10_aspect_coverage_degraded_by_reranking')
        if any(r['error'] is not None for r in arms.values()):labels.append('retained_execution_error')
        rows.append({'qid':qid,'stages':values,'categories':labels,
            'rerank_minus_hybrid_ndcg10':arms['hybrid-rerank20']['metrics']['ndcg@10']-arms['hybrid']['metrics']['ndcg@10'],
            'zero_body_inputs':zero_body[qid],'reranked_input_count':input_counts[qid]})
    summary={}
    for stage in rows[0]['stages']:
        summary[stage]={}
        for metric in ('unique_sources','source_recall','weighted_aspect_coverage'):
            defined=[r['stages'][stage][metric] for r in rows if r['stages'][stage][metric] is not None]
            summary[stage][metric]={'mean':sum(defined)/len(defined) if defined else None,'defined_queries':len(defined)}
    return {'dataset':data.name,'queries':len(rows),'mean_stage_coverage':summary,
        'nonexclusive_category_counts':dict(Counter(c for r in rows for c in r['categories'])),
        'rows':rows,'scope':'Post-hoc source/annotation-level attribution on frozen candidates. Categories overlap; a positive source is not proof that the retrieved span supports the answer. Pool-to-top10 loss also reflects the smaller cutoff, not solely model quality.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();verify_checksums(a.run);data=load_external(a.dataset)
    if json.loads((a.run/'experiment.json').read_text())['status']!='completed':raise ValueError('Incomplete run.')
    if digest(a.dataset/'manifest.json')!=json.loads((a.run/'protocol.json').read_text())['dataset_manifest_sha256']:
        raise ValueError('Dataset mismatch.')
    from transformers import AutoTokenizer
    import arkb.retrieval.qwen_rerank as q
    frozen=a.run/'measured-source/src/arkb/retrieval/qwen_rerank.py'
    if digest(frozen)!=digest(Path(q.__file__)):raise ValueError('Run audit with the measured reranker implementation.')
    tokenizer=AutoTokenizer.from_pretrained(q.QWEN_MODEL,revision=q.QWEN_REVISION,cache_dir='.arkb/models',
        padding_side='left',local_files_only=True,trust_remote_code=False)
    reserved=len(tokenizer.encode(q.PREFIX,add_special_tokens=False))+len(tokenizer.encode(q.SUFFIX,add_special_tokens=False))
    cap=512-reserved;rows=[];candidates=read_jsonl(a.run/'candidates.jsonl')
    for raw in candidates:
        query=data.queries[raw['qid']]
        for hit in raw['rerank_inputs']:
            prefix=f'<Instruct>: {q.INSTRUCTION}\n<Query>: {query}\n<Document>: '+(hit['metadata'].get('title') or '')+'\n\n'
            pair=prefix+hit['content']
            full=tokenizer(pair,padding=False,truncation=False,return_offsets_mapping=True)
            retained=tokenizer(pair,padding=False,truncation='longest_first',max_length=cap,return_offsets_mapping=True)
            body=lambda offsets:sum(end>len(prefix) for start,end in offsets)
            rows.append({'qid':raw['qid'],'source':hit['source'],'full_main_tokens':len(full['input_ids']),
                'retained_main_tokens':len(retained['input_ids']),'truncated':len(full['input_ids'])>cap,
                'body_tokens':body(full['offset_mapping']),'retained_body_tokens':body(retained['offset_mapping'])})
    summary={'dataset':data.name,'candidates':len(rows),'max_total_tokens':512,'reserved_template_tokens':reserved,
        'main_capacity':cap,'truncation_side':tokenizer.truncation_side,
        'truncated_candidates':sum(r['truncated'] for r in rows),
        'candidates_with_no_retained_body':sum(r['retained_body_tokens']==0 for r in rows),
        'queries_with_any_truncation':len({r['qid'] for r in rows if r['truncated']}),
        'input_candidates_sha256':digest(a.run/'candidates.jsonl'),
        'interpretation':'Measured input loss, not a causal estimate of its ranking effect. Labels were not used to select or truncate model inputs.'}
    a.output.mkdir(parents=True,exist_ok=False)
    write_json(a.output/'summary.json',summary)
    write_json(a.output/'stage-attribution.json',stage_attribution(data,read_jsonl(a.run/'rows.jsonl'),candidates,rows))
    with (a.output/'rows.jsonl').open('w') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    write_json(a.output/'checksums.json',{p.name:digest(p) for p in a.output.iterdir() if p.is_file()})
    print(json.dumps(summary))


if __name__=='__main__':main()
