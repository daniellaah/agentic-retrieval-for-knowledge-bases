"""Post-hoc exact-score tie-policy replay on frozen candidates; no new inference."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from arkb.evaluation.external import digest,load_external,read_jsonl,verify_checksums,write_json,score_ranking,reference_metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path)
    parser.add_argument('--dataset',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();verify_checksums(args.run);data=load_external(args.dataset)
    if args.output.exists():raise ValueError('Use a new diagnostic path.')
    if json.loads((args.run/'experiment.json').read_text())['status']!='completed':raise ValueError('Incomplete run.')
    if digest(args.dataset/'manifest.json')!=json.loads((args.run/'protocol.json').read_text())['dataset_manifest_sha256']:
        raise ValueError('Unregistered dataset.')
    from transformers import AutoTokenizer
    import arkb.retrieval.qwen_rerank as q
    if digest(Path(q.__file__))!=digest(args.run/'measured-source/src/arkb/retrieval/qwen_rerank.py'):
        raise ValueError('Expected the measured input-construction implementation.')
    tokenizer=AutoTokenizer.from_pretrained(q.QWEN_MODEL,revision=q.QWEN_REVISION,cache_dir='.arkb/models',
        local_files_only=True,trust_remote_code=False,padding_side='left')
    capacity=512-len(tokenizer.encode(q.PREFIX,add_special_tokens=False))-len(tokenizer.encode(q.SUFFIX,add_special_tokens=False))
    mapping=data.source_map();measured=read_jsonl(args.run/'rows.jsonl');rows=[];alternative={}
    by_query={qid:{r['arm']:r for r in measured if r['qid']==qid} for qid in data.queries}
    for raw in read_jsonl(args.run/'candidates.jsonl'):
        qid=raw['qid'];arms=by_query[qid];pool=raw['rerank_inputs'];score_map={x['source']:x['score'] for x in raw.get('rerank_scores',[])}
        if arms['hybrid-rerank20']['error'] is not None:
            raise ValueError('This diagnostic requires all candidate scores; retained execution errors need a separately registered replay policy.')
        if {h['source'] for h in pool}!=set(score_map):raise ValueError('Missing candidate scores.')
        pairs=[f'<Instruct>: {q.INSTRUCTION}\n<Query>: {data.queries[qid]}\n<Document>: '+
               (h['metadata'].get('title') or '')+'\n\n'+h['content'] for h in pool]
        ids=tokenizer(pairs,padding=False,truncation='longest_first',return_attention_mask=False,max_length=capacity)['input_ids'] if pairs else []
        hashes=[hashlib.sha256(json.dumps(x).encode()).hexdigest() for x in ids]
        counts=Counter(score_map.values())
        order=sorted(range(len(pool)),key=lambda i:(-score_map[pool[i]['source']],i))
        ranking=[mapping[pool[i]['source']] for i in order]+arms['hybrid']['ranking'][len(pool):]
        metrics=score_ranking(data,qid,ranking);alternative[qid]=ranking
        rows.append({'qid':qid,'candidate_count':len(pool),'unique_effective_main_inputs':len(set(hashes)),
            'input_token_hashes':hashes,'unique_scores':len(counts),'candidates_in_tied_score_groups':sum(n for n in counts.values() if n>1),
            'measured_ranking':arms['hybrid-rerank20']['ranking'],'stable_input_order_ranking':ranking,
            'hybrid_metrics':arms['hybrid']['metrics'],'measured_metrics':arms['hybrid-rerank20']['metrics'],
            'stable_tie_metrics':metrics})
    reference=reference_metrics(data.qrels,alternative);max_error=max(
        abs(row['stable_tie_metrics'][k]-v) for row in rows for k,v in reference[row['qid']].items())
    if max_error>1e-9:raise ValueError('Alternative metric reference mismatch.')
    keys=rows[0]['stable_tie_metrics']
    means={name:{k:sum(r[name][k] for r in rows)/len(rows) for k in keys}
           for name in ('hybrid_metrics','measured_metrics','stable_tie_metrics')}
    write_json(args.output,{'schema':'arkb-p4-posthoc-tie-policy-v1','dataset':data.name,'queries':len(rows),
        'candidate_inputs_sha256':digest(args.run/'candidates.jsonl'),'measured_rows_sha256':digest(args.run/'rows.jsonl'),
        'audit_sha256':digest(Path(__file__)),'rule':'Exact equal scores retain original frozen hybrid-pool order; all other scores and candidates are unchanged.',
        'all_inputs_identical_queries':sum(r['candidate_count']>1 and r['unique_effective_main_inputs']==1 for r in rows),
        'all_scores_equal_queries':sum(r['candidate_count']>1 and r['unique_scores']==1 for r in rows),
        'queries_with_score_ties':sum(r['candidates_in_tied_score_groups']>0 for r in rows),
        'changed_rankings':sum(r['measured_ranking']!=r['stable_input_order_ranking'] for r in rows),
        'means':means,'stable_tie_minus_measured':{k:means['stable_tie_metrics'][k]-means['measured_metrics'][k] for k in keys},
        'reference_max_error':max_error,'rows':rows,
        'scope':'Exploratory deterministic policy replay chosen after inspecting these public-development results. No new model calls, unseen validation, human answer score, or product change. Input token equality includes all main tokens; the reserved prefix/suffix are identical constants.',
        'release_eligible':False})
    print(json.dumps({'dataset':data.name,'queries':len(rows),'mean_metrics':means,'reference_max_error':max_error}))


if __name__=='__main__':main()
