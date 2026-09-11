"""Offline rank/length attribution on retained candidates; no model inference."""
import argparse
import hashlib
import json
from pathlib import Path

from arkb.agent.tools import _evidence
from arkb.evaluation.controlled import fused_candidates
from arkb.evaluation.v2 import load_dataset,evidence_scores
from arkb.retrieval.qwen_rerank import QWEN_MODEL,QWEN_REVISION,PREFIX,SUFFIX,INSTRUCTION


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('bundle',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--reranker-cache',default='.arkb/models');a=p.parse_args()
    if json.loads((a.bundle/'experiment.json').read_text())['status']!='completed':raise ValueError('Incomplete experiment.')
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(QWEN_MODEL,revision=QWEN_REVISION,cache_dir=a.reranker_cache,
                                     local_files_only=True,trust_remote_code=False,padding_side='left')
    protocol=json.loads((a.bundle/'protocol.json').read_text());cfg=protocol['component']
    limit=cfg['reranker_max_length']-len(tok.encode(PREFIX,add_special_tokens=False))-len(tok.encode(SUFFIX,add_special_tokens=False))
    dataset=load_dataset(a.bundle/'dataset',notes_dir=a.bundle/'dataset/corpus',allow_provisional=True)
    cases={c['id']:c for c in dataset.cases};raw=[json.loads(l) for l in (a.bundle/'candidate-inputs.jsonl').read_text().splitlines()]
    results=[]
    for item in raw:
        case=cases[item['case_id']];ranked={};lengths=[]
        for depth in (10,20,40):
            union,_=fused_candidates(item['legs'],depth,pool_size=cfg['fusion_pool_size'],rrf_k=cfg['rrf_k'])
            first={f['id']:None for f in case['evidence_requirements']}
            for k in range(1,len(union)+1):
                score=evidence_scores(case,[_evidence(h) for h in union[:k]],dataset)
                for facet,ok in score['facet_satisfied'].items():
                    if ok and first[facet] is None:first[facet]=k
            ranked[f'depth{depth}_first_full_facet_rank']=first
        scores=item['rerank']['scores'];candidates=item['rerank']['candidates']
        # Same ordering as production Reranker, including identity tie breaks.
        from arkb.retrieval.models import SearchResult
        hits=[SearchResult(**h) for h in candidates]
        order=sorted(range(len(hits)),key=lambda i:(-scores[i],hits[i].identity))
        for i,hit in enumerate(hits):
            pair=f'<Instruct>: {INSTRUCTION}\n<Query>: {case["query"]}\n<Document>: {hit.metadata.get("title") or ""}\n\n{hit.content}'
            ids=tok(pair,padding=False,truncation=False,return_attention_mask=False)['input_ids']
            lengths.append({'input_rank':i+1,'reranked_rank':order.index(i)+1,'source':hit.source,'chunk_id':hit.chunk_id,
                'score':scores[i],'pair_tokens':len(ids),'pair_budget':limit,'would_truncate':len(ids)>limit})
        results.append({'case_id':case['id'],'family':case['intent_family_id'],**ranked,'reranker_candidates':lengths})
    result={'dataset_manifest_sha256':hashlib.sha256((a.bundle/'dataset/manifest.json').read_bytes()).hexdigest(),
            'raw_sha256':hashlib.sha256((a.bundle/'candidate-inputs.jsonl').read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'reranker_tokenizer_sha256':hashlib.sha256(tok.backend_tokenizer.to_str().encode()).hexdigest(),
            'reranker_revision':QWEN_REVISION,'queries':results,
            'truncated_candidate_pairs':sum(c['would_truncate'] for r in results for c in r['reranker_candidates']),
            'candidate_pairs':sum(len(r['reranker_candidates']) for r in results),
            'note':'Offline candidate rank and tokenizer-input length checks, not model attention or causal explanation of scores.'}
    with a.output.open('x') as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='queries'}))


if __name__=='__main__':main()
