"""Replay P4 scores and rank assembly from retained candidates; no model calls."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def rank_assembly(data,rows,candidates):
    from arkb.knowledge.models import _document_id
    mapping=data.source_map();docs={mapping_key:d for mapping_key,d in zip(mapping,data.corpus)}
    # Rebuild stable document/chunk identity and fusion independently of rrf().
    key=lambda h:(_document_id(data.name,h['source']),'chunk',h['chunk_id'])
    def unique(hits):
        seen=set();out=[]
        for h in hits:
            if h['source'] not in seen:out.append(h);seen.add(h['source'])
        return out
    if len(candidates)!=len(data.queries) or {c['qid'] for c in candidates}!=set(data.queries):
        raise ValueError('Incomplete candidate records.')
    indexed={(r['qid'],r['arm']):r for r in rows};checked=0;span_checks=0
    for c in candidates:
        votes={};known={};outputs={}
        for leg,hits in c['legs'].items():
            if len(hits)>500 or len({key(h) for h in hits})!=len(hits):raise ValueError('Invalid leg candidates.')
            outputs[leg]=unique(hits)[:100]
            for rank,h in enumerate(hits,1):
                if h['source'] not in mapping:raise ValueError('Unknown candidate source.')
                identity=key(h);known[identity]=h;votes.setdefault(identity,[]).append(1/(60+rank))
        if set(c['legs'])=={'bm25','semantic'}:
            fused=[known[k] for k in sorted(known,key=lambda k:(-math.fsum(votes[k]),k))]
            hybrid=unique(fused)[:100];outputs['hybrid']=hybrid;pool=hybrid[:20]
            if [key(h) for h in c['rerank_inputs']]!=[key(h) for h in pool]:raise ValueError('Rerank pool differs from fusion.')
            for h in c['rerank_inputs']:
                d=docs[h['source']];body=d['text'].strip()
                if h['source_id']!=key(h)[0] or h['content']!=body[h['start_char']:h['end_char']]:
                    raise ValueError('Retained rerank input differs from source body.')
                if h['metadata'].get('title')!=' '.join(d['title'].splitlines()).strip():raise ValueError('Rerank title mismatch.')
                span_checks+=1
            if 'rerank_scores' in c:
                scores={h['source']:h['score'] for h in c['rerank_scores']}
                if set(scores)!={h['source'] for h in pool} or len(scores)!=len(c['rerank_scores']):raise ValueError('Rerank score coverage mismatch.')
                if any(not math.isfinite(s) for s in scores.values()):raise ValueError('Invalid rerank score.')
                front=sorted(pool,key=lambda h:(-scores[h['source']],key(h)))
                if [h['source'] for h in front]!=[h['source'] for h in c['rerank_scores']]:raise ValueError('Rerank order mismatch.')
                outputs['hybrid-rerank20']=front+hybrid[20:]
        for arm in ('bm25','semantic','hybrid','hybrid-rerank20'):
            r=indexed[c['qid'],arm];ranking=[mapping[h['source']] for h in outputs.get(arm,[])]
            if r['ranking']!=ranking:raise ValueError('Ranking assembly mismatch: '+c['qid']+' '+arm)
            checked+=1
    return {'rankings_reassembled':checked,'source_body_spans_verified':span_checks}


def replay(run,dataset):
    import arkb
    from arkb.evaluation.external import load_external,read_jsonl,digest
    if not Path(arkb.__file__).resolve().is_relative_to(run/'measured-source/src'):raise ValueError('Expected frozen implementation.')
    data=load_external(dataset);protocol=json.loads((run/'protocol.json').read_text())
    if digest(dataset/'manifest.json')!=protocol['dataset_manifest_sha256']:raise ValueError('Dataset registration mismatch.')
    spec=importlib.util.spec_from_file_location('measured_p4',run/'run_p4.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=read_jsonl(run/'rows.jsonl');summary=module.summarize(data,rows)
    if summary!=json.loads((run/'summary.json').read_text()):raise ValueError('Summary replay differs.')
    if protocol['track']=='indexed':audit=rank_assembly(data,rows,read_jsonl(run/'candidates.jsonl'))
    else:
        from arkb.retrieval.bm25 import BM25Retriever
        retriever=BM25Retriever(data.records(),index_id=data.name);mapping=data.source_map()
        for row in rows:
            ranking=[mapping[h.source] for h in retriever.search(data.queries[row['qid']],top_k=100).results]
            if ranking!=row['ranking']:raise ValueError('Native BM25 retrieval replay differs: '+row['qid'])
        audit={'lexical_rankings_recomputed':len(rows)}
    return {'rows_replayed':len(rows),'reference_max_error':summary['reference_max_error'],**audit,'release_eligible':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--child',action='store_true');p.add_argument('--archive-root',type=Path)
    a=p.parse_args();a.run=a.run.resolve();a.dataset=a.dataset.resolve()
    if a.child:print(json.dumps(replay(a.run,a.dataset)));return
    from arkb.evaluation.external import verify_checksums,digest
    if a.archive_root:
        root=a.archive_root.resolve();manifest=json.loads((root/'archive-manifest.json').read_text())
        for name,h in manifest['files'].items():
            path=root/name
            if not path.resolve().is_relative_to(root) or path.is_symlink() or digest(path)!=h:raise ValueError('Archive integrity mismatch.')
        if not a.run.is_relative_to(root) or not a.dataset.is_relative_to(root):raise ValueError('Inputs outside archive.')
    else:verify_checksums(a.run)
    if json.loads((a.run/'experiment.json').read_text())['status']!='completed':raise ValueError('Incomplete run.')
    subprocess.run([sys.executable,str(Path(__file__).resolve()),str(a.run),'--dataset',str(a.dataset),'--child'],
        env={**os.environ,'PYTHONPATH':str(a.run/'measured-source/src'),'PYTHONDONTWRITEBYTECODE':'1','HF_HUB_OFFLINE':'1'},check=True)


if __name__=='__main__':main()
