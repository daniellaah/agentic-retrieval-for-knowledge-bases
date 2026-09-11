"""Freeze, run and replay external retrieval with the production ARKB components."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

from arkb.evaluation.external import digest, load_external, write_json, score_ranking, reference_metrics


def emit(stream, row):
    stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n'); stream.flush()


def unique_hits(hits, limit=100):
    selected=[]; seen=set()
    for hit in hits:
        if hit.source not in seen:
            selected.append(hit); seen.add(hit.source)
            if len(selected)==limit: break
    return selected


def verify_snapshot(runtime, storage, manifest):
    import hashlib
    import numpy as np
    from arkb.knowledge.qdrant import QdrantIndex
    _,records,vectors=storage.load_snapshot(manifest.index_version)
    backend=storage.build_metadata(manifest.index_version)['backend']
    index=QdrantIndex(runtime.qdrant_client(backend['url']),backend['collection'],manifest.embedding_spec,vault_id=manifest.vault_id)
    index.verify_snapshot(records,vectors)
    return {'manifest':asdict(manifest),'vectors_sha256':hashlib.sha256(np.asarray(vectors).tobytes()).hexdigest(),
            'record_count':len(records),'collection':backend['collection']}


def reranker_files():
    from arkb.retrieval.qwen_rerank import QWEN_MODEL, QWEN_REVISION
    directory=Path('.arkb/models')/('models--'+QWEN_MODEL.replace('/','--'))/'snapshots'/QWEN_REVISION
    files={p.relative_to(directory).as_posix():digest(p) for p in sorted(directory.rglob('*')) if p.is_file()}
    if not any(n.endswith('.safetensors') for n in files):raise ValueError('Missing pinned reranker weights.')
    return files


def summarize(data, rows):
    import numpy as np
    result={}; max_error=0.0
    arms=sorted({r['arm'] for r in rows})
    for arm in arms:
        rs=[r for r in rows if r['arm']==arm]
        if len(rs)!=len(data.queries) or {r['qid'] for r in rs}!=set(data.queries):
            raise ValueError('Incomplete/duplicate external matrix.')
        reference=reference_metrics(data.qrels,{r['qid']:r['ranking'] for r in rs})
        for r in rs:
            rescored=score_ranking(data,r['qid'],r['ranking'])
            if rescored!=r['metrics']: raise ValueError('Score replay mismatch.')
            for metric,v in reference[r['qid']].items():
                if rescored[metric] is not None: max_error=max(max_error,abs(rescored[metric]-v))
        names=rs[0]['metrics']
        result[arm]={'attempted':len(rs),'errors':sum(r['error'] is not None for r in rs),
            'empty_rankings':sum(not r['ranking'] for r in rs),
            'metrics':{m:sum(r['metrics'][m] for r in rs)/len(rs) for m in names},
            'mean_stage_ms':sum(r['elapsed_ms'] for r in rs)/len(rs)}
    comparisons={}
    if 'hybrid' in arms and 'hybrid-rerank20' in arms:
        left={r['qid']:r for r in rows if r['arm']=='hybrid'}
        right={r['qid']:r for r in rows if r['arm']=='hybrid-rerank20'}
        for m in next(iter(left.values()))['metrics']:
            delta=np.array([right[q]['metrics'][m]-left[q]['metrics'][m] for q in sorted(left)])
            rng=np.random.default_rng(20260910)
            means=np.mean(rng.choice(delta,size=(2000,len(delta)),replace=True),axis=1)
            comparisons[m]={'delta':float(delta.mean()),'ci95':np.quantile(means,[.025,.975]).tolist(),
                'wins':int((delta>0).sum()),'ties':int((delta==0).sum()),'losses':int((delta<0).sum())}
    if max_error>1e-9: raise ValueError('trec_eval cross-check failed.')
    return {'dataset':data.name,'queries':len(data.queries),'corpus_units':len(data.corpus),
        'arms':result,'rerank_minus_hybrid':comparisons,'reference_max_error':max_error,
        'reference_provider':'pytrec-eval-terrier==0.5.10, linear gains, relevance>0',
        'errors_count_as_empty_rankings':True,'release_eligible':False,
        'answer_quality':'not evaluated by retrieval labels'}


def execute(a):
    from arkb.retrieval.bm25 import BM25Retriever
    from arkb.retrieval.hybrid import rrf
    from arkb.runtime import Runtime
    from arkb.config import RuntimeConfig, RetrievalConfig
    from arkb.knowledge.sqlite import SQLiteStorage
    from arkb.evaluation.environment import runtime_environment
    out=a.output; data=load_external(a.dataset); mapping=data.source_map()
    meta={'status':'running','started_at':datetime.now(timezone.utc).isoformat(),
          'data_manifest_sha256':digest(a.dataset/'manifest.json')}
    write_json(out/'experiment.json',meta); rows=[]
    try:
        with (out/'rows.jsonl').open('x') as stream, (out/'candidates.jsonl').open('x') as candidates:
            if a.track=='lexical':
                start=perf_counter(); retriever=BM25Retriever(data.records(),index_id=data.name)
                meta['index_setup_ms']=(perf_counter()-start)*1000
                for i,(qid,query) in enumerate(data.queries.items()):
                    start=perf_counter(); error=None; ranking=[]
                    try: ranking=[mapping[h.source] for h in retriever.search(query,top_k=100).results]
                    except Exception as e: error={'type':type(e).__name__,'message':str(e)}
                    row={'qid':qid,'arm':'bm25-native-unit','ranking':ranking,'elapsed_ms':(perf_counter()-start)*1000,
                         'error':error,'metrics':score_ranking(data,qid,ranking)}
                    rows.append(row); emit(stream,row)
                    if i%25==0: print(data.name,'lexical',i+1,flush=True)
            else:
                if not (a.dataset/'corpus').exists(): data.materialize(a.dataset/'corpus')
                data.verify_materialized(a.dataset/'corpus')
                config=RuntimeConfig(offline=True,tokenizer_cache=Path('.uv-cache/tokenizers').resolve(),
                                     qdrant_url=a.qdrant_url)
                with Runtime(config) as runtime:
                    meta['environment_before']=runtime_environment(runtime)
                    (out/'reference-tokenizer.json').write_text(runtime.tokenizer().to_str())
                    tags=lambda:{m.model:m.digest for m in runtime.model_client().list().models
                                  if m.model in ('qwen3-embedding:0.6b','qwen3.5:4b')}
                    meta['models_before']=tags(); write_json(out/'experiment.json',meta)
                    print(data.name,'indexing full corpus',flush=True); start=perf_counter()
                    build=runtime.index(db=out/'index.sqlite',vault_id=data.name,notes_dir=a.dataset/'corpus',
                        chunking='recursive',chunk_size=512,chunk_overlap=64,batch_size=32,max_batch_tokens=8192)
                    meta['index_setup_ms']=(perf_counter()-start)*1000
                    meta['index_report']=asdict(build); write_json(out/'experiment.json',meta)
                    print(data.name,'indexed',build.manifest.chunk_count,meta['index_setup_ms'],flush=True)
                    with SQLiteStorage(out/'index.sqlite',read_only=True) as storage:
                        meta['snapshot_before']=verify_snapshot(runtime,storage,build.manifest)
                        meta['reranker_files_before']=reranker_files()
                        engine=runtime.retrieval_engine(storage,build.manifest,modes=('bm25','semantic'),rerank=True,
                            exact=True,settings=RetrievalConfig(candidate_k=500,reranker_cache=str(Path('.arkb/models').resolve())))
                        meta['reranker_identity']=engine.reranker.scorer.identity
                        for i,(qid,query) in enumerate(data.queries.items()):
                            raw={'qid':qid,'legs':{},'stage_ms':{},'rerank_inputs':[]}; outputs={}; failure=None
                            try:
                                for leg in ('bm25','semantic'):
                                    start=perf_counter(); response=engine.search(query,mode=leg,top_k=500)
                                    if response.index_id!=build.manifest.index_version: raise ValueError('Snapshot changed.')
                                    raw['stage_ms'][leg]=(perf_counter()-start)*1000
                                    raw['legs'][leg]=[{'source':h.source,'chunk_id':h.chunk_id,'score':h.score,
                                        'start_char':h.start_char,'end_char':h.end_char} for h in response.results]
                                    outputs[leg]=list(response.results)
                                start=perf_counter(); outputs['hybrid']=list(rrf({k:outputs[k] for k in ('bm25','semantic')},k=60))
                                raw['stage_ms']['hybrid']=(perf_counter()-start)*1000
                                pool=unique_hits(outputs['hybrid'],20)
                                raw['rerank_inputs']=[asdict(h) for h in pool]
                                start=perf_counter(); reranked=engine.reranker.rerank(query,pool)
                                raw['stage_ms']['hybrid-rerank20']=(perf_counter()-start)*1000
                                raw['rerank_scores']=[{'source':h.source,'score':h.score} for h in reranked]
                                outputs['hybrid-rerank20']=list(reranked)+unique_hits(outputs['hybrid'])[20:]
                            except Exception as e: failure={'type':type(e).__name__,'message':str(e)}
                            for arm in ('bm25','semantic','hybrid','hybrid-rerank20'):
                                ranking=[mapping[h.source] for h in unique_hits(outputs.get(arm,[]))]
                                row={'qid':qid,'arm':arm,'ranking':ranking,'error':failure if arm not in outputs else None,
                                    'elapsed_ms':raw['stage_ms'].get(arm,0.0),'metrics':score_ranking(data,qid,ranking)}
                                rows.append(row); emit(stream,row)
                            emit(candidates,raw)
                            if i%10==0: print(data.name,'components',i+1,'/',len(data.queries),flush=True)
                        meta['snapshot_after']=verify_snapshot(runtime,storage,build.manifest)
                        if meta['snapshot_before']!=meta['snapshot_after']: raise ValueError('Snapshot drift.')
                        meta['reranker_files_after']=reranker_files()
                        if meta['reranker_files_before']!=meta['reranker_files_after']:raise ValueError('Reranker files changed.')
                    meta['models_after']=tags()
                    if meta['models_before']!=meta['models_after']: raise ValueError('Model identity drift.')
                    meta['environment_after']=runtime_environment(runtime)
                    if meta['environment_before']!=meta['environment_after']:raise ValueError('Runtime environment changed.')
                data.verify_materialized(a.dataset/'corpus')
        if digest(a.dataset/'manifest.json')!=meta['data_manifest_sha256']: raise ValueError('Dataset manifest drift.')
        load_external(a.dataset)
        write_json(out/'summary.json',summarize(data,rows))
        meta['status']='completed'; meta['rows']=len(rows)
    except BaseException as e:
        meta.update(status='failed',error={'type':type(e).__name__,'message':str(e)})
        raise
    finally:
        meta['finished_at']=datetime.now(timezone.utc).isoformat(); write_json(out/'experiment.json',meta)
        write_json(out/'checksums.json',{p.relative_to(out).as_posix():digest(p) for p in out.rglob('*')
            if p.is_file() and p.name not in ('checksums.json','execution.log') and not p.name.endswith(('-wal','-shm','.pyc'))})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--track',choices=('lexical','indexed'),required=True)
    p.add_argument('--qdrant-url',default='http://127.0.0.1:32776')
    p.add_argument('--reuse-index',type=Path,help='Copy a prior compatible P4 SQLite cache into a NEW run; preserve failed attempt.')
    p.add_argument('--execute',action='store_true'); p.add_argument('--replay',action='store_true')
    a=p.parse_args(); a.dataset=a.dataset.resolve(); a.output=a.output.resolve()
    if a.replay:
        from arkb.evaluation.external import read_jsonl, verify_checksums
        verify_checksums(a.output)
        protocol=json.loads((a.output/'protocol.json').read_text())
        if digest(a.dataset/'manifest.json')!=protocol['dataset_manifest_sha256']:raise ValueError('Unregistered dataset.')
        print(json.dumps(summarize(load_external(a.dataset),read_jsonl(a.output/'rows.jsonl')),indent=2)); return
    if a.execute: execute(a); return
    a.output.mkdir(parents=True,exist_ok=False)
    if a.reuse_index:
        import sqlite3
        with sqlite3.connect('file:'+str(a.reuse_index.resolve())+'?mode=ro',uri=True) as source:
            with sqlite3.connect(a.output/'index.sqlite') as target:source.backup(target)
    root=Path(__file__).resolve().parents[2]
    shutil.copytree(root/'src',a.output/'measured-source/src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('pyproject.toml','uv.lock'): shutil.copyfile(root/name,a.output/'measured-source'/name)
    shutil.copyfile(__file__,a.output/'run_p4.py')
    shutil.copyfile(a.dataset/'manifest.json',a.output/'data-manifest.json')
    write_json(a.output/'protocol.json',{'schema':'arkb-p4-external-v1','dataset_manifest_sha256':digest(a.dataset/'manifest.json'),
        'track':a.track,'cutoffs':[10,100],'positive_threshold':1,'gain':'linear','errors':'retained; empty ranking if no output',
        'lexical_unit':'whole official unit','indexed_chunking':{'size':512,'overlap':64},'indexed_leg_depth':500,
        'aggregation':'first occurrence per original source ID, at most 100 unique IDs in frozen candidate lists',
        'rerank':'first 20 unique hybrid sources using best-ranked chunk; untouched tail to 100',
        'rrf_k':60,'latency':'per stage; shared retrieval excludes synthetic end-to-end timing',
        'reused_index_sha256':digest(a.reuse_index) if a.reuse_index else None,
        'release_eligible':False})
    env={**os.environ,'PYTHONPATH':str(a.output/'measured-source/src'),'PYTHONDONTWRITEBYTECODE':'1',
         'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false'}
    command=[sys.executable,'-u',str(a.output/'run_p4.py'),'--execute','--dataset',str(a.dataset),
             '--output',str(a.output),'--track',a.track,'--qdrant-url',a.qdrant_url]
    with (a.output/'execution.log').open('x') as log:
        result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    print(json.dumps({'output':str(a.output),'exit_code':result.returncode})); raise SystemExit(result.returncode)


if __name__=='__main__': main()
