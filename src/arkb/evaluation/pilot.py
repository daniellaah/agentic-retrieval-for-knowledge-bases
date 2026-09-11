"""Run v2 provisional retrieval and explicit match/read contract diagnostics.

No Agent, no answer judge, no release-quality claim. The unchanged production
Runtime supplies retrieval; gold labels are used only after observations return.
"""
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
from time import perf_counter

import httpx
import numpy as np

from arkb.evaluation.v2 import load_dataset, evidence_scores, ranking_scores, dataset_report
from arkb.evaluation.runs import _code_metadata, _write_json
from arkb.config import RuntimeConfig, RetrievalConfig, DEFAULT_EMBEDDING_MODEL
from arkb.runtime import Runtime
from arkb.knowledge.sqlite import SQLiteStorage
from arkb.knowledge.qdrant import QdrantIndex
from arkb.agent.tools import _evidence


def score_observations(case, hits, dataset, *, k):
    labels={r['source']:r['grade'] for r in dataset.qrels if r['query_id']==case['id']}
    ranked=list(dict.fromkeys(hit['source'] for hit in hits))
    return {'source_metrics_from_top_k_chunks':ranking_scores(labels,ranked,k=k),
            'evidence':evidence_scores(case,hits,dataset),'raw_result_count':len(hits),
            'unique_source_count':len(ranked),'ranked_sources':ranked}


def score_case(case, hits, dataset, *, method):
    result=score_observations(case,hits,dataset,k=10)
    if method in ('explicit_match','explicit_read'):
        # These tools expose occurrences or a selected body, not a ranked list.
        result['source_metrics_from_top_k_chunks']=None
    if method=='explicit_match':
        expected={r['source'] for r in dataset.qrels if r['query_id']==case['id'] and r['grade']>=2}
        result['exact_source_set_equal']={h['source'] for h in hits}==expected
    return result


def run_pilot(dataset_dir, output, *, qdrant_url, host='http://127.0.0.1:11434',
              tokenizer_cache=Path('.uv-cache/tokenizers'), reranker_cache=Path('.arkb/models'),
              allow_provisional=False, protocol_path=Path('evaluation/experiments/pilot-protocol.json')):
    dataset_dir,output=Path(dataset_dir).resolve(),Path(output).resolve()
    dataset=load_dataset(dataset_dir,notes_dir=dataset_dir/'corpus',allow_provisional=allow_provisional)
    protocol=json.loads(Path(protocol_path).read_text())
    report=dataset_report(dataset)
    expected={'manifest_sha256':hashlib.sha256((dataset_dir/'manifest.json').read_bytes()).hexdigest(),
              'case_count':report['case_count'],'intent_family_count':report['intent_families'],
              'top_k_chunks':10,'candidate_k':20,'rerank_candidates':20,'chunk_size':512,'chunk_overlap':64,
              'embedding_context_length':8192,'embedding_model':DEFAULT_EMBEDDING_MODEL,
              'reranker_max_length':512,'explicit_match_occurrence_limit':10000,'trials_per_method_case':1,
              'planned_rows':sum(1 if c['task_type'] in ('exact_lookup','direct_read','no_retrieval') else 4 for c in dataset.cases)}
    if any(protocol.get(key)!=value for key,value in expected.items()):
        raise ValueError('Dataset or supported execution settings differ from the pilot protocol.')
    output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(protocol_path,output/'protocol.json')
    # Retain replay inputs even if the development dataset is later revised.
    shutil.copytree(dataset_dir,output/'dataset')
    dataset_dir=output/'dataset'
    source_root=Path(__file__).resolve().parents[3]
    shutil.copytree(source_root/'src',output/'measured-source/src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('pyproject.toml','uv.lock'):
        shutil.copyfile(source_root/name,output/'measured-source'/name)
    config=RuntimeConfig(host=host,qdrant_url=qdrant_url,offline=True,tokenizer_cache=tokenizer_cache,
                         embedding_model=DEFAULT_EMBEDDING_MODEL)
    code_before=_code_metadata()
    manifest_hash=hashlib.sha256((dataset_dir/'manifest.json').read_bytes()).hexdigest()
    metadata={'status':'running','started_at':datetime.now(timezone.utc).isoformat(),
              'dataset':dataset_report(dataset),'manifest_sha256':manifest_hash,
              'source':code_before,'schema':'arkb-pilot-run-v1','baseline_budget':{'top_k_chunks':10,'candidate_k':20},
              'limits':['provisional labels','unassessed documents are not verified negatives',
                        'no answer/Agent scoring','single-run uncontrolled load/cache timing'],
              'model_services':asdict(config),'planned_rows':sum(1 if c['task_type'] in ('exact_lookup','direct_read','no_retrieval') else 4 for c in dataset.cases)}
    _write_json(output/'run_metadata.json',metadata)
    rows=[]
    try:
        def provider():
            with httpx.Client(trust_env=False,timeout=10) as client:
                tags=client.get(host+'/api/tags');tags.raise_for_status()
                model=next(m for m in tags.json()['models'] if m['name']==DEFAULT_EMBEDDING_MODEL)
                return {'embedding_model':model['name'],'digest':model['digest'],
                        'ollama_version':client.get(host+'/api/version').json(),
                        'qdrant_version':client.get(qdrant_url+'/').json()}
        metadata['provider_before']=provider()
        metadata['packages']={n:version(n) for n in ('numpy','ollama','qdrant-client','torch','transformers')}
        with Runtime(config) as runtime:
            build=runtime.index(db=output/'index.sqlite',vault_id=output.name,notes_dir=dataset_dir/'corpus')
            _write_json(output/'index-report.json',asdict(build))
            with SQLiteStorage(output/'index.sqlite',read_only=True) as storage:
                def snapshot():
                    manifest,records,vectors=storage.load_snapshot(build.manifest.index_version)
                    backend=storage.build_metadata(manifest.index_version)['backend']
                    index=QdrantIndex(runtime.qdrant_client(qdrant_url),backend['collection'],
                                      manifest.embedding_spec,vault_id=output.name)
                    index.verify_snapshot(records,vectors)
                    return {'collection':backend['collection'],
                            'vectors_sha256':hashlib.sha256(np.asarray(vectors).tobytes()).hexdigest(),
                            'records':[(r.document_id,r.document_revision,r.chunk_id) for r in records]}
                metadata['snapshot_before']=snapshot()
                engine=runtime.retrieval_engine(storage,build.manifest,modes=('bm25','semantic','hybrid'),
                    rerank=True,settings=RetrievalConfig(reranker_cache=str(reranker_cache)))
                metadata['reranker_identity']=engine.reranker.scorer.identity
                metadata['snapshot']=asdict(build.manifest)
                tools=runtime.agent_tools(engine=engine,directory=dataset_dir/'corpus',vault_id=output.name)
                with (output/'results.jsonl').open('x') as stream:
                    for position,case in enumerate(dataset.cases):
                        if case['task_type'] in ('exact_lookup','direct_read','no_retrieval'):
                            methods=[{'exact_lookup':'explicit_match','direct_read':'explicit_read','no_retrieval':'not_applicable'}[case['task_type']]]
                        else:
                            names=['bm25','semantic','hybrid','hybrid_rerank'];offset=position%len(names)
                            methods=names[offset:]+names[:offset]
                        for method in methods:
                            row={'case_id':case['id'],'family':case['intent_family_id'],'task_type':case['task_type'],
                                 'query':case['query'],'method':method,'status':'not_applicable',
                                 'observations':None,'metrics':None,'error':None,'latency_ms':None}
                            if method!='not_applicable':
                                start=perf_counter()
                                try:
                                    if method=='explicit_match':
                                        hits=tools.match(**case['exact_pattern'],limit=10000)['results']
                                    elif method=='explicit_read':
                                        hits=[tools.read(source=case['read_source'])['result']]
                                    else:
                                        response=engine.search(case['query'],mode='hybrid' if method=='hybrid_rerank' else method,
                                                               rerank=method=='hybrid_rerank',top_k=10)
                                        hits=[_evidence(hit) for hit in response.results]
                                        row['raw_response']=asdict(response)
                                    row.update(status='ok',observations=hits,latency_ms=(perf_counter()-start)*1000)
                                    row['metrics']=score_case(case,hits,dataset,method=method)
                                except Exception as error:
                                    row.update(status='error',error={'type':type(error).__name__,'message':str(error)},
                                               latency_ms=(perf_counter()-start)*1000,metrics=None)
                            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();rows.append(row)
                        print(f'Completed {case["id"]} ({len(rows)} rows)',flush=True)
                metadata['snapshot_after']=snapshot()
        # Revalidate corpus and every annotation hash after the run.
        load_dataset(dataset_dir,notes_dir=dataset_dir/'corpus',allow_provisional=allow_provisional)
        if hashlib.sha256((dataset_dir/'manifest.json').read_bytes()).hexdigest()!=manifest_hash or _code_metadata()!=code_before:
            raise ValueError('Code or dataset changed during pilot.')
        if len(rows)!=metadata['planned_rows']:
            raise ValueError('Incomplete pilot matrix.')
        metadata['provider_after']=provider()
        if metadata['provider_before']!=metadata['provider_after'] or metadata['snapshot_before']!=metadata['snapshot_after']:
            raise ValueError('Provider or index changed during pilot.')
        cases={c['id']:c for c in dataset.cases}
        replay_rows=[json.loads(line) for line in (output/'results.jsonl').read_text().splitlines()]
        for row in replay_rows:
            if row['status']=='ok' and row['metrics']!=score_case(cases[row['case_id']],row['observations'],dataset,method=row['method']):
                raise ValueError('Pilot metric replay mismatch.')
        _write_json(output/'validation.json',{'valid':True,'rows_checked':len(replay_rows),
            'metrics_replayed':sum(r['status']=='ok' for r in replay_rows),'snapshot_vectors_verified':True,
            'input_drift':False,'release_eligible':False})
        metadata['status']='completed'
        summary={'dataset':dataset_report(dataset),'total_rows':len(rows),'statuses':dict(Counter(r['status'] for r in rows)),
                 'release_eligible':False,'by_method':{}}
        for method in sorted({r['method'] for r in rows}):
            group=[r for r in rows if r['method']==method]
            values=[r['metrics']['evidence']['evidence_coverage'] for r in group if r['status']=='ok'
                    and r['metrics']['evidence']['evidence_coverage'] is not None]
            summary['by_method'][method]={'rows':len(group),'errors':sum(r['status']=='error' for r in group),
                'mean_provisional_evidence_coverage':sum(values)/len(values) if values else None,
                'coverage_defined_cases':len(values)}
        _write_json(output/'summary.json',summary)
    except BaseException as error:
        metadata.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                        error={'type':type(error).__name__,'message':str(error)})
        raise
    finally:
        metadata['completed_rows']=len(rows);metadata['finished_at']=datetime.now(timezone.utc).isoformat()
        _write_json(output/'run_metadata.json',metadata)
        _write_json(output/'checksums.json',{p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(output.rglob('*')) if p.is_file() and p.name!='checksums.json'
            and not p.name.endswith(('-wal','-shm','.pyc','.lock'))})
    return summary


def main(argv=None):
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=Path('evaluation/data/v2/pilot'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--qdrant-url',required=True)
    p.add_argument('--allow-provisional',action='store_true')
    a=p.parse_args(argv)
    result=run_pilot(a.dataset,a.output,qdrant_url=a.qdrant_url,allow_provisional=a.allow_provisional)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
