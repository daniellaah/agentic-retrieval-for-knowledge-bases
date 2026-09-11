"""ARKB Agent diagnostics on fixed Bright-Pro queries or isolated MuSiQue pairs."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter

from arkb.evaluation.external import ExternalDataset, digest, load_external, write_json, verify_checksums
from arkb.evaluation.multihop import OUTPUT_INSTRUCTION, parse_prediction, musique_metrics


def index_musique_context(runtime, output, case_index, directory):
    # A vault is bound to one source directory. Separate databases preserve
    # that production guard and keep every variant's active snapshot isolated.
    db = output/'context-indexes'/f'{case_index}.sqlite'
    db.parent.mkdir(parents=True, exist_ok=True)
    if db.exists():
        raise ValueError('MuSiQue context index already exists; retain the prior attempt.')
    started = perf_counter()
    build = runtime.index(db=db, vault_id='p4-musique', notes_dir=directory,
                          chunking='none', batch_size=32)
    return db, build, (perf_counter()-started)*1000


def execute(a):
    from arkb.runtime import Runtime
    from arkb.config import RuntimeConfig
    from arkb.knowledge.sqlite import SQLiteStorage
    from arkb.agent.observation import AgentBudget
    from arkb.evaluation.environment import runtime_environment
    from arkb.evaluation.deadline import evaluation_deadline
    out=a.output; rows=[]; predictions=[]
    metadata={'status':'running','started_at':datetime.now(timezone.utc).isoformat(),'track':a.track}
    write_json(out/'experiment.json',metadata)
    try:
        preparation_started=perf_counter()
        with ExitStack() as resources, Runtime(RuntimeConfig(offline=True,tokenizer_cache=Path('.uv-cache/tokenizers').resolve(),qdrant_url=a.qdrant_url)) as runtime:
            metadata['environment_before']=runtime_environment(runtime)
            (out/'reference-tokenizer.json').write_text(runtime.tokenizer().to_str())
            def definition():
                info=runtime.model_client().show('qwen3.5:4b').model_dump(mode='json')
                return {'parameters':info.get('parameters'),'capabilities':info.get('capabilities'),
                    'model_info':{k:v for k,v in (info.get('model_info') or {}).items() if 'context_length' in k or 'architecture' in k},
                    'template_sha256':hashlib.sha256((info.get('template') or '').encode()).hexdigest()}
            metadata['chat_model_definition_before']=definition()
            models=lambda:{m.model:m.digest for m in runtime.model_client().list().models
                           if m.model in ('qwen3.5:4b','qwen3-embedding:0.6b')}
            metadata['models_before']=models()
            if a.track=='musique':
                selected=json.loads((out/'selected.json').read_text())
                cases=selected['rows']
            else:
                data=load_external(a.dataset);domain=data.name.removeprefix('bright-')
                samples=json.loads((out/'agentic_sample_ids.json').read_text())
                ids=[str(x) for x in samples['tasks'][domain]]
                cases=[{'id':q,'question':data.queries[q]} for q in ids]
                metadata['dataset_manifest_sha256']=digest(a.dataset/'manifest.json')
                data.verify_materialized(a.dataset/'corpus')
                if json.loads((a.index_run/'experiment.json').read_text())['status']!='completed':
                    raise ValueError('Bright Agent requires a completed index experiment.')
                metadata['verified_index_artifacts']=verify_checksums(a.index_run)
                metadata['index_checksums_sha256']=digest(a.index_run/'checksums.json')
                storage=resources.enter_context(SQLiteStorage(a.index_run/'index.sqlite',read_only=True))
                manifest=storage.active_manifest(data.name)
                metadata['index_manifest']=asdict(manifest)
                engine=runtime.retrieval_engine(storage,manifest,modes=('bm25','semantic'),exact=True)
                bright_tools=runtime.agent_tools(engine=engine,directory=a.dataset/'corpus',vault_id=data.name)
                write_json(out/'experiment.json',metadata)
            metadata['preparation_ms']=(perf_counter()-preparation_started)*1000
            write_json(out/'experiment.json',metadata)
            with (out/'rows.jsonl').open('x') as stream:
                for i,case in enumerate(cases):
                    if a.track=='musique':
                        corpus=[{'id':str(p['idx']),'title':p['title'],'text':p['paragraph_text']} for p in case['paragraphs']]
                        d=ExternalDataset(f'musique-{i}',corpus,{'q':case['question']},{'q':{}},{},{'scope':'per-question supplied context'})
                        directory=out/'contexts'/str(i);d.materialize(directory)
                        db,build,context_index_ms=index_musique_context(runtime,out,i,directory)
                        vault='p4-musique';source_map={k:int(v) for k,v in d.source_map().items()}
                        prompt=case['question']+OUTPUT_INSTRUCTION
                    else:
                        directory=a.dataset/'corpus';db=a.index_run/'index.sqlite';vault=data.name
                        build=None;prompt=case['question'];source_map=data.source_map()
                    start=perf_counter();error=None;result=None
                    observer=runtime.agent_observer(budget=AgentBudget(max_tool_calls=12,max_query_calls=10,
                        max_read_calls=6,max_evidence_tokens=4000,max_elapsed_ms=120000))
                    with ExitStack() as case_resources:
                        if a.track=='musique':
                            storage=case_resources.enter_context(SQLiteStorage(db,read_only=True))
                            manifest=storage.active_manifest(vault)
                            engine=runtime.retrieval_engine(storage,manifest,modes=('bm25','semantic'),exact=True)
                            tools=runtime.agent_tools(engine=engine,directory=directory,vault_id=vault)
                        else:tools=bright_tools
                        try:
                            with evaluation_deadline(180):
                                result=runtime.run_agent(prompt,tools=tools,model='qwen3.5:4b',max_turns=8,think=True,observer=observer)
                        except Exception as e:
                            error={'type':type(e).__name__,'message':str(e)};result=getattr(e,'agent_result',None)
                    row={'case_index':i,'id':case['id'],'elapsed_ms':(perf_counter()-start)*1000,'error':error,
                         'result':asdict(result) if result is not None else None,
                         'stop_reason':result.stop_reason if result is not None else 'error'}
                    if a.track=='musique':
                        row['context_indexing']={'elapsed_ms':context_index_ms,'build_report':asdict(build)}
                    row['loaded_models_after']=[{k:m.model_dump(mode='json').get(k) for k in ('model','digest','size','size_vram','context_length')}
                        for m in runtime.model_client().ps().models if m.model in ('qwen3.5:4b','qwen3-embedding:0.6b')]
                    if a.track=='musique':
                        prediction,parse_error=parse_prediction(result.trace.final_response if result else '',source_map,
                                                                stopped=row['stop_reason'])
                        prediction={'id':case['id'],**prediction};predictions.append(prediction)
                        row.update(prediction=prediction,parse_error=parse_error,answerable_gold=case['answerable'])
                    else:
                        # The trace is kept for manual/independent output review; no
                        # LLM-generated score is silently promoted to gold quality.
                        row['answer_quality']='pending independent review'
                    rows.append(row);stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
                    print(a.track,i+1,'/',len(cases),row['stop_reason'],flush=True)
            metadata['models_after']=models()
            if metadata['models_before']!=metadata['models_after']:raise ValueError('Model identity changed.')
            metadata['chat_model_definition_after']=definition()
            if metadata['chat_model_definition_before']!=metadata['chat_model_definition_after']:raise ValueError('Chat model definition changed.')
            metadata['environment_after']=runtime_environment(runtime)
            if metadata['environment_before']!=metadata['environment_after']:raise ValueError('Runtime environment changed.')
        summary={'rows':len(rows),'execution_errors':sum(r['error'] is not None for r in rows),
                 'stop_reasons':{s:sum(r['stop_reason']==s for r in rows) for s in {r['stop_reason'] for r in rows}},
                 'release_eligible':False}
        if a.track=='musique':
            summary.update(metrics=musique_metrics(cases,predictions),parse_errors=sum(r['parse_error'] is not None for r in rows))
            with (out/'predictions.jsonl').open('w') as f:
                for p in predictions:f.write(json.dumps(p)+'\n')
            with (out/'gold.jsonl').open('w') as f:
                for c in cases:f.write(json.dumps(c)+'\n')
        else:
            data.verify_materialized(a.dataset/'corpus')
            if digest(a.dataset/'manifest.json')!=metadata['dataset_manifest_sha256']:raise ValueError('Dataset changed.')
            if digest(a.index_run/'checksums.json')!=metadata['index_checksums_sha256']:raise ValueError('Index registration changed.')
            verify_checksums(a.index_run)
            summary['answer_quality']='pending independent review'
        write_json(out/'summary.json',summary);metadata['status']='completed'
    except BaseException as e:
        metadata.update(status='failed',error={'type':type(e).__name__,'message':str(e)});raise
    finally:
        metadata['finished_at']=datetime.now(timezone.utc).isoformat();write_json(out/'experiment.json',metadata)
        write_json(out/'checksums.json',{p.relative_to(out).as_posix():digest(p) for p in out.rglob('*')
            if p.is_file() and p.name not in ('checksums.json','execution.log') and not p.name.endswith(('-wal','-shm','.pyc'))})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--track',choices=('musique','bright'),required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dataset',type=Path);p.add_argument('--index-run',type=Path)
    p.add_argument('--selected',type=Path,default=Path('evaluation/results/p4-data/musique-selected.json'))
    p.add_argument('--samples',type=Path,default=Path('evaluation/results/p4-inputs/official/bright-pro/agentic_sample_ids.json'))
    p.add_argument('--qdrant-url',default='http://127.0.0.1:32776');p.add_argument('--execute',action='store_true')
    a=p.parse_args();a.output=a.output.resolve()
    if a.dataset:a.dataset=a.dataset.resolve()
    if a.index_run:a.index_run=a.index_run.resolve()
    if a.track=='bright' and (not a.dataset or not a.index_run):p.error('Bright track requires dataset and completed index run.')
    if a.execute:execute(a);return
    a.output.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[2]
    shutil.copytree(root/'src',a.output/'measured-source/src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for n in ('pyproject.toml','uv.lock'):shutil.copyfile(root/n,a.output/'measured-source'/n)
    shutil.copyfile(__file__,a.output/'run_p4_agents.py')
    shutil.copyfile(a.selected if a.track=='musique' else a.samples,a.output/('selected.json' if a.track=='musique' else 'agentic_sample_ids.json'))
    if a.dataset:shutil.copyfile(a.dataset/'manifest.json',a.output/'data-manifest.json')
    write_json(a.output/'protocol.json',{'schema':'arkb-p4-agent-v3' if a.track=='musique' else 'arkb-p4-agent-v2','track':a.track,'model':'qwen3.5:4b',
        'trials':1,'max_turns':8,'think':True,'budget':{'tools':12,'queries':10,'reads':6,'evidence_tokens':4000,'cooperative_deadline_ms':120000},
        'query_suffix':OUTPUT_INSTRUCTION if a.track=='musique' else '',
        'harness_hard_deadline_ms':180000,
        'hard_deadline_scope':'Each run_agent call, excluding per-case index preparation. POSIX timer raises a retained execution error; server-side work cancellation is not guaranteed. Native code may defer signal handling; actual elapsed time is reported.',
        'instrumentation_revision':'v2.1 records MuSiQue per-context indexing costs and uses a RuntimeError deadline subtype to avoid HTTP transport remapping. The 180-second limit, prompts, selection, tools and product budgets match v2. SO v2 had only a tool timeout and no model-request errors.',
        'amendment':'v1 first Bright Stack Overflow attempt interrupted after more than 17 minutes in exact matching, before any completed case. v2 adds an evaluation-only hard deadline; product code and cooperative budget are unchanged. v1 is retained as an incomplete attempt.',
        'selection_sha256':digest(a.output/('selected.json' if a.track=='musique' else 'agentic_sample_ids.json')),
        'dataset_manifest_sha256':digest(a.dataset/'manifest.json') if a.dataset else None,
        'engine_lifetime':'one immutable Bright index snapshot for all cases; independent MuSiQue context per case',
        'musique_v3_amendment':'v2 completed one variant, then source-scope protection rejected the second directory in the same vault database. v3 uses one fresh SQLite database per variant. The failed v2 attempt is retained; all 200 registered variants restart in a new run. No prompt, model, budget, selection, source identity or production source changes.' if a.track=='musique' else None,
        'musique_context_indexing':{'unit':'one whole original paragraph per opaque Markdown source; original 20-paragraph context per variant',
            'chunking':'none','batch_size':32,'scope':'Each variant has a separate SQLite database and active snapshot. Embedding caches are not shared between variants. Answers, answerability and support labels are scoring-side only.',
            'cost':'Per-case context indexing elapsed time and BuildReport recorded separately from Agent elapsed time and the run_agent hard deadline.'} if a.track=='musique' else None,
        'chat_context':'Uses current product request options (temperature=0) and server defaults; effective loaded context_length retained after each attempt. Submitted evidence is not proof of provider consumption when requests fail or the provider truncates context.',
        'failure_prediction':'null answerability cannot earn correct abstention credit',
        'release_eligible':False,'protocol_scope':'ARKB agent diagnostic, not replication of benchmark paper agent'})
    cmd=[sys.executable,'-u',str(a.output/'run_p4_agents.py'),'--execute','--output',str(a.output),
         '--track',a.track,'--qdrant-url',a.qdrant_url]
    if a.dataset:cmd+=['--dataset',str(a.dataset),'--index-run',str(a.index_run)]
    env={**os.environ,'PYTHONPATH':str(a.output/'measured-source/src'),'PYTHONDONTWRITEBYTECODE':'1',
         'TOKENIZERS_PARALLELISM':'false','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
    with (a.output/'execution.log').open('x') as log:result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
    print(json.dumps({'output':str(a.output),'exit_code':result.returncode}));raise SystemExit(result.returncode)


if __name__=='__main__':main()
