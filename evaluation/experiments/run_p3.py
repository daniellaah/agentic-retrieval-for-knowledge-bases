"""Freeze and execute the P3 candidate-depth and stopping interventions."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text())
def write(path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str)+'\n')
def emit(stream, row): stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n'); stream.flush()


def execute(output, args):
    import httpx
    import numpy as np
    import arkb
    from arkb.agent.observation import AgentBudget
    from arkb.config import RuntimeConfig, RetrievalConfig
    from arkb.runtime import Runtime
    from arkb.knowledge.sqlite import SQLiteStorage
    from arkb.knowledge.qdrant import QdrantIndex
    from arkb.evaluation.v2 import load_dataset, dataset_report
    from arkb.evaluation.outputs import score_agent, score_agent_output, pending_review
    from arkb.evaluation.controlled import fused_candidates, component_rows, stopping_diagnostics, stopping_order
    from arkb.evaluation.gates import evaluation_gate
    from arkb.evaluation.runs import _code_metadata

    if not Path(arkb.__file__).resolve().is_relative_to(output/'measured-source/src'):
        raise ValueError('Expected frozen measured sources.')
    protocol = read(output/'protocol.json'); metadata = read(output/'experiment.json')
    dataset = load_dataset(output/'dataset', notes_dir=output/'dataset/corpus', allow_provisional=True)
    cases = {c['id']: c for c in dataset.cases}; component = protocol['component']; stop = protocol['stopping']
    selected = [c for c in dataset.cases if c['task_type'] not in component['excluded_tasks']]
    if (digest(output/'dataset/manifest.json') != protocol['dataset_manifest_sha256']
            or len(selected) != component['planned_queries']
            or len(selected)*len(component['arms']) != component['planned_rows']
            or len(set(stop['cases'])) != len(stop['cases']) or set(stop['cases'])-cases.keys()
            or len(stop['cases'])*len(stop['arms'])*stop['trials'] != stop['planned_rows']):
        raise ValueError('Protocol/dataset matrix mismatch.')
    source_before = _code_metadata()
    def services():
        with httpx.Client(trust_env=False, timeout=15) as client:
            tags = client.get(args.host+'/api/tags'); tags.raise_for_status()
            needed = {protocol[k] for k in ('agent_model', 'embedding_model')}
            models = {m['name']: m['digest'] for m in tags.json()['models'] if m['name'] in needed}
            if models != protocol['model_digests']: raise ValueError('Registered model missing or changed.')
            return {'models': models, 'ollama': client.get(args.host+'/api/version').json(),
                    'qdrant': client.get(args.qdrant_url+'/').json()}
    raw_inputs, retrieval_rows, agent_rows = [], [], []
    metadata.update(status='running', started_at=datetime.now(timezone.utc).isoformat(),
                    source=source_before, dataset=dataset_report(dataset))
    write(output/'experiment.json', metadata)
    try:
        metadata['services_before'] = services()
        metadata['packages'] = {n: version(n) for n in ('numpy','ollama','qdrant-client','tokenizers','torch','transformers')}
        config = RuntimeConfig(host=args.host, qdrant_url=args.qdrant_url, offline=True,
                               tokenizer_cache=args.tokenizer_cache.resolve(), embedding_model=protocol['embedding_model'])
        with Runtime(config) as runtime:
            (output/'reference-tokenizer.json').write_text(runtime.tokenizer().to_str())
            build = runtime.index(db=output/'index.sqlite', vault_id=output.name, notes_dir=output/'dataset/corpus')
            write(output/'index-report.json', asdict(build))
            with SQLiteStorage(output/'index.sqlite', read_only=True) as storage:
                def snapshot():
                    manifest, records, vectors = storage.load_snapshot(build.manifest.index_version)
                    backend = storage.build_metadata(manifest.index_version)['backend']
                    index = QdrantIndex(runtime.qdrant_client(args.qdrant_url), backend['collection'], manifest.embedding_spec, vault_id=output.name)
                    index.verify_snapshot(records, vectors)
                    return {'manifest': asdict(manifest), 'collection': backend['collection'],
                            'vectors_sha256': hashlib.sha256(np.asarray(vectors).tobytes()).hexdigest()}
                metadata['snapshot_before'] = snapshot()
                engine = runtime.retrieval_engine(storage, build.manifest, modes=('bm25','semantic'), rerank=True,
                    exact=True, settings=RetrievalConfig(reranker_cache=str(args.reranker_cache.resolve()),
                    reranker_max_length=component['reranker_max_length']))
                scorer = engine.reranker.scorer
                metadata['reranker_identity'] = scorer.identity
                from arkb.retrieval.qwen_rerank import QWEN_MODEL, QWEN_REVISION
                model_dir = args.reranker_cache.resolve()/('models--'+QWEN_MODEL.replace('/','--'))/'snapshots'/QWEN_REVISION
                def weights():
                    files = {p.relative_to(model_dir).as_posix(): digest(p) for p in sorted(model_dir.rglob('*')) if p.is_file()}
                    if not any(n.endswith('.safetensors') for n in files): raise ValueError('Missing pinned reranker weights.')
                    return files
                metadata['reranker_files_before'] = weights()
                with (output/'candidate-inputs.jsonl').open('x') as raw_stream, (output/'component-results.jsonl').open('x') as result_stream:
                    for case in selected:
                        raw = {'case_id': case['id'], 'query': case['query'], 'legs': {}, 'leg_elapsed_ms': {}}
                        for mode in ('bm25','semantic'):
                            started = perf_counter()
                            response = engine.search(case['query'], mode=mode, top_k=component['max_leg_depth'])
                            if response.index_id != build.manifest.index_version: raise ValueError('Wrong retrieval snapshot.')
                            raw['legs'][mode] = [asdict(h) for h in response.results]
                            raw['leg_elapsed_ms'][mode] = (perf_counter()-started)*1000
                        _, pool = fused_candidates(raw['legs'], component['rerank_leg_depth'],
                                    pool_size=component['fusion_pool_size'], rrf_k=component['rrf_k'])
                        started = perf_counter(); scores = list(scorer.score(case['query'], pool))
                        raw['rerank'] = {'candidates': [asdict(h) for h in pool], 'scores': scores,
                            'identity': scorer.identity, 'score_type': scorer.score_type, 'elapsed_ms': (perf_counter()-started)*1000}
                        emit(raw_stream, raw); raw_inputs.append(raw)
                        for row in component_rows(raw, case, dataset, protocol):
                            emit(result_stream, row); retrieval_rows.append(row)
                        print(f'Component {case["id"]}: {len(retrieval_rows)}/{component["planned_rows"]}', flush=True)
                metadata['reranker_files_after'] = weights()
                # Release CPU reranker before Agent inference; no concurrent model workloads.
                del scorer, engine
                with (output/'agent-results.jsonl').open('x') as stream:
                    for cid, arm, trial in stopping_order(stop):
                        started = perf_counter(); error = None
                        observer = runtime.agent_observer(budget=AgentBudget(**stop['budget']))
                        try:
                            result = runtime.ask(cases[cid]['query'], db=output/'index.sqlite', vault_id=output.name,
                                notes_dir=output/'dataset/corpus', model=protocol['agent_model'], max_turns=stop['max_turns'],
                                think=stop['think'], observer=observer, search_stall_reminder=stop['arms'][arm])
                        except Exception as failure:
                            error = {'type': type(failure).__name__, 'message': str(failure)}
                            result = getattr(failure, 'agent_result', None)
                            if result is None: raise RuntimeError('Missing observed partial result; experiment invalid.') from failure
                        row = {'case_id': cid, 'arm': arm, 'trial': trial, 'stop_reason': result.stop_reason,
                            'error': error, 'elapsed_ms': (perf_counter()-started)*1000, 'observation': result.observation,
                            **score_agent(cases[cid], result, dataset),
                            'stopping_diagnostics': stopping_diagnostics(cases[cid], result.observation, dataset)}
                        emit(stream, row); agent_rows.append(row)
                        print(f'Agent {cid} {arm} trial={trial}: {result.stop_reason} ({len(agent_rows)}/{stop["planned_rows"]})', flush=True)
                metadata['snapshot_after'] = snapshot()
        metadata['services_after'] = services()
        load_dataset(output/'dataset', notes_dir=output/'dataset/corpus', allow_provisional=True)
        stable = (metadata['snapshot_before'] == metadata['snapshot_after'] and metadata['services_before'] == metadata['services_after']
            and metadata['reranker_files_before'] == metadata['reranker_files_after'] and _code_metadata() == source_before
            and digest(output/'dataset/manifest.json') == protocol['dataset_manifest_sha256'])
        replay = [row for raw in raw_inputs for row in component_rows(raw, cases[raw['case_id']], dataset, protocol)]
        if replay != retrieval_rows: raise ValueError('Component replay mismatch.')
        for row in agent_rows:
            score = score_agent_output(cases[row['case_id']], row['trace']['final_response'], row['stop_reason'], row['observation'], dataset)
            if any(row[k] != v for k,v in score.items()): raise ValueError('Agent metric replay mismatch.')
        expected = {(cid,arm,t) for cid in stop['cases'] for arm in stop['arms'] for t in range(stop['trials'])}
        gate = evaluation_gate(agent_rows, expected, inputs_stable=stable, replay_valid=True, dataset_provisional=dataset.provisional)
        if not gate['experiment_valid'] or len(retrieval_rows) != component['planned_rows']: raise ValueError('Incomplete or drifting experiment.')
        write(output/'gate.json', gate)
        write(output/'output-review.json', [{'case_id': r['case_id'], 'arm': r['arm'], 'trial': r['trial'],
            'packet': r['packet'], 'review': pending_review(r['packet'])} for r in agent_rows])
        write(output/'validation.json', {'valid': True, 'component_rows_replayed': len(retrieval_rows), 'agent_rows_replayed': len(agent_rows),
            'input_drift': False, 'snapshot_vectors_verified': True, 'release_eligible': False})
        metadata.update(status='completed', component_rows=len(retrieval_rows), agent_rows=len(agent_rows))
    except BaseException as error:
        metadata.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed', error={'type':type(error).__name__, 'message':str(error)})
        raise
    finally:
        metadata['finished_at'] = datetime.now(timezone.utc).isoformat(); write(output/'experiment.json', metadata)
        write(output/'checksums.json', {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob('*'))
            if p.is_file() and p.name not in ('checksums.json','execution.log') and not p.name.endswith(('-wal','-shm','.pyc','.lock'))})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--qdrant-url', required=True)
    p.add_argument('--host', default='http://127.0.0.1:11434')
    p.add_argument('--tokenizer-cache', type=Path, default=Path('.uv-cache/tokenizers'))
    p.add_argument('--reranker-cache', type=Path, default=Path('.arkb/models'))
    p.add_argument('--execute', action='store_true', help=argparse.SUPPRESS)
    a = p.parse_args(); output = a.output.resolve()
    if a.execute: execute(output,a); return
    root = Path(__file__).resolve().parents[2]; protocol = Path(__file__).with_name('p3-protocol.json')
    dataset = root/read(protocol)['dataset']
    if digest(dataset/'manifest.json') != read(protocol)['dataset_manifest_sha256']: raise ValueError('Unregistered dataset revision.')
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(dataset, output/'dataset')
    shutil.copytree(root/'src', output/'measured-source/src', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('pyproject.toml','uv.lock'): shutil.copyfile(root/name, output/'measured-source'/name)
    shutil.copyfile(protocol, output/'protocol.json'); shutil.copyfile(__file__, output/'run_p3.py')
    write(output/'experiment.json', {'status':'prepared','protocol_version':read(protocol)['protocol_version'],
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'protocol_sha256':digest(protocol),'runner_sha256':digest(Path(__file__))})
    command = [sys.executable,'-u',str(output/'run_p3.py'),'--execute','--output',str(output),
        '--qdrant-url',a.qdrant_url,'--host',a.host,'--tokenizer-cache',str(a.tokenizer_cache.resolve()),
        '--reranker-cache',str(a.reranker_cache.resolve())]
    env = {**os.environ,'PYTHONPATH':str(output/'measured-source/src'),'PYTHONDONTWRITEBYTECODE':'1',
           'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false'}
    with (output/'execution.log').open('x') as log: result = subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    print(json.dumps({'output':str(output),'exit_code':result.returncode})); raise SystemExit(result.returncode)


if __name__ == '__main__': main()
