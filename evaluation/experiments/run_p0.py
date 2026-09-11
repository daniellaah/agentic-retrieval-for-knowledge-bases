"""Freeze code/corpus, rebuild schema v2, and run the preregistered P0 matrix.

No existing index or collection is modified. Results never overwrite a run.
The child uses frozen Python sources, allowing later development during inference.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str, allow_nan=False) + '\n')


def read(path):
    return json.loads(path.read_text())


def json_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def execute(output, host, qdrant_url, cache, reranker_cache):
    import httpx
    import numpy as np
    import torch
    import arkb
    from arkb.config import RuntimeConfig, RetrievalConfig
    from arkb.evaluation.models import AgentEvalConfig, BaselineEvalConfig
    from arkb.evaluation.runs import run_agent_evaluation, run_baseline_evaluation
    from arkb.evaluation.agent_metrics import evaluate_case, summarize_agent_results
    from arkb.evaluation.baselines import source_metrics
    from arkb.agent.state import AgentTrace, AgentToolTrace
    from arkb.evaluation.datasets import load_agent_eval_dataset
    from arkb.knowledge.sqlite import SQLiteStorage
    from arkb.knowledge.qdrant import QdrantIndex
    from arkb.runtime import Runtime

    if not Path(arkb.__file__).resolve().is_relative_to(output / 'measured-source'):
        raise RuntimeError('Frozen source package was not selected.')
    protocol = read(output / 'protocol.json')
    torch.set_num_threads(protocol['torch_threads'])
    notes, db = output / 'corpus', output / 'index.sqlite'
    vault = output.name
    config = RuntimeConfig(host=host, qdrant_url=qdrant_url, offline=True, tokenizer_cache=cache,
                           embedding_model=protocol['embedding_model'])
    corpus_hashes = lambda: {p.name: digest(p) for p in sorted(notes.glob('*.md'))}
    def models():
        with httpx.Client(trust_env=False, timeout=10) as c:
            tags = c.get(host + '/api/tags'); tags.raise_for_status()
            return {m['name']: m['digest'] for m in tags.json()['models']
                    if m['name'] in (protocol['embedding_model'], protocol['agent_model'])}
    before_models = models()
    if len(before_models) != 2:
        raise RuntimeError('Required models unavailable; no substitutions allowed.')
    metadata = read(output / 'experiment.json')
    metadata.update(status='running', models_before=before_models, corpus_before=corpus_hashes(),
                    packages={n: version(n) for n in ('numpy','ollama','qdrant-client','torch','transformers')},
                    platform=platform.platform(), python=platform.python_version())
    with httpx.Client(trust_env=False, timeout=10) as c:
        metadata['ollama_version'] = c.get(host + '/api/version').json()
        metadata['qdrant_version'] = c.get(qdrant_url + '/').json()
        metadata['provider_model_info'] = {name: c.post(host + '/api/show', json={'model': name}).json()
                                           for name in before_models}
    write(output / 'experiment.json', metadata)
    try:
        with Runtime(config) as runtime:
            build = runtime.index(db=db, notes_dir=notes, vault_id=vault,
                chunking=protocol['chunking'], chunk_size=protocol['chunk_size'],
                chunk_overlap=protocol['chunk_overlap'], context_length=protocol['embedding_context_length'])
            write(output / 'index-report.json', asdict(build))
            def snapshot():
                with SQLiteStorage(db, read_only=True) as store:
                    manifest = store.active_manifest(vault)
                    _, records, vectors = store.load_snapshot(manifest.index_version)
                    backend = store.build_metadata(manifest.index_version)['backend']
                    index = QdrantIndex(runtime.qdrant_client(qdrant_url), backend['collection'],
                                        manifest.embedding_spec, vault_id=vault)
                    index.verify_snapshot(records, vectors)
                    return {'manifest': asdict(manifest), 'collection': backend['collection'],
                            'vectors_sha256': hashlib.sha256(np.asarray(vectors).tobytes()).hexdigest(),
                            'corpus': [(r.document_id, r.document_revision, r.chunk_id) for r in records]}
            before = snapshot()
            write(output / 'snapshot-before.json', before)
            print('Index built and vectors verified.', flush=True)
            baseline = run_baseline_evaluation(BaselineEvalConfig(
                dataset_path=output / 'cases.jsonl', output_dir=output / 'baselines', db=db,
                notes_dir=notes, vault_id=vault, index_version=build.manifest.index_version,
                runtime_config=config, baselines=tuple(protocol['baseline_methods']),
                top_k=protocol['top_k_chunks'], exact=protocol['qdrant_exact'],
                retrieval_config=RetrievalConfig(reranker_cache=str(reranker_cache),
                    candidate_k=protocol['candidate_k'], rrf_k=protocol['rrf_k'],
                    rerank_candidates=protocol['rerank_candidates'], reranker_max_length=protocol['reranker_max_length'],
                    bm25_k1=protocol['bm25_k1'], bm25_b=protocol['bm25_b'])), runtime=runtime)
            print('Baselines complete; starting 120 Agent trials.', flush=True)
            agent = run_agent_evaluation(AgentEvalConfig(
                dataset_path=output / 'cases.jsonl', output_dir=output / 'agent-4b',
                db=db, notes_dir=notes, vault_id=vault, model=protocol['agent_model'],
                max_turns=protocol['agent_max_turns'], num_trials=protocol['agent_trials'],
                think=protocol['agent_think'], runtime_config=config), runtime=runtime)
            after = snapshot()
            write(output / 'snapshot-after.json', after)
        cases = {c.id: c for c in load_agent_eval_dataset(output / 'cases.jsonl', notes_dir=notes)}
        agent_rows = json_rows(output / 'agent-4b/results.jsonl')
        if {(r['case']['id'],r['trial']) for r in agent_rows} != {(c,t) for c in cases for t in range(3)} or len(agent_rows) != 120:
            raise ValueError('Incomplete Agent matrix.')
        replayed = []
        for row in agent_rows:
            payload = row['trace']
            trace = AgentTrace(**{**payload, 'tool_calls':[AgentToolTrace(**c) for c in payload['tool_calls']]}) if payload else None
            result = evaluate_case(cases[row['case']['id']], trace)
            if json.loads(json.dumps(asdict(result))) != row['metrics']:
                raise ValueError('Agent replay mismatch.')
            replayed.append(result)
        baseline_rows = json_rows(output / 'baselines/results.jsonl')
        if len(baseline_rows) != 160 or {(r['case_id'],r['baseline']) for r in baseline_rows} != {(c,m) for c in cases for m in protocol['baseline_methods']}:
            raise ValueError('Incomplete baseline matrix.')
        for row in baseline_rows:
            if row['status'] == 'ok':
                sources = tuple(dict.fromkeys(h['source'] for h in row['response']['results']))
                if source_metrics(cases[row['case_id']].expected_sources, sources, ks=(1,3,5,10)) != row['metrics']:
                    raise ValueError('Baseline replay mismatch.')
        after_models = models()
        metadata.update(models_after=after_models, corpus_after=corpus_hashes())
        if before != after or metadata['corpus_before'] != metadata['corpus_after'] or before_models != after_models:
            raise ValueError('Experiment inputs drifted.')
        replay_summary = summarize_agent_results(replayed)
        if any(agent.summary[k] != v for k,v in replay_summary.items()):
            raise ValueError('Agent summary replay mismatch.')
        write(output / 'validation.json', {'valid': True, 'agent_rows_replayed':120,
            'baseline_rows_checked':160,'snapshot_vectors_verified':True,'input_drift':False})
        write(output / 'comparison.json', {'protocol':protocol, 'agent':agent.summary,
                                         'baselines':baseline.summary,'budget_matched':False})
        metadata['status'] = 'completed'
    except BaseException as error:
        metadata.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                        error={'type':type(error).__name__,'message':str(error)})
        raise
    finally:
        metadata['finished_at'] = datetime.now(timezone.utc).isoformat()
        write(output / 'experiment.json', metadata)
        # Checksums cover retained files; SQLite WAL/SHM/lock files are runtime state.
        hashes = {p.relative_to(output).as_posix(): digest(p) for p in sorted(output.rglob('*'))
                  if p.is_file() and p.name not in ('checksums.json','execution.log')
                  and not p.name.endswith(('-wal','-shm','.lock','.pyc'))}
        write(output / 'checksums.json', hashes)
    print('P0 complete and replay verified.', flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--host',default='http://127.0.0.1:11434')
    parser.add_argument('--qdrant-url',required=True)
    parser.add_argument('--execute',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--tokenizer-cache',type=Path,default=Path('.uv-cache/tokenizers'))
    parser.add_argument('--reranker-cache',type=Path,default=Path('.arkb/models'))
    args=parser.parse_args()
    output=args.output.resolve()
    if args.execute:
        execute(output,args.host,args.qdrant_url,args.tokenizer_cache.resolve(),args.reranker_cache.resolve())
        return
    root=Path(__file__).resolve().parents[2]
    protocol_path=Path(__file__).with_name('p0-protocol.json')
    protocol=read(protocol_path)
    if digest(root/protocol['dataset']) != protocol['dataset_sha256']:
        raise ValueError('v1 dataset differs from preregistration.')
    output.mkdir(parents=True,exist_ok=False)
    (output/'corpus').mkdir()
    for path in sorted((root/protocol['corpus']).glob('*.md')):
        shutil.copyfile(path,output/'corpus'/path.name)
        (output/'corpus'/path.name).chmod(0o444)
    shutil.copyfile(root/protocol['dataset'],output/'cases.jsonl')
    shutil.copyfile(protocol_path,output/'protocol.json')
    shutil.copytree(root/'src',output/'measured-source/src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('uv.lock','pyproject.toml'):
        shutil.copyfile(root/name,output/'measured-source'/name)
    shutil.copyfile(__file__,output/'run_p0.py')
    write(output/'experiment.json',{'status':'prepared','created_at':datetime.now(timezone.utc).isoformat(),
        'source_commit':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        'protocol_sha256':digest(protocol_path),'runner_sha256':digest(Path(__file__)),
        'host':args.host,'qdrant_url':args.qdrant_url})
    env={**os.environ,'PYTHONPATH':str(output/'measured-source/src'),'PYTHONDONTWRITEBYTECODE':'1',
         'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false'}
    command=[sys.executable,'-u',str(output/'run_p0.py'),'--execute','--output',str(output),
        '--host',args.host,'--qdrant-url',args.qdrant_url,'--tokenizer-cache',str(args.tokenizer_cache.resolve()),
        '--reranker-cache',str(args.reranker_cache.resolve())]
    with (output/'execution.log').open('x') as log:
        result=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT)
    print(json.dumps({'output':str(output),'exit_code':result.returncode}))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
