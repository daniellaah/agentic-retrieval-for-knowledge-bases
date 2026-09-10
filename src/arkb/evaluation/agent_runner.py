"""Run agent evaluation through Runtime.ask and preserve inspectable artifacts."""

import argparse
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from typing import Protocol
from uuid import uuid4

from arkb.agent.state import AgentResult
from arkb.config import RetrievalConfig, RuntimeConfig
from arkb.evaluation.agent import evaluate_case, summarize_agent_results
from arkb.evaluation.datasets import corpus_manifest, parse_agent_eval_dataset, source_hashes
from arkb.evaluation.models import (
    BASELINE_MODES, AgentEvalConfig, AgentEvalRun, AgentEvalTrial,
    BaselineEvalConfig, BaselineEvalRun,
)
from arkb.runtime import Runtime


SCHEMA_VERSION = 'agent-evaluation-v1'


class AgentEvalRuntime(Protocol):
    def ask(self, query: str, **options) -> AgentResult: ...


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Cannot serialize {type(value).__name__} as evaluation JSON.')


def _json(value, *, indent=None) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=indent, default=_json_default)


def _write_json(path: Path, value):
    path.write_text(_json(value, indent=2) + '\n', encoding='utf-8')


def _knowledge_metadata(config: AgentEvalConfig | BaselineEvalConfig, *, index_version=None) -> dict:
    """Record live file hashes and the published snapshot without loading models."""
    from arkb.knowledge.sqlite import SQLiteStorage

    root = config.notes_dir.resolve()
    notes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(root.glob('*.md'))
             if p.is_file() and p.resolve().is_relative_to(root)}
    snapshot = None
    if config.db.exists():
        with SQLiteStorage(config.db, read_only=True) as storage:
            manifest = storage.get_manifest(index_version) if index_version is not None else storage.active_manifest(config.vault_id)
            if manifest is not None:
                snapshot = {
                    'manifest': asdict(manifest),
                    'build_metadata': storage.build_metadata(manifest.index_version),
                    'corpus': corpus_manifest(storage.snapshot_records(manifest.index_version)),
                }
    return {'notes_sha256': notes, 'snapshot': snapshot}


def _code_metadata() -> dict:
    package = Path(__file__).resolve().parent.parent
    try:
        git = subprocess.run(['git', '-C', str(package), 'rev-parse', 'HEAD'],
                             capture_output=True, text=True, check=False, timeout=5)
        commit = git.stdout.strip() if git.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        commit = None
    return {'source_commit': commit, 'source_hashes': source_hashes(package),
            'python': platform.python_version()}


def _run_trial(case, trial, *, runtime, config, metadata, client):
    started_at, started = _now(), perf_counter()
    trace, error = None, None
    options = {'db': config.db, 'vault_id': config.vault_id, 'notes_dir': config.notes_dir,
               'model': config.model, 'max_turns': config.max_turns, 'think': config.think}
    if client is not None:
        options['client'] = client
    try:
        trace = runtime.ask(case.query, **options).trace
    except Exception as failure:
        partial = getattr(failure, 'agent_result', None)
        if isinstance(partial, AgentResult):
            trace = partial.trace
        error = {'type': type(failure).__name__, 'message': str(failure)}
    metrics = evaluate_case(case, trace)
    return AgentEvalTrial(
        case=case, trial=trial, trace=trace, metrics=metrics, error=error,
        runtime_metadata={
            'run_id': metadata['run_id'], 'run_metadata_file': 'run_metadata.json',
            'dataset_sha256': metadata['dataset_sha256'], 'model': config.model,
            'max_turns': config.max_turns, 'think': config.think,
            'runtime_type': metadata['runtime_type'], 'started_at': started_at,
            'elapsed_ms': (perf_counter() - started) * 1000,
        },
    )


def _failed_trial(row: AgentEvalTrial) -> dict:
    return {'case_id': row.case.id, 'trial': row.trial, 'query': row.case.query,
            'task_type': row.case.task_type, 'expected_sources': row.metrics.expected_sources,
            'retrieved_sources': row.metrics.retrieved_sources, 'tool_sequence': row.metrics.tool_calls,
            'stop_reason': row.metrics.stop_reason, 'failure_reasons': row.metrics.failure_reasons,
            'error': row.error}


def run_agent_evaluation(
    config: AgentEvalConfig, *, runtime: AgentEvalRuntime | None = None, client=None,
    trial_metadata: dict | None = None,
    trial_annotations: Callable[[AgentEvalTrial], dict] | None = None,
) -> AgentEvalRun:
    """Run every case/trial with fresh runtime conversation state and no retries.

    Own a Runtime only when one is not supplied. Inputs validate before any ask.
    Ordinary per-trial runtime exceptions retain partial traces and do not stop
    later trials. Interruptions and artifact I/O errors propagate; each completed
    row is flushed, and metadata marks incomplete runs instead of reporting success.
    Optional experiment metadata and deterministic annotations are namespaced in
    saved rows; neither is passed to Runtime or used in v1 scoring.
    """
    raw = config.dataset_path.read_bytes()
    cases = parse_agent_eval_dataset(raw, notes_dir=config.notes_dir)
    output = (config.output_dir if config.output_dir is not None else
              Path('evaluation/results') / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ-') + uuid4().hex[:8]))
    if output.exists():
        raise FileExistsError(f'Evaluation output must be a new directory: {output}')
    knowledge = _knowledge_metadata(config)
    metadata = {
        'schema_version': SCHEMA_VERSION, 'run_id': output.name, 'status': 'running',
        'started_at': _now(), 'config': asdict(config),
        'working_directory': str(Path.cwd()), 'output_dir': output.resolve(),
        'dataset_sha256': hashlib.sha256(raw).hexdigest(),
        'knowledge_before': knowledge, **_code_metadata(),
        'configured_trials': len(cases) * config.num_trials,
        'completed_trials': 0,
    }
    output.mkdir(parents=True, exist_ok=False)
    results = []
    try:
        with ExitStack() as resources:
            if runtime is None:
                runtime = resources.enter_context(Runtime(config.runtime_config))
            metadata['runtime_type'] = f'{type(runtime).__module__}.{type(runtime).__qualname__}'
            actual_config = getattr(runtime, 'config', None)
            metadata['effective_runtime_config'] = asdict(actual_config) if isinstance(actual_config, RuntimeConfig) else None
            (output / 'cases.jsonl').write_bytes(raw)
            _write_json(output / 'run_metadata.json', metadata)
            with (output / 'results.jsonl').open('x', encoding='utf-8') as stream:
                for case in cases:
                    for trial in range(config.num_trials):
                        row = _run_trial(case, trial, runtime=runtime, config=config,
                                         metadata=metadata, client=client)
                        if trial_metadata:
                            # Experiment context is namespaced; it cannot override v1 fields.
                            row.runtime_metadata['experiment'] = dict(trial_metadata)
                        payload = {'schema_version': SCHEMA_VERSION, **asdict(row)}
                        if trial_annotations is not None:
                            payload['analysis'] = trial_annotations(row)
                        stream.write(_json(payload) + '\n')
                        stream.flush()
                        results.append(row)
            summary = {
                'schema_version': SCHEMA_VERSION, 'run_id': metadata['run_id'],
                **summarize_agent_results([row.metrics for row in results]),
                'failed_trials': [_failed_trial(row) for row in results if not row.metrics.success],
            }
            metadata['knowledge_after'] = _knowledge_metadata(config)
            metadata['knowledge_changed'] = metadata['knowledge_after'] != knowledge
            summary['knowledge_changed'] = metadata['knowledge_changed']
            _write_json(output / 'summary.json', summary)
            (output / 'report.md').write_text(render_agent_report(summary), encoding='utf-8')
        metadata['status'] = 'completed'
    except BaseException as failure:
        metadata['status'] = 'interrupted' if isinstance(failure, KeyboardInterrupt) else 'failed'
        raise
    finally:
        metadata.update(completed_trials=len(results), finished_at=_now())
        _write_json(output / 'run_metadata.json', metadata)
    return AgentEvalRun(output_dir=output, results=tuple(results), summary=summary)


def run_baseline_evaluation(config: BaselineEvalConfig, *, runtime=None) -> BaselineEvalRun:
    """Use the v1 dataset/artifact lifecycle with fixed RetrievalEngine requests.

    Capture a ready snapshot once and prepare only selected capabilities. Setup
    errors fail the run; per-case retrieval errors remain structured results.
    Injected runtimes remain caller-owned and only retrieval_engine is invoked.
    """
    from arkb.evaluation.baselines import (
        SCHEMA_VERSION as baseline_schema, execute_baseline, render_baseline_report, summarize_baselines,
    )
    from arkb.knowledge.documents import DocumentAccess
    from arkb.knowledge.sqlite import SQLiteStorage

    raw = config.dataset_path.read_bytes()
    cases = parse_agent_eval_dataset(raw, notes_dir=config.notes_dir)
    output = config.output_dir or Path('evaluation/results') / (
        datetime.now(timezone.utc).strftime('baselines-%Y%m%dT%H%M%S%fZ-') + uuid4().hex[:8])
    if output.exists():
        raise FileExistsError(f'Evaluation output must be a new directory: {output}')
    with ExitStack() as resources:
        storage = resources.enter_context(SQLiteStorage(config.db, read_only=True))
        manifest = (storage.get_manifest(config.index_version) if config.index_version is not None
                    else storage.active_manifest(config.vault_id))
        if manifest is None or manifest.status != 'ready' or manifest.vault_id != config.vault_id:
            raise ValueError('Baseline evaluation requires a ready snapshot in the requested vault.')
        records = storage.snapshot_records(manifest.index_version)
        known_sources = {r.chunk.source for r in records}
        if any(set(case.expected_sources) - known_sources for case in cases):
            raise ValueError('Expected sources are outside the pinned snapshot.')
        scope = storage.build_metadata(manifest.index_version)['backend'].get('source_scope')
        if scope is not None and Path(scope).resolve() != config.notes_dir.resolve():
            raise ValueError('notes_dir differs from the indexed knowledge base.')
        # Preflight only: the same document loader/revision contract as live
        # Agent evidence. No read tool is invoked, and no label enters the engine.
        live = {(r.document_id, r.document_revision)
                for r in DocumentAccess(config.notes_dir, vault_id=config.vault_id).records()}
        if live != {(r.document_id, r.document_revision) for r in records}:
            raise ValueError('Live knowledge differs from the pinned snapshot; align inputs before comparing.')
        knowledge = _knowledge_metadata(config, index_version=manifest.index_version)
        metadata = {
            'schema_version': baseline_schema, 'run_id': output.name, 'status': 'running',
            'started_at': _now(), 'config': asdict(config), 'index_version': manifest.index_version,
            'working_directory': str(Path.cwd()), 'output_dir': output.resolve(),
            'dataset_sha256': hashlib.sha256(raw).hexdigest(), 'knowledge_before': knowledge,
            'configured_results': len(cases) * len(config.baselines), 'completed_results': 0,
            'method_order': 'selected baseline order rotated once per dataset case; sequential execution',
            'source_constraints': 'entire pinned vault; no case filter field in AgentEvalCase; labels are never filters',
            'baseline_modes': {name: {'mode': BASELINE_MODES[name][0], 'rerank': BASELINE_MODES[name][1]}
                               for name in config.baselines}, **_code_metadata(),
        }
        output.mkdir(parents=True, exist_ok=False)
        results = []
        try:
            (output / 'cases.jsonl').write_bytes(raw)
            _write_json(output / 'run_metadata.json', metadata)
            started = perf_counter()
            if runtime is None:
                runtime = resources.enter_context(Runtime(config.runtime_config))
            metadata['runtime_type'] = f'{type(runtime).__module__}.{type(runtime).__qualname__}'
            actual_config = getattr(runtime, 'config', None)
            metadata['effective_runtime_config'] = asdict(actual_config) if isinstance(actual_config, RuntimeConfig) else None
            engine = None
            if any(case.task_type != 'no_retrieval' for case in cases):
                modes = tuple(dict.fromkeys(BASELINE_MODES[name][0] for name in config.baselines))
                engine = runtime.retrieval_engine(storage, manifest, modes=modes,
                    rerank='hybrid_rerank' in config.baselines, settings=config.retrieval_config,
                    exact=config.exact, records=records)
            metadata['setup_ms'] = (perf_counter() - started) * 1000
            reranker = getattr(engine, 'reranker', None)
            metadata['reranker_identity'] = reranker.scorer.identity if reranker is not None else None
            _write_json(output / 'run_metadata.json', metadata)
            with (output / 'results.jsonl').open('x', encoding='utf-8') as stream:
                for i, case in enumerate(cases):
                    offset = i % len(config.baselines)
                    for name in config.baselines[offset:] + config.baselines[:offset]:
                        row = execute_baseline(case, name, engine=engine, top_k=config.top_k,
                            ks=config.metric_ks, index_version=manifest.index_version, known_sources=known_sources)
                        stream.write(_json({'schema_version': baseline_schema, 'run_id': output.name, **asdict(row)}) + '\n')
                        stream.flush()
                        results.append(row)
            metadata['knowledge_after'] = _knowledge_metadata(config, index_version=manifest.index_version)
            metadata['knowledge_changed'] = metadata['knowledge_after'] != knowledge
            summary = {'schema_version': baseline_schema, 'run_id': output.name,
                       **summarize_baselines(results, config=config), 'knowledge_changed': metadata['knowledge_changed']}
            _write_json(output / 'summary.json', summary)
            (output / 'report.md').write_text(render_baseline_report(summary), encoding='utf-8')
            metadata['status'] = 'invalidated' if metadata['knowledge_changed'] else 'completed'
        except BaseException as failure:
            metadata['status'] = 'interrupted' if isinstance(failure, KeyboardInterrupt) else 'failed'
            metadata['error'] = {'type': type(failure).__name__, 'message': str(failure)}
            raise
        finally:
            metadata.update(completed_results=len(results), finished_at=_now())
            _write_json(output / 'run_metadata.json', metadata)
    return BaselineEvalRun(output_dir=output, results=tuple(results), summary=summary)


def run_evaluation(config: AgentEvalConfig | BaselineEvalConfig, **options):
    """Select v1 Agent execution or fixed baselines without mixing their metrics."""
    if isinstance(config, BaselineEvalConfig):
        return run_baseline_evaluation(config, **options)
    if isinstance(config, AgentEvalConfig):
        return run_agent_evaluation(config, **options)
    raise TypeError('Expected AgentEvalConfig or BaselineEvalConfig.')


def _cell(value) -> str:
    if value is None:
        return 'unavailable / undefined'
    if isinstance(value, (tuple, list)):
        value = ', '.join(value) if value else '(none)'
    return escape(str(value)).replace('|', '\\|').replace('\n', '<br>')


def render_agent_report(summary: dict) -> str:
    """Human-readable aggregate and explicit failed-trial details; no judge."""
    lines = ['# Agent Evaluation v1', '', f'Run: {_cell(summary["run_id"])}', '',
             'Source coverage and tool behavior only; answer quality is not graded.', '',
             '| Metric | Value |', '| --- | ---: |']
    for key in ('total_cases', 'total_trials', 'task_success_rate', 'average_source_recall',
                'average_tool_calls', 'average_turns', 'max_turn_failure_rate',
                'unnecessary_retrieval_rate', 'missing_trace_trials'):
        value = summary[key]
        lines.append(f'| {key} | {value:.4f} |' if isinstance(value, float) else f'| {key} | {_cell(value)} |')
    lines += ['', 'Recall excludes no-retrieval and unavailable traces. Unnecessary retrieval uses only',
              'observed no-retrieval trials. Other behavior means use observed traces.',
              'Exact denominators are recorded in summary.json.', '',
              '## Task types', '', '| Task type | Trials | Success rate | Mean recall |', '| --- | ---: | ---: | ---: |']
    for name, group in summary['by_task_type'].items():
        recall = group['average_source_recall']
        lines.append(f'| {name} | {group["total_trials"]} | {group["task_success_rate"]:.4f} | '
                     + (f'{recall:.4f}' if recall is not None else 'undefined') + ' |')
    for heading, key in (('Tool requests', 'tool_usage_distribution'), ('Stop reasons', 'stop_reason_distribution')):
        lines += ['', f'## {heading}', '', '| Name | Count |', '| --- | ---: |']
        lines += [f'| {_cell(name)} | {count} |' for name, count in summary[key].items()]
    if summary['knowledge_changed']:
        lines += ['', 'Knowledge inputs changed during this run. See before/after fingerprints in run_metadata.json.']
    lines += ['', '## Failed trials', '']
    if not summary['failed_trials']:
        lines.append('No failed trials.')
    for failed in summary['failed_trials']:
        lines += [f'### {_cell(failed["case_id"])} / trial {failed["trial"]}', '',
                  '| Field | Value |', '| --- | --- |']
        for key in ('query', 'task_type', 'expected_sources', 'retrieved_sources', 'tool_sequence',
                    'stop_reason', 'failure_reasons'):
            lines.append(f'| {key} | {_cell(failed[key])} |')
        if failed['error'] is not None:
            lines.append(f'| runtime error | {_cell(failed["error"]["type"] + ": " + failed["error"]["message"])} |')
        lines.append('')
    return '\n'.join(lines).rstrip() + '\n'


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=AgentEvalConfig.dataset_path)
    parser.add_argument('--output', type=Path, help='New run directory; defaults to evaluation/results/<run-id>.')
    parser.add_argument('--generation-model', '--model', dest='model', default=AgentEvalConfig.model)
    parser.add_argument('--max-turns', type=int, default=AgentEvalConfig.max_turns)
    parser.add_argument('--num-trials', type=int, default=AgentEvalConfig.num_trials)
    parser.add_argument('--db', type=Path, default=AgentEvalConfig.db)
    parser.add_argument('--notes-dir', type=Path, default=AgentEvalConfig.notes_dir)
    parser.add_argument('--vault-id', default=AgentEvalConfig.vault_id)
    parser.add_argument('--think', action=argparse.BooleanOptionalAction, default=AgentEvalConfig.think)
    parser.add_argument('--host', default=RuntimeConfig.host)
    parser.add_argument('--timeout', type=float, default=RuntimeConfig.timeout)
    parser.add_argument('--qdrant-url')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--tokenizer-cache', type=Path)
    parser.add_argument('--baselines', nargs='+', choices=tuple(BASELINE_MODES),
                        help='Run selected fixed retrieval baselines instead of Agent trials.')
    parser.add_argument('--top-k', type=int, default=BaselineEvalConfig.top_k)
    parser.add_argument('--index-version', help='Pin a baseline run to this ready snapshot; defaults to the initial active snapshot.')
    parser.add_argument('--candidate-k', type=int, default=RetrievalConfig.candidate_k)
    parser.add_argument('--rrf-k', type=float, default=RetrievalConfig.rrf_k)
    parser.add_argument('--rerank-candidates', type=int, default=RetrievalConfig.rerank_candidates)
    parser.add_argument('--reranker-cache')
    parser.add_argument('--reranker-max-length', type=int, default=RetrievalConfig.reranker_max_length)
    parser.add_argument('--exact', action='store_true', help='Use Qdrant exact search for baseline evaluation.')
    args = parser.parse_args(argv)
    selected = {value.split('=', 1)[0] for value in argv if value.startswith('--')}
    baseline_options = {'--top-k', '--index-version', '--candidate-k', '--rrf-k', '--rerank-candidates',
                        '--reranker-cache', '--reranker-max-length', '--exact'}
    if not args.baselines and selected & baseline_options:
        parser.error('Baseline retrieval options require --baselines; they do not configure Agent execution.')
    if args.baselines and selected & {'--model', '--generation-model', '--max-turns', '--think', '--no-think'}:
        parser.error('Agent generation options cannot be used with --baselines.')
    try:
        shared = dict(
            dataset_path=args.dataset, output_dir=args.output, db=args.db,
            notes_dir=args.notes_dir, vault_id=args.vault_id,
            runtime_config=RuntimeConfig(host=args.host, timeout=args.timeout,
                qdrant_url=args.qdrant_url, offline=args.offline, tokenizer_cache=args.tokenizer_cache),
        )
        if args.baselines:
            if args.num_trials != 1:
                raise ValueError('Baselines perform one request per applicable case; --num-trials must be 1.')
            config = BaselineEvalConfig(**shared, baselines=tuple(args.baselines), top_k=args.top_k,
                index_version=args.index_version, exact=args.exact,
                retrieval_config=RetrievalConfig(candidate_k=args.candidate_k, rrf_k=args.rrf_k,
                    rerank_candidates=args.rerank_candidates, reranker_cache=args.reranker_cache,
                    reranker_max_length=args.reranker_max_length))
        else:
            config = AgentEvalConfig(**shared, model=args.model, max_turns=args.max_turns,
                                    num_trials=args.num_trials, think=args.think)
        run = run_evaluation(config)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(_json({'output_dir': run.output_dir, 'summary': run.summary}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
