"""Run agent evaluation through Runtime.ask and preserve inspectable artifacts."""

import argparse
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter
from typing import Protocol
from uuid import uuid4

from arkb.agent.state import AgentResult
from arkb.config import RuntimeConfig
from arkb.evaluation.agent import evaluate_case, summarize_agent_results
from arkb.evaluation.datasets import corpus_manifest, parse_agent_eval_dataset, source_hashes
from arkb.evaluation.models import AgentEvalConfig, AgentEvalRun, AgentEvalTrial
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


def _knowledge_metadata(config: AgentEvalConfig) -> dict:
    """Record live file hashes and the published snapshot without loading models."""
    from arkb.knowledge.sqlite import SQLiteStorage

    root = config.notes_dir.resolve()
    notes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(root.glob('*.md'))
             if p.is_file() and p.resolve().is_relative_to(root)}
    snapshot = None
    if config.db.exists():
        with SQLiteStorage(config.db, read_only=True) as storage:
            manifest = storage.active_manifest(config.vault_id)
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
) -> AgentEvalRun:
    """Run every case/trial with fresh runtime conversation state and no retries.

    Own a Runtime only when one is not supplied. Inputs validate before any ask.
    Ordinary per-trial runtime exceptions retain partial traces and do not stop
    later trials. Interruptions and artifact I/O errors propagate; each completed
    row is flushed, and metadata marks incomplete runs instead of reporting success.
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
                        stream.write(_json({'schema_version': SCHEMA_VERSION, **asdict(row)}) + '\n')
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
    args = parser.parse_args(argv)
    try:
        config = AgentEvalConfig(
            dataset_path=args.dataset, output_dir=args.output, model=args.model,
            max_turns=args.max_turns, num_trials=args.num_trials, db=args.db,
            notes_dir=args.notes_dir, vault_id=args.vault_id, think=args.think,
            runtime_config=RuntimeConfig(host=args.host, timeout=args.timeout,
                                         qdrant_url=args.qdrant_url, offline=args.offline,
                                         tokenizer_cache=args.tokenizer_cache),
        )
        run = run_agent_evaluation(config)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(_json({'output_dir': run.output_dir, 'summary': run.summary}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
