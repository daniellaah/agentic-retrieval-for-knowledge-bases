"""Thin, controlled multi-model orchestration over Agent Evaluation v1."""

import argparse
from contextlib import closing
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import re
import subprocess
from uuid import uuid4

import httpx

from arkb.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_RETRIEVAL_MODE, RetrievalConfig, RuntimeConfig
from arkb.evaluation.ablation_analysis import compare_models, render_comparison, trial_behavior
from arkb.evaluation.agent_runner import (
    _code_metadata, _failed_trial, _json, _knowledge_metadata, _now, _write_json, render_agent_report,
    run_agent_evaluation,
)
from arkb.evaluation.datasets import parse_agent_eval_dataset
from arkb.evaluation.models import AgentEvalCase, AgentEvalConfig, AgentEvalTrial


DEFAULT_MODELS = ('qwen3.5:4b', 'qwen3.5:9b', 'qwen3.5:27b')
SMOKE_CASES = ('exact_001', 'semantic_005', 'read_001', 'explore_004', 'qa_006', 'none_001')


def model_directory(model):
    return re.sub(r'[^a-zA-Z0-9._-]+', '-', model).strip('.-')


@dataclass(frozen=True, kw_only=True)
class AgentModelAblationConfig:
    models: tuple[str, ...] = DEFAULT_MODELS
    evaluation: AgentEvalConfig = field(default_factory=lambda: AgentEvalConfig(num_trials=3))
    output_dir: Path | None = None
    phase: str = 'formal'
    smoke_case_ids: tuple[str, ...] = SMOKE_CASES
    check_only: bool = False

    def __post_init__(self):
        if not isinstance(self.models, (list, tuple)) or len(self.models) != 3:
            raise ValueError('Phase 1 requires exactly three model tags, in increasing capacity order.')
        object.__setattr__(self, 'models', tuple(self.models))
        if any(not isinstance(m, str) or not m.strip() or m != m.strip() for m in self.models):
            raise ValueError('Model tags must be nonblank without surrounding whitespace.')
        directories = [model_directory(m) for m in self.models]
        if not all(directories) or len(set(directories)) != len(directories):
            raise ValueError('Model tags must have distinct, safe output directory names.')
        if not isinstance(self.evaluation, AgentEvalConfig):
            raise ValueError('evaluation must be AgentEvalConfig.')
        if self.evaluation.output_dir is not None:
            raise ValueError('Set the experiment output_dir, not evaluation.output_dir.')
        if self.phase not in ('formal', 'smoke') or type(self.check_only) is not bool:
            raise ValueError('phase must be formal/smoke and check_only must be boolean.')
        if self.evaluation.think != AgentEvalConfig.think:
            raise ValueError('Phase 1 preserves the current Agent Runtime default think setting.')
        if self.evaluation.runtime_config.embedding_model not in (None, DEFAULT_EMBEDDING_MODEL):
            raise ValueError('Phase 1 cannot change the embedding model.')
        if (not isinstance(self.smoke_case_ids, (list, tuple)) or not self.smoke_case_ids
                or any(not isinstance(c, str) or not c.strip() for c in self.smoke_case_ids)
                or len(set(self.smoke_case_ids)) != len(self.smoke_case_ids)):
            raise ValueError('smoke_case_ids must be distinct nonblank case IDs.')
        object.__setattr__(self, 'smoke_case_ids', tuple(self.smoke_case_ids))
        if self.output_dir is not None:
            if not isinstance(self.output_dir, (str, Path)) or not str(self.output_dir).strip():
                raise ValueError('output_dir must be a nonblank path.')
            object.__setattr__(self, 'output_dir', Path(self.output_dir))


def _error(stage, error):
    return {'stage': stage, 'type': type(error).__name__, 'message': str(error)}


def inspect_environment(config, knowledge) -> dict:
    """Read services and exact local tags. Never pull models or build an index."""
    settings = config.evaluation.runtime_config
    result = {'models': {}, 'missing_models': [], 'errors': [], 'warnings': [],
              'ollama_version': None, 'qdrant': None, 'embedding': None}
    with httpx.Client(base_url=settings.host, timeout=min(settings.timeout, 15), trust_env=False) as http:
        try:
            tags = http.get('/api/tags').raise_for_status().json()['models']
            inventory = {m.get('model', m.get('name')): m for m in tags}
            result['available_tags'] = sorted(inventory)
            for model in config.models:
                if model not in inventory:
                    result['missing_models'].append(model)
                    result['models'][model] = {'status': 'missing'}
                    continue
                entry = {'status': 'available', 'artifact': inventory[model]}
                try:
                    info = http.post('/api/show', json={'model': model}).raise_for_status().json()
                    entry.update(parameters=info.get('parameters'), capabilities=info.get('capabilities'),
                                 template_sha256=hashlib.sha256(info.get('template', '').encode()).hexdigest(),
                                 system=info.get('system'), model_info=info.get('model_info'))
                    if 'tools' not in (info.get('capabilities') or ()):
                        raise ValueError('Installed model does not advertise tool calling.')
                    if config.evaluation.think and 'thinking' not in (info.get('capabilities') or ()):
                        raise ValueError('Installed model does not advertise thinking support.')
                except Exception as error:
                    entry.update(status='unavailable', error=_error('model_inspection', error))
                result['models'][model] = entry
            result['embedding'] = inventory.get(DEFAULT_EMBEDDING_MODEL)
            if result['embedding'] is None:
                raise ValueError(f'Missing embedding model: {DEFAULT_EMBEDDING_MODEL}')
        except Exception as error:
            result['errors'].append(_error('ollama_inventory', error))
        try:
            result['ollama_version'] = http.get('/api/version').raise_for_status().json()
        except Exception as error:
            result['warnings'].append(_error('ollama_version', error))
    snapshot = knowledge['snapshot']
    if snapshot is None:
        result['errors'].append(_error('index', ValueError('A published shared index snapshot is required.')))
    else:
        from arkb.knowledge.models import EmbeddingSpec
        from arkb.knowledge.qdrant import QdrantIndex, check_qdrant_collection, connect_qdrant
        from arkb.knowledge.sqlite import SQLiteStorage
        manifest, backend = snapshot['manifest'], snapshot['build_metadata']['backend']
        spec = EmbeddingSpec(**manifest['embedding_spec'])
        if spec.model != DEFAULT_EMBEDDING_MODEL:
            result['errors'].append(_error('embedding', ValueError('Snapshot embedding differs from Phase 1.')))
        if result['embedding'] and result['embedding'].get('digest') != spec.model_revision:
            result['errors'].append(_error('embedding', ValueError('Installed embedding digest differs from snapshot.')))
        try:
            url = settings.qdrant_url or backend['url']
            with closing(connect_qdrant(url, min(settings.qdrant_timeout or settings.timeout, 15))) as client:
                info = check_qdrant_collection(client, backend['collection'], spec=spec,
                                               vault_id=manifest['vault_id'])
                count = client.count(backend['collection'], exact=True).count
                if count != manifest['chunk_count']:
                    raise ValueError('Qdrant point count differs from the saved snapshot.')
                with SQLiteStorage(config.evaluation.db, read_only=True) as storage:
                    _, records, vectors = storage.load_snapshot(manifest['index_version'])
                QdrantIndex(client, backend['collection'], spec,
                             vault_id=manifest['vault_id']).verify_snapshot(records, vectors)
                result['qdrant'] = {'url': url, 'collection': backend['collection'],
                                    'points': count, 'snapshot_vectors_verified': True,
                                    'configuration': info.model_dump(mode='json')}
        except Exception as error:
            result['errors'].append(_error('qdrant', error))
    try:
        from arkb.knowledge.embeddings import load_tokenizer, tokenizer_fingerprint
        tokenizer = load_tokenizer(cache_dir=settings.tokenizer_cache, local_files_only=True)
        fingerprint = tokenizer_fingerprint(tokenizer)
        result['tokenizer_fingerprint'] = fingerprint
        if snapshot and fingerprint != snapshot['build_metadata']['backend']['input']['tokenizer']:
            raise ValueError('Cached embedding tokenizer differs from the snapshot.')
    except Exception as error:
        result['errors'].append(_error('tokenizer', error))
    return result


def _controls(config, raw, knowledge):
    from arkb.agent.loop import SYSTEM_INSTRUCTION
    from arkb.agent.tools import tool_definitions
    return {
        'dataset_path': config.evaluation.dataset_path.resolve(),
        'dataset_sha256': hashlib.sha256(raw).hexdigest(),
        'knowledge': knowledge, 'runtime_config': asdict(config.evaluation.runtime_config),
        'max_turns': config.evaluation.max_turns, 'think': config.evaluation.think,
        'generation_parameters': {'temperature': 0, 'stream': False, 'num_ctx': None,
                                  'num_predict': None, 'seed': None,
                                  'unspecified_options': 'Inherited provider defaults; no per-model overrides.'},
        'retrieval': {'default_mode': DEFAULT_RETRIEVAL_MODE, 'settings': asdict(RetrievalConfig()),
                      'reranker_enabled': False, 'exact': False,
                      'available_modes': ['bm25', 'semantic', 'hybrid']},
        'agent_system_prompt': SYSTEM_INSTRUCTION,
        'tool_definitions': tool_definitions(('bm25', 'semantic', 'hybrid'), default_mode=DEFAULT_RETRIEVAL_MODE),
        'source_hashes': _code_metadata()['source_hashes'],
    }


def _extra_code_metadata():
    result = {}
    for name, args in (('git_branch', ['branch', '--show-current']),
                       ('git_status', ['status', '--porcelain'])):
        try:
            result[name] = subprocess.run(['git', *args], capture_output=True, text=True,
                                           check=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            result[name] = None
    packages = {}
    for name in ('ollama', 'qdrant-client', 'httpx', 'numpy', 'tokenizers'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    result['packages'] = packages
    return result


def load_trial_results(path):
    """Replay flushed v1 rows, including partial runs, using original metrics."""
    from arkb.agent.state import AgentToolTrace, AgentTrace
    from arkb.evaluation.agent import evaluate_case

    rows = []
    if not path.exists():
        return ()
    for line in path.read_text(encoding='utf-8').splitlines():
        saved = json.loads(line)
        case = AgentEvalCase(**saved['case'])
        trace = saved['trace']
        if trace is not None:
            trace = AgentTrace(**{**trace, 'tool_calls': [AgentToolTrace(**c) for c in trace['tool_calls']]})
        metrics = evaluate_case(case, trace)
        if json.loads(_json(asdict(metrics))) != saved['metrics']:
            raise ValueError('Saved outcome differs from Agent Evaluation v1 replay.')
        rows.append(AgentEvalTrial(case=case, trial=saved['trial'], trace=trace, metrics=metrics,
                                   runtime_metadata=saved['runtime_metadata'], error=saved['error']))
    return tuple(rows)


def run_agent_model_ablation(config: AgentModelAblationConfig, *, runtime=None) -> Path:
    """Each model reuses the v1 runner; trial failures remain rows with no retries.

    Formal execution requires the entire model set and retrieval environment.
    Smoke may inspect available models, retaining missing-model slots explicitly.
    Every output is new. Fingerprints are checked across model runs; drift makes
    comparisons invalid and stops further models. No historical rows are pooled.
    """
    base = config.evaluation
    raw = base.dataset_path.read_bytes()
    cases = parse_agent_eval_dataset(raw, notes_dir=base.notes_dir)
    selected = cases
    if config.phase == 'smoke':
        missing = set(config.smoke_case_ids) - {c.id for c in cases}
        if missing:
            raise ValueError(f'Unknown smoke case IDs: {sorted(missing)}')
        selected = [c for c in cases if c.id in config.smoke_case_ids]
    output = config.output_dir or Path('evaluation/results/agent_model_ablation') / (
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ-') + config.phase + '-' + uuid4().hex[:8])
    if output.exists():
        raise FileExistsError(f'Experiment output must be a new directory: {output}')
    knowledge = _knowledge_metadata(base)
    controls = _controls(config, raw, knowledge)
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / 'config.json', asdict(config))
    (output / 'cases.jsonl').write_bytes(raw)
    if config.phase == 'smoke':
        smoke_path = output / 'smoke_cases.jsonl'
        smoke_path.write_text(''.join(_json(asdict(c)) + '\n' for c in selected), encoding='utf-8')
        base = replace(base, dataset_path=smoke_path, num_trials=1)
    metadata = {
        'schema_version': 'agent-model-ablation-v1', 'experiment_id': output.name,
        'status': 'preflight', 'phase': config.phase, 'started_at': _now(),
        'controls': controls, 'configured_models': config.models,
        'case_count': len(selected), 'num_trials': base.num_trials,
        'configured_agent_runs': len(selected) * base.num_trials * len(config.models),
        'completed_agent_runs': 0, 'models': {}, **_code_metadata(), **_extra_code_metadata(),
        'limitations': ['AgentTrace does not retain token usage.',
                       'Runtime sets temperature=0 but does not set num_ctx, num_predict or seed.',
                       'Latency includes cold loading, retrieval and provider cache effects; model order is fixed.'],
    }
    runs = {}
    _write_json(output / 'run_metadata.json', metadata)
    try:
        environment = inspect_environment(config, knowledge)
        metadata['environment'] = environment
        unavailable = [m for m in config.models if environment['models'].get(m, {}).get('status') != 'available']
        blocked = bool(environment['errors']) or (config.phase == 'formal' and bool(unavailable))
        metadata['status'] = 'blocked' if blocked else 'checked' if config.check_only else 'running'
        for model in config.models:
            entry = environment['models'].get(model, {'status': 'unavailable'})
            metadata['models'][model] = {'status': entry['status'], 'output_dir': model_directory(model)}
        _write_json(output / 'run_metadata.json', metadata)
        if not blocked and not config.check_only:
            for model in config.models:
                if model in unavailable:
                    continue
                current = _controls(config, config.evaluation.dataset_path.read_bytes(), _knowledge_metadata(config.evaluation))
                if current != controls:
                    metadata['status'] = 'invalid_controls'
                    metadata['controls_after'] = current
                    break
                model_output = output / model_directory(model)
                entry = metadata['models'][model]
                entry['status'] = 'running'
                _write_json(output / 'run_metadata.json', metadata)
                try:
                    run = run_agent_evaluation(
                        replace(base, model=model, output_dir=model_output), runtime=runtime,
                        trial_metadata={'experiment_id': output.name, 'metadata_file': '../run_metadata.json',
                                        'phase': config.phase, 'agent_model': model,
                                        'num_trials': base.num_trials},
                        trial_annotations=trial_behavior,
                    )
                    runs[model] = run.results
                    entry.update(status='completed', completed_trials=len(run.results))
                except (OSError, ValueError, RuntimeError) as error:
                    # Model/trial errors normally stay inside v1. Setup/artifact
                    # failures remain explicit and do not suppress other models.
                    entry.update(status='failed', error=_error('evaluation_runner', error))
                current = _controls(config, config.evaluation.dataset_path.read_bytes(), _knowledge_metadata(config.evaluation))
                if current != controls:
                    metadata.update(status='invalid_controls', controls_after=current)
                    break
            if metadata['status'] == 'running':
                complete = all(e['status'] == 'completed' for e in metadata['models'].values())
                metadata['status'] = ('smoke_completed' if complete else 'smoke_partial') if config.phase == 'smoke' else (
                    'completed' if complete else 'incomplete')
    except BaseException as error:
        metadata.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                        error=_error('experiment', error))
        raise
    finally:
        for model, entry in metadata['models'].items():
            if model not in runs:
                recovered = load_trial_results(output / model_directory(model) / 'results.jsonl')
                if recovered:
                    runs[model] = recovered
                    entry['completed_trials'] = len(recovered)
            if entry['status'] == 'running':
                entry['status'] = metadata['status']
        metadata.update(finished_at=_now(), completed_agent_runs=sum(len(rows) for rows in runs.values()))
        _write_json(output / 'run_metadata.json', metadata)
        report = compare_models(config.models, runs, cases=selected, num_trials=base.num_trials,
                                experiment_id=output.name, status=metadata['status'])
        report['preflight'] = metadata.get('environment')
        report['model_statuses'] = metadata['models']
        _write_json(output / 'comparison.json', report)
        text = render_comparison(report)
        if report['preflight']:
            text += '\n## Preflight\n\n' + '\n'.join(
                f'- {m}: {metadata["models"].get(m, {}).get("status", "unavailable")}' for m in config.models) + '\n'
            for error in report['preflight']['errors']:
                text += f'- {error["stage"]}: {error["type"]}: {error["message"]}\n'
        (output / 'comparison.md').write_text(text, encoding='utf-8')
        for model, rows in runs.items():
            model_output = output / model_directory(model)
            summary_path = model_output / 'summary.json'
            summary = (json.loads(summary_path.read_text()) if summary_path.exists() else {
                'schema_version': 'agent-evaluation-v1', 'run_id': model_output.name,
                'failed_trials': [_failed_trial(row) for row in rows if not row.metrics.success],
                'knowledge_changed': metadata['status'] == 'invalid_controls',
            })
            summary.update(report['summaries'][model])
            summary['experiment_status'] = metadata['status']
            summary['model_run_status'] = metadata['models'][model]['status']
            _write_json(summary_path, summary)
            (model_output / 'report.md').write_text(render_agent_report(summary), encoding='utf-8')
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs=3, default=DEFAULT_MODELS, help='Exact installed tags, in 4B/9B/27B order.')
    parser.add_argument('--baseline-metadata', type=Path, help='Reuse the existing evaluation config, except model/output/trials.')
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--db', type=Path)
    parser.add_argument('--notes-dir', type=Path)
    parser.add_argument('--vault-id')
    parser.add_argument('--host')
    parser.add_argument('--qdrant-url')
    parser.add_argument('--num-trials', type=int, default=3)
    parser.add_argument('--max-turns', type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--phase', choices=('formal', 'smoke'), default='formal')
    parser.add_argument('--smoke-cases', nargs='+', default=SMOKE_CASES)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args(argv)
    try:
        values = json.loads(args.baseline_metadata.read_text())['config'] if args.baseline_metadata else {}
        runtime = values.pop('runtime_config', {})
        values.update(output_dir=None, num_trials=args.num_trials)
        for name in ('dataset', 'db', 'notes_dir', 'vault_id', 'max_turns'):
            value = getattr(args, name)
            if value is not None:
                values['dataset_path' if name == 'dataset' else name] = value
        for name in ('host', 'qdrant_url'):
            if getattr(args, name) is not None:
                runtime[name] = getattr(args, name)
        values['runtime_config'] = RuntimeConfig(**runtime)
        config = AgentModelAblationConfig(models=tuple(args.models), evaluation=AgentEvalConfig(**values),
            output_dir=args.output, phase=args.phase, smoke_case_ids=tuple(args.smoke_cases), check_only=args.check_only)
        output = run_agent_model_ablation(config)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    metadata = json.loads((output / 'run_metadata.json').read_text())
    print(_json({'output_dir': output, 'status': metadata['status'],
                 'completed_agent_runs': metadata['completed_agent_runs'], 'models': metadata['models']}, indent=2))
    return 0 if metadata['status'] in ('completed', 'checked', 'smoke_completed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
