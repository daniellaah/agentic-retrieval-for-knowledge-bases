"""Model orchestration and replay contracts; no real model or vector services."""

from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pytest

from arkb.agent.state import AgentResult, AgentState, AgentToolTrace
from arkb.evaluation.ablation_analysis import compare_models, summarize_trials, trial_behavior
from arkb.evaluation.agent_metrics import evaluate_case
from arkb.evaluation.model_ablation import (
    AgentModelAblationConfig, DEFAULT_MODELS, inspect_environment, load_trial_results,
    main, run_agent_model_ablation,
)
from arkb.evaluation.models import AgentEvalCase, AgentEvalConfig, AgentEvalTrial
from tests.evaluation.test_agent_metrics import observation, trace


def trial(case, run, number=0, error=None):
    run = replace(run, query=case.query) if run else None
    return AgentEvalTrial(case=case, trial=number, trace=run, metrics=evaluate_case(case, run),
                           runtime_metadata={'elapsed_ms': 12.5}, error=error)


@pytest.fixture
def config(tmp_path, monkeypatch):
    notes = tmp_path / 'notes'
    notes.mkdir()
    (notes / 'a.md').write_text('# A\nEvidence')
    cases = [AgentEvalCase(id='explore_004', query='Find | A\nnow', task_type='exploratory_retrieval',
                           expected_sources=('a.md',), min_read_sources=1),
             AgentEvalCase(id='none_001', query='Hello', task_type='no_retrieval')]
    dataset = tmp_path / 'cases.jsonl'
    dataset.write_text(''.join(json.dumps(asdict(c)) + '\n' for c in cases))
    config = AgentModelAblationConfig(
        evaluation=AgentEvalConfig(dataset_path=dataset, notes_dir=notes, db=tmp_path / 'missing.sqlite', num_trials=3),
        output_dir=tmp_path / 'experiment', smoke_case_ids=tuple(c.id for c in cases))
    monkeypatch.setattr('arkb.evaluation.model_ablation.inspect_environment', lambda *args: {
        'models': {m: {'status': 'available'} for m in DEFAULT_MODELS}, 'missing_models': [], 'errors': []})
    return config


class FakeRuntime:
    def __init__(self):
        self.requests = []

    def ask(self, query, **options):
        self.requests.append((query, options))
        if options['model'] == DEFAULT_MODELS[1]:
            raise ConnectionError('Ollama failed | preserved\nmessage')
        calls = [observation('read', 'a.md')] if query != 'Hello' else []
        return SimpleNamespace(trace=trace(*calls, query=query))


def test_defaults_and_invalid_config():
    config = AgentModelAblationConfig()
    assert config.models == DEFAULT_MODELS and config.evaluation.num_trials == 3
    assert config.evaluation.max_turns == 8 and config.evaluation.think is True
    for options in ({'models': ('a',)}, {'models': ('a', 'a', 'c')}, {'models': ('a/b', 'a:b', 'c')},
                    {'models': ('', 'b', 'c')}, {'phase': 'other'}, {'check_only': 1},
                    {'evaluation': AgentEvalConfig(think=False)}, {'smoke_case_ids': ('a', 'a')}):
        with pytest.raises(ValueError):
            AgentModelAblationConfig(**options)


def test_official_matrix_expands_to_360_independent_calls(config):
    from tests.evaluation.test_agent_dataset import DATASET, NOTES
    config = replace(config, evaluation=replace(config.evaluation, dataset_path=DATASET, notes_dir=NOTES))
    runtime = FakeRuntime()
    output = run_agent_model_ablation(config, runtime=runtime)
    assert len(runtime.requests) == 360
    metadata = json.loads((output / 'run_metadata.json').read_text())
    assert metadata['configured_agent_runs'] == metadata['completed_agent_runs'] == 360
    for model in DEFAULT_MODELS:
        assert metadata['models'][model]['completed_trials'] == 120


def test_models_trials_isolation_failures_serialization_and_reports(config):
    runtime = FakeRuntime()
    output = run_agent_model_ablation(config, runtime=runtime)
    assert len(runtime.requests) == 18
    assert [r[1]['model'] for r in runtime.requests] == [m for m in DEFAULT_MODELS for _ in range(6)]
    normalized = [{k: v for k, v in r[1].items() if k != 'model'} for r in runtime.requests]
    assert all(n == normalized[0] for n in normalized)
    assert normalized[0]['max_turns'] == 8 and normalized[0]['think'] is True
    report = json.loads((output / 'comparison.json').read_text())
    assert report['status'] == 'completed'
    assert [report['summaries'][m]['task_success_rate'] for m in DEFAULT_MODELS] == [1, 0, 1]
    assert report['summaries'][DEFAULT_MODELS[1]]['runtime_error_rate'] == 1
    assert report['case_groups']['scaling_regressions'] == ['explore_004', 'none_001']
    for model, slug in zip(DEFAULT_MODELS, ('qwen3.5-4b', 'qwen3.5-9b', 'qwen3.5-27b')):
        rows = load_trial_results(output / slug / 'results.jsonl')
        assert len(rows) == 6
        assert [r.trial for r in rows] == [0, 1, 2, 0, 1, 2]
        assert all(r.runtime_metadata['model'] == model for r in rows)
        assert all(r.runtime_metadata['experiment']['experiment_id'] == output.name for r in rows)
        serialized = json.loads((output / slug / 'results.jsonl').read_text().splitlines()[0])
        assert serialized['analysis']['prompt_tokens'] is None
        assert serialized['analysis']['latency_ms'] >= 0
        assert (output / slug / 'report.md').exists()
    markdown = (output / 'comparison.md').read_text()
    assert markdown.count('| Exploratory success |') == 1  # Overall only; not an undefined task metric.
    for value in ('exploratory_retrieval', 'Trial stability', '3/3', '0/3', 'explore_004',
                  'Find \\| A<br>now', 'Ollama failed', 'query/annotation'):
        assert value in markdown
    saved = (output / 'qwen3.5-4b/results.jsonl').read_bytes()
    with pytest.raises(FileExistsError):
        run_agent_model_ablation(config, runtime=runtime)
    assert (output / 'qwen3.5-4b/results.jsonl').read_bytes() == saved


def test_evidence_turn_read_rules_and_waste_never_guess_unobserved_execution():
    case = AgentEvalCase(id='x', query='Q', task_type='exploratory_retrieval',
                         expected_sources=('a.md', 'b.md'), min_source_recall=.5, min_read_sources=1)
    row = trial(case, trace(observation('search', 'a.md', 'b.md', turn=1),
                           observation('read', 'a.md', turn=2),
                           observation('search', 'b.md', turn=2), observation('read', 'b.md', turn=3)))
    result = trial_behavior(row)
    assert result['evidence_sufficient_turn'] == 2
    assert result['evidence_sufficient_call_index'] == 1
    assert result['wasted_tool_calls_after_sufficient_evidence'] == 2
    assert result['expected_sources_found_per_tool_call'] == .5
    assert result['search_calls'] == result['read_calls'] == 2
    unobserved = replace(row, trace=replace(row.trace, tool_calls=[*row.trace.tool_calls, AgentToolTrace(4, 'read', {}, None)]))
    assert trial_behavior(unobserved)['wasted_tool_calls_after_sufficient_evidence'] is None
    assert trial_behavior(trial(case, trace(observation('search', 'a.md', 'b.md'))))['evidence_sufficient_turn'] is None
    # Evidence sufficiency ignores prior call-budget violations and abnormal stopping.
    constrained = replace(case, max_tool_calls=0, forbidden_tools=('read',))
    assert trial_behavior(trial(constrained, row.trace))['evidence_sufficient_turn'] == 2
    direct = replace(case, task_type='direct_read', min_source_recall=1)
    assert trial_behavior(trial(direct, row.trace))['evidence_sufficient_turn'] == 3


def test_unavailable_and_no_retrieval_keep_undefined_metrics():
    case = AgentEvalCase(id='none', query='Hello', task_type='no_retrieval')
    for run in (None, trace(query='Hello', turns=1)):
        behavior = trial_behavior(trial(case, run))
        assert behavior['evidence_sufficient_turn'] is None
        assert behavior['wasted_tool_calls_after_sufficient_evidence'] is None
        assert behavior['expected_sources_found_per_tool_call'] is None
        assert behavior['total_tokens'] is None
    assert trial_behavior(trial(case, None))['search_calls'] is None


def test_case_weighted_statistics_and_complete_case_groups():
    cases = [AgentEvalCase(id=k, query=k, task_type='no_retrieval') for k in ('win', 'same', 'regress', 'flaky')]
    successes = ((0, 0, 3, 1), (2, 0, 2, 2), (3, 0, 0, 2))
    runs = {m: [trial(c, trace(reason='final' if t < rates[i] else 'max_turns'), number=t)
                for i, c in enumerate(cases) for t in range(3)]
            for m, rates in zip(DEFAULT_MODELS, successes)}
    report = compare_models(DEFAULT_MODELS, runs, cases=cases, num_trials=3, experiment_id='test', status='completed')
    assert report['case_groups'] == {'scaling_wins': ['win', 'flaky'],
                                    'scaling_insensitive': ['same'], 'scaling_regressions': ['regress']}
    stat = report['summaries'][DEFAULT_MODELS[0]]['case_statistics']
    assert stat['mean_success_probability_across_cases'] == pytest.approx(1 / 3)
    assert stat['always_pass_cases'] == ['regress'] and stat['flaky_cases'] == ['flaky']
    assert stat['per_case']['flaky']['success_rate'] == pytest.approx(1 / 3)
    # Unequal observed trials must not silently become a pooled success rate.
    unequal = [*runs[DEFAULT_MODELS[0]][:3], trial(cases[1], trace())]
    summary = summarize_trials(unequal, num_trials=3)
    assert summary['task_success_rate'] == .25
    assert summary['case_statistics']['empirical_pass_at_1'] == .5
    assert summary['case_statistics']['incomplete_cases'] == ['same']
    partial = compare_models(DEFAULT_MODELS, {DEFAULT_MODELS[0]: runs[DEFAULT_MODELS[0]]},
                             cases=cases, num_trials=3, experiment_id='test', status='incomplete')
    assert not any(partial['case_groups'].values())


def test_missing_models_block_formal_without_substitution_but_allow_labeled_smoke(config, monkeypatch):
    environment = {'models': {m: {'status': 'available' if i == 0 else 'missing'}
                              for i, m in enumerate(DEFAULT_MODELS)},
                   'missing_models': list(DEFAULT_MODELS[1:]), 'errors': []}
    monkeypatch.setattr('arkb.evaluation.model_ablation.inspect_environment', lambda *args: environment)
    runtime = FakeRuntime()
    out = run_agent_model_ablation(config, runtime=runtime)
    assert not runtime.requests
    comparison = json.loads((out / 'comparison.json').read_text())
    assert comparison['status'] == 'blocked' and all(s is None for s in comparison['summaries'].values())
    smoke = replace(config, phase='smoke', output_dir=out.parent / 'smoke')
    out = run_agent_model_ablation(smoke, runtime=runtime)
    assert len(runtime.requests) == 2
    metadata = json.loads((out / 'run_metadata.json').read_text())
    assert metadata['status'] == 'smoke_partial' and metadata['num_trials'] == 1
    assert metadata['configured_agent_runs'] == 6 and metadata['completed_agent_runs'] == 2
    assert all(r[1]['model'] == DEFAULT_MODELS[0] for r in runtime.requests)
    assert config.evaluation.dataset_path.read_bytes() == (out / 'cases.jsonl').read_bytes()


def test_runner_level_failure_does_not_block_other_models(config, monkeypatch):
    from arkb.evaluation.runs import run_agent_evaluation
    seen = []
    def runner(config, **options):
        seen.append(config.model)
        if config.model == DEFAULT_MODELS[0]:
            raise OSError('model setup failed')
        return run_agent_evaluation(config, **options)
    monkeypatch.setattr('arkb.evaluation.model_ablation.run_agent_evaluation', runner)
    out = run_agent_model_ablation(config, runtime=FakeRuntime())
    assert seen == list(DEFAULT_MODELS)
    metadata = json.loads((out / 'run_metadata.json').read_text())
    assert metadata['status'] == 'incomplete' and metadata['completed_agent_runs'] == 12
    assert metadata['models'][DEFAULT_MODELS[0]]['error']['message'] == 'model setup failed'


def test_controls_drift_invalidates_comparison_and_stops_later_models(config):
    class ChangingRuntime(FakeRuntime):
        def ask(self, *args, **options):
            (config.evaluation.notes_dir / 'a.md').write_text('Changed corpus')
            return super().ask(*args, **options)
    runtime = ChangingRuntime()
    out = run_agent_model_ablation(config, runtime=runtime)
    report = json.loads((out / 'comparison.json').read_text())
    assert report['status'] == 'invalid_controls'
    assert len(runtime.requests) == 6 and not any(report['case_groups'].values())


def test_interruption_keeps_partial_rows_and_marks_case_incomplete(config):
    class Interrupted(FakeRuntime):
        def ask(self, *args, **options):
            if self.requests:
                raise KeyboardInterrupt()
            return super().ask(*args, **options)
    with pytest.raises(KeyboardInterrupt):
        run_agent_model_ablation(config, runtime=Interrupted())
    metadata = json.loads((config.output_dir / 'run_metadata.json').read_text())
    assert metadata['status'] == 'interrupted' and metadata['completed_agent_runs'] == 1
    comparison = json.loads((config.output_dir / 'comparison.json').read_text())
    stats = comparison['summaries'][DEFAULT_MODELS[0]]['case_statistics']
    assert stats['incomplete_cases'] == ['explore_004']


def test_preflight_checks_exact_tags_without_downloading(config, monkeypatch):
    import httpx
    calls = []
    def handle(request):
        calls.append(request.url.path)
        if request.url.path == '/api/tags':
            return httpx.Response(200, json={'models': [{'model': 'qwen3.5:4b', 'digest': 'four'},
                                                       {'model': 'qwen3-embedding:0.6b', 'digest': 'embed'}]})
        if request.url.path == '/api/version':
            return httpx.Response(200, json={'version': 'test'})
        assert request.url.path == '/api/show'
        return httpx.Response(200, json={'capabilities': ['tools', 'thinking'], 'template': '{{ .Prompt }}'})
    client = httpx.Client(transport=httpx.MockTransport(handle), base_url='http://ollama.test')
    monkeypatch.setattr('arkb.evaluation.model_ablation.httpx.Client', lambda **kw: client)
    monkeypatch.setattr('arkb.knowledge.embeddings.load_tokenizer', lambda **kw: object())
    monkeypatch.setattr('arkb.knowledge.embeddings.tokenizer_fingerprint', lambda *args: 'fixed')
    environment = inspect_environment(config, {'snapshot': None})
    assert environment['missing_models'] == list(DEFAULT_MODELS[1:])
    assert environment['models'][DEFAULT_MODELS[0]]['status'] == 'available'
    assert [e['stage'] for e in environment['errors']] == ['index']
    assert '/api/pull' not in calls and '/api/chat' not in calls


def test_cli_reuses_baseline_configuration_and_preserves_three_trials(config, monkeypatch, capsys):
    baseline = config.output_dir.parent / 'baseline.json'
    baseline.write_text(json.dumps({'config': {**asdict(config.evaluation), 'num_trials': 1}}, default=str))
    monkeypatch.setattr('arkb.evaluation.model_ablation.inspect_environment', lambda *args: {
        'models': {m: {'status': 'available'} for m in DEFAULT_MODELS}, 'errors': [], 'missing_models': []})
    assert main(['--baseline-metadata', str(baseline), '--output', str(config.output_dir), '--check-only']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'checked'
    metadata = json.loads((config.output_dir / 'run_metadata.json').read_text())
    assert metadata['num_trials'] == 3 and metadata['configured_agent_runs'] == 18
    assert metadata['controls']['runtime_config'] == asdict(config.evaluation.runtime_config)
