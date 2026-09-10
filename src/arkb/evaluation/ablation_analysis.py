"""Replayable model comparisons over unchanged Agent Evaluation v1 outcomes."""

from collections import Counter
from dataclasses import replace
from statistics import mean

from arkb.evaluation.agent import evaluate_case, summarize_agent_results
from arkb.evaluation.agent_runner import _cell


FOCUSED_CASES = ('semantic_005', 'explore_001', 'explore_002', 'explore_004',
                 'explore_005', 'explore_006', 'explore_007', 'qa_006', 'exact_001')


def trial_behavior(row) -> dict:
    """Measure annotated source sufficiency, never answer quality or intent.

    Sufficiency is the first completed observation satisfying v1 evidence rules,
    independent of final stopping and tool-budget compliance. Same-turn calls
    were requested together: counting them does not establish avoidable work.
    An unobserved call could have failed or never executed, so an executed-call
    count crossing such a call is unknown. Tokens are not retained by AgentTrace.
    """
    trace, case, metrics = row.trace, row.case, row.metrics
    sufficient_turn = sufficient_index = wasted = None
    if trace is not None and case.expected_sources:
        evidence_case = replace(case, allowed_tools=None, forbidden_tools=(), max_tool_calls=None)
        for index, call in enumerate(trace.tool_calls):
            if call.result is None:
                continue
            prefix = replace(trace, tool_calls=trace.tool_calls[:index + 1], stop_reason='final')
            if evaluate_case(evidence_case, prefix).success:
                sufficient_turn, sufficient_index = call.turn, index
                later = trace.tool_calls[index + 1:]
                if all(c.result is not None for c in later):
                    wasted = len(later)
                break
    expected_found = (len(set(case.expected_sources) & set(metrics.retrieved_sources))
                      if metrics.retrieved_sources is not None and case.expected_sources else None)
    return {
        'match_calls': metrics.tool_counts.get('match', 0) if metrics.tool_counts is not None else None,
        'search_calls': metrics.tool_counts.get('search', 0) if metrics.tool_counts is not None else None,
        'read_calls': metrics.tool_counts.get('read', 0) if metrics.tool_counts is not None else None,
        'evidence_sufficient_turn': sufficient_turn,
        'evidence_sufficient_call_index': sufficient_index,
        'wasted_tool_calls_after_sufficient_evidence': wasted,
        'expected_sources_found': expected_found,
        'expected_sources_found_per_tool_call': (
            expected_found / metrics.tool_call_count
            if expected_found is not None and metrics.tool_call_count else None),
        'latency_ms': row.runtime_metadata.get('elapsed_ms'),
        'model_request_count': metrics.turn_count,
        'prompt_tokens': None, 'completion_tokens': None, 'total_tokens': None,
        'runtime_error': row.error is not None or metrics.stop_reason == 'error',
        'retry_count': 0,
    }


def _behavior_summary(rows) -> dict:
    behaviors = [trial_behavior(row) for row in rows]
    result = {}
    for field in ('match_calls', 'search_calls', 'read_calls', 'evidence_sufficient_turn',
                  'wasted_tool_calls_after_sufficient_evidence',
                  'expected_sources_found_per_tool_call', 'latency_ms', 'model_request_count'):
        values = [b[field] for b in behaviors if b[field] is not None]
        result['average_' + field] = mean(values) if values else None
        result[field + '_defined_trials'] = len(values)
    result['normal_final_rate'] = mean(row.metrics.stop_reason == 'final' for row in rows)
    result['runtime_error_rate'] = mean(b['runtime_error'] for b in behaviors)
    result['runtime_error_count'] = sum(b['runtime_error'] for b in behaviors)
    result['errors_by_type'] = dict(sorted(Counter(
        row.error['type'] for row in rows if row.error is not None).items()))
    result['errors'] = [{'case_id': row.case.id, 'trial': row.trial, 'error': row.error,
                         'partial_trace_available': row.trace is not None}
                        for row in rows if row.error is not None]
    return result


def summarize_trials(rows, *, num_trials: int) -> dict:
    """Keep v1 denominators, add case-weighted observed probabilities/stability."""
    base = summarize_agent_results([row.metrics for row in rows])
    base.update(_behavior_summary(rows))
    for task, group in base['by_task_type'].items():
        group.update(_behavior_summary([r for r in rows if r.case.task_type == task]))
    base['exploratory_success_rate'] = base['by_task_type'].get('exploratory_retrieval', {}).get('task_success_rate')
    cases = {}
    for case_id in sorted({r.case.id for r in rows}):
        trials = [r for r in rows if r.case.id == case_id]
        successes = sum(r.metrics.success for r in trials)
        complete = len(trials) == num_trials and {r.trial for r in trials} == set(range(num_trials))
        cases[case_id] = {
            'successes': successes, 'trials': len(trials), 'configured_trials': num_trials,
            'success_rate': successes / len(trials), 'complete': complete,
            'stability': ('incomplete' if not complete else 'always_pass' if successes == len(trials)
                          else 'always_fail' if successes == 0 else 'flaky'),
        }
    probability = mean(c['success_rate'] for c in cases.values())
    base['case_statistics'] = {
        'per_case': cases, 'empirical_pass_at_1': probability,
        'mean_success_probability_across_cases': probability,
        **{name + '_cases': [key for key, c in cases.items() if c['stability'] == name]
           for name in ('always_pass', 'flaky', 'always_fail', 'incomplete')},
    }
    return base


def compare_models(models, runs, *, cases, num_trials, experiment_id, status) -> dict:
    summaries = {m: summarize_trials(runs[m], num_trials=num_trials) if runs.get(m) else None
                 for m in models}
    details, groups = {}, {'scaling_wins': [], 'scaling_insensitive': [], 'scaling_regressions': []}
    for case in cases:
        per_model = {}
        rates = []
        for model in models:
            rows = [r for r in runs.get(model, ()) if r.case.id == case.id]
            stat = summaries[model]['case_statistics']['per_case'].get(case.id) if summaries[model] else None
            rates.append(stat['success_rate'] if stat and stat['complete'] else None)
            per_model[model] = {
                'statistics': stat,
                'trials': [{
                    'trial': r.trial, 'success': r.metrics.success, 'recall': r.metrics.source_recall,
                    'tool_sequence': r.metrics.tool_calls, 'turns': r.metrics.turn_count,
                    'stop_reason': r.metrics.stop_reason, 'error': r.error,
                    'missing_expected_sources': (sorted(set(case.expected_sources) - set(r.metrics.retrieved_sources))
                                                 if r.metrics.retrieved_sources is not None else None),
                    'read_sources': r.metrics.read_sources,
                    'tool_arguments': [{'turn': c.turn, 'name': c.name, 'arguments': c.arguments}
                                       for c in r.trace.tool_calls] if r.trace else None,
                    'behavior': trial_behavior(r),
                } for r in rows],
            }
        details[case.id] = {'query': case.query, 'expected_sources': case.expected_sources,
                            'task_type': case.task_type, 'models': per_model}
        if status == 'completed' and all(rate is not None for rate in rates):
            if any(rate > rates[0] for rate in rates[1:]):
                groups['scaling_wins'].append(case.id)
            if len(set(rates)) == 1:
                groups['scaling_insensitive'].append(case.id)
            if any(b < a for a, b in zip(rates, rates[1:])):
                groups['scaling_regressions'].append(case.id)
    return {
        'schema_version': 'agent-model-ablation-v1', 'experiment_id': experiment_id,
        'status': status, 'models': list(models), 'summaries': summaries,
        'case_comparison': details, 'case_groups': groups,
        'focused_cases': {key: details[key] for key in FOCUSED_CASES if key in details},
        'definitions': {
            'empirical_pass_at_1': 'Mean per-case successes / observed trials; no pass@k extrapolation.',
            'scaling_wins': 'Any larger model has a higher empirical case success rate than the first model.',
            'scaling_insensitive': 'All three empirical case success rates are exactly equal.',
            'scaling_regressions': 'An adjacent larger model has a lower empirical case success rate; may overlap wins.',
            'classification': 'Complete, control-valid matrices only; empirical differences are not significance tests.',
            'evidence_sufficient_turn': 'Turn returning the first observation meeting existing source/read labels; '
                                        'does not certify answer correctness. No-retrieval is undefined.',
            'wasted_tool_calls': 'Completed calls after that observation, including same-turn batches. '
                                 'Null if sufficiency is absent or later execution is uncertain; not a claim of avoidability.',
            'latency': 'Wall time of Runtime.ask, including model loading/retrieval and failed calls; no warmup correction.',
            'tokens': 'Unavailable in existing AgentTrace; null, never estimated.',
            'error_attribution': 'Preserve exception type/message; existing trace cannot reliably separate model, tool and infrastructure stages.',
        },
    }


COMPARISON_METRICS = (
    ('Task success', 'task_success_rate'), ('Source recall', 'average_source_recall'),
    ('Exploratory success', 'exploratory_success_rate'),
    ('Avg tool calls', 'average_tool_calls'), ('Avg turns', 'average_turns'),
    ('Max-turn failure', 'max_turn_failure_rate'), ('Unnecessary retrieval', 'unnecessary_retrieval_rate'),
    ('Avg match calls', 'average_match_calls'), ('Avg search calls', 'average_search_calls'),
    ('Avg read calls', 'average_read_calls'), ('Normal final', 'normal_final_rate'),
    ('Runtime errors', 'runtime_error_rate'),
    ('Avg wasted calls', 'average_wasted_tool_calls_after_sufficient_evidence'),
    ('Avg latency (ms)', 'average_latency_ms'),
)


def _number(value):
    return f'{value:.4f}' if isinstance(value, (float, int)) else 'unavailable'


def _comparison_table(models, summaries):
    lines = ['| Metric | ' + ' | '.join(_cell(m) for m in models) + ' |',
             '| --- | ' + ' | '.join('---:' for _ in models) + ' |']
    for label, field in COMPARISON_METRICS:
        if field == 'exploratory_success_rate' and not any(
                field in summary for summary in summaries.values() if summary):
            continue
        lines.append('| ' + label + ' | ' + ' | '.join(
            _number(summaries[m].get(field) if summaries.get(m) else None) for m in models) + ' |')
    return lines


def render_comparison(report) -> str:
    models, summaries = report['models'], report['summaries']
    lines = ['# Phase 1: Agent Model Ablation', '', f'Experiment: {_cell(report["experiment_id"])}',
             f'Status: {_cell(report["status"])}', '',
             'Only Agent model changes. Evaluation v1 source/tool success rules are unchanged.', '',
             *_comparison_table(models, summaries)]
    tasks = sorted({t for s in summaries.values() if s for t in s['by_task_type']})
    for task in tasks:
        lines += ['', f'## {task}', '', *_comparison_table(models, {
            m: summaries[m]['by_task_type'].get(task) if summaries[m] else None for m in models})]
    lines += ['', '## Trial stability', '',
              '| Model | Empirical pass@1 | Always pass | Flaky | Always fail | Incomplete |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for m in models:
        stat = summaries[m]['case_statistics'] if summaries[m] else None
        lines.append('| ' + _cell(m) + ' | ' + (' | '.join([
            _number(stat['empirical_pass_at_1']), *[str(len(stat[k + '_cases']))
             for k in ('always_pass', 'flaky', 'always_fail', 'incomplete')]]) if stat else
             'unavailable | unavailable | unavailable | unavailable | unavailable') + ' |')
    lines += ['', '## Per-case success probability', '',
              '| Case | ' + ' | '.join(_cell(m) for m in models) + ' |',
              '| --- | ' + ' | '.join('---:' for _ in models) + ' |']
    for case_id, case in report['case_comparison'].items():
        stats = [case['models'][m]['statistics'] for m in models]
        lines.append('| ' + case_id + ' | ' + ' | '.join(
            f'{s["successes"]}/{s["trials"]}' if s else 'unavailable' for s in stats) + ' |')
    for name, ids in report['case_groups'].items():
        lines += ['', f'## {name}', '', ', '.join(ids) or 'No qualifying cases / comparison unavailable.']
    representative = set(report['focused_cases'])
    for ids in report['case_groups'].values():
        representative.update(ids[:3])
    lines += ['', '## Representative and focused cases', '']
    for case_id in sorted(representative):
        case = report['case_comparison'][case_id]
        lines += [f'### {case_id}', '', f'Query: {_cell(case["query"])}',
                  f'Expected sources: {_cell(case["expected_sources"])}', '',
                  '| Model / trial | Success | Recall | Tools | Turns | Stop | Evidence turn | Wasted | Error |',
                  '| --- | --- | ---: | --- | ---: | --- | ---: | ---: | --- |']
        for model, entry in case['models'].items():
            for row in entry['trials']:
                b = row['behavior']
                lines.append('| ' + ' | '.join(_cell(v) for v in (
                    f'{model} / {row["trial"]}', row['success'], row['recall'], row['tool_sequence'],
                    row['turns'], row['stop_reason'], b['evidence_sufficient_turn'],
                    b['wasted_tool_calls_after_sufficient_evidence'], row['error'])) + ' |')
        lines.append('')
    lines += ['## Distributions and runtime errors', '']
    for model, s in summaries.items():
        if s:
            lines += [f'### {_cell(model)}', '', f'Tools: {_cell(s["tool_usage_distribution"])}',
                      f'Stops: {_cell(s["stop_reason_distribution"])}', f'Errors: {_cell(s["errors_by_type"])}', '']
            for error in s['errors']:
                lines.append(f'- {_cell(error["case_id"])} / trial {error["trial"]}: {_cell(error["error"])}')
    lines += ['', '## Interpretation limits', '']
    if report['status'] != 'completed':
        lines += ['The controlled formal matrix is incomplete or invalid. Model-capacity conclusions, a new fixed',
                  'baseline recommendation, and readiness for Embedding Ablation are not established.',
                  'Historical or smoke results must not fill missing formal cells.']
    else:
        lines += ['Compare source recall with success, final/max-turn rates, and conditional wasted-call means.',
                  'Similar recall with improved stopping supports a control/evidence-judgment hypothesis.',
                  'Shared low recall warrants retrieval/query/annotation investigation; shared failures warrant',
                  'tool/runtime/annotation inspection. Error reductions alone do not establish reasoning improvements.',
                  'Near-equal 9B/27B quality must be assessed alongside measured latency; memory is not measured.',
                  'Three trials give empirical case probabilities, not statistical or causal certainty.']
    for name, definition in report['definitions'].items():
        lines.append(f'- **{name}**: {definition}')
    lines += ['', 'Full traces and raw errors: each model directory’s results.jsonl.',
              'Exact search queries, read selectors and missing sources: comparison.json → focused_cases.',
              'Recall excludes no-retrieval and unavailable traces. Tool/turn means use observed traces.',
              'Unnecessary retrieval uses observed no-retrieval trials. Runtime-error/final rates use all trials.',
              'Conditional efficiency means exclude undefined trials; every denominator is saved in summary.json.']
    return '\n'.join(lines).rstrip() + '\n'
