"""Fixed engine requests and source-level metrics; no agent execution."""

from dataclasses import replace
from html import escape
from time import perf_counter

from arkb.evaluation.metrics import ranking_metrics
from arkb.evaluation.models import BASELINE_MODES, BaselineEvalResult
from arkb.retrieval.models import SearchResponse


SCHEMA_VERSION = 'retrieval-baselines-v1'


def source_metrics(expected_sources, ranked_sources, *, ks):
    """Binary v1 labels, using the existing metric formulas at every cutoff."""
    values = {}
    for k in ks:
        result = ranking_metrics(dict.fromkeys(expected_sources, 1), ranked_sources, k=k)
        values[f'recall_at_{k}'] = result['recall_at_k']
        values[f'ndcg_at_{k}'] = result['ndcg_at_k']
        values['mrr'] = result['mrr']
    return values


def execute_baseline(case, name, *, engine, top_k, ks, index_version, known_sources):
    """Pass only the original query and fixed engine options, never labels.

    Errors have no ranking or scores; a successful empty response has zero
    coverage. No-retrieval rows are retained without a request or fake latency.
    """
    mode, rerank = BASELINE_MODES[name]
    row = BaselineEvalResult(case_id=case.id, query=case.query, task_type=case.task_type,
        expected_sources=case.expected_sources, baseline=name, mode=mode, rerank=rerank,
        status='not_applicable', retrieved_sources=None, ranked_sources=None,
        metrics=source_metrics((), (), ks=ks), latency_ms=None, response=None)
    if case.task_type == 'no_retrieval':
        return row
    started = perf_counter()
    try:
        response = engine.search(case.query, mode=mode, rerank=rerank, top_k=top_k, filters=None)
    except Exception as failure:
        return replace(row, status='error', latency_ms=(perf_counter() - started) * 1000,
                       error={'type': type(failure).__name__, 'message': str(failure)})
    elapsed = (perf_counter() - started) * 1000
    try:
        if (not isinstance(response, SearchResponse) or response.query != case.query
                or response.index_id != index_version or len(response.results) > top_k
                or len({hit.identity for hit in response.results}) != len(response.results)
                or any(hit.source not in known_sources for hit in response.results)):
            raise ValueError('Engine returned an invalid ranking or evidence outside the pinned snapshot.')
        sources = tuple(hit.source for hit in response.results)
        ranked = tuple(dict.fromkeys(sources))
        return replace(row, status='ok', retrieved_sources=sources, ranked_sources=ranked,
            response=response, latency_ms=elapsed,
            metrics=source_metrics(case.expected_sources, ranked, ks=ks))
    except ValueError as failure:
        return replace(row, status='error', response=response if isinstance(response, SearchResponse) else None,
            latency_ms=elapsed, error={'type': type(failure).__name__, 'message': str(failure)})


def _aggregate(rows, metric_names):
    values = {key: [row.metrics[key] for row in rows if row.metrics[key] is not None] for key in metric_names}
    times = [row.latency_ms for row in rows if row.latency_ms is not None]
    return {
        'case_count': len(rows), 'applicable_cases': sum(row.status != 'not_applicable' for row in rows),
        'not_applicable_cases': sum(row.status == 'not_applicable' for row in rows),
        'evaluated_cases': sum(row.status == 'ok' for row in rows),
        'failure_count': sum(row.status == 'error' for row in rows),
        'metrics': {key: sum(v) / len(v) if v else None for key, v in values.items()},
        'metric_denominators': {key: len(v) for key, v in values.items()},
        'mean_latency_ms': sum(times) / len(times) if times else None,
        'latency_defined_cases': len(times),
    }


def summarize_baselines(rows, *, config):
    metric_names = source_metrics((), (), ks=config.metric_ks).keys()
    return {
        'total_cases': len({row.case_id for row in rows}), 'total_results': len(rows),
        'failure_count': sum(row.status == 'error' for row in rows),
        'settings': {'baselines': config.baselines, 'top_k': config.top_k, 'metric_ks': config.metric_ks,
            'ranking_unit': 'unique sources in first-occurrence order within top_k engine results',
            'mrr_depth': 'all unique sources in the single returned list',
            'latency': 'engine request including embedding/reranking inference and failed requests; excludes setup and scoring'},
        'overall': {name: _aggregate([r for r in rows if r.baseline == name], metric_names) for name in config.baselines},
        'by_task_type': {task: {name: _aggregate([r for r in rows if r.baseline == name and r.task_type == task], metric_names)
                               for name in config.baselines} for task in sorted({r.task_type for r in rows})},
        'failed_cases': [{'case_id': r.case_id, 'query': r.query, 'task_type': r.task_type,
                          'baseline': r.baseline, 'error': r.error} for r in rows if r.error is not None],
    }


def render_baseline_report(summary):
    def cell(value):
        if value is None:
            return 'N/A'
        if isinstance(value, float):
            return f'{value:.4f}'
        return escape(str(value)).replace('|', '\\|').replace('\n', '<br>')
    ks, top_k = summary['settings']['metric_ks'], summary['settings']['top_k']
    columns = [(f'R@{k}', f'recall_at_{k}') for k in ks] + [('MRR', 'mrr'), (f'nDCG@{top_k}', f'ndcg_at_{top_k}')]
    def table(groups):
        lines = ['| Method | Evaluated / applicable | N/A | Errors | ' + ' | '.join(c for c, _ in columns) + ' | Latency ms |',
                 '| --- | ---: | ---: | ---: | ' + ' | '.join('---:' for _ in columns) + ' | ---: |']
        for name, group in groups.items():
            lines.append(f'| {name} | {group["evaluated_cases"]} / {group["applicable_cases"]} | '
                f'{group["not_applicable_cases"]} | {group["failure_count"]} | '
                + ' | '.join(cell(group['metrics'][key]) for _, key in columns)
                + f' | {cell(group["mean_latency_ms"])} |')
        return lines
    lines = ['# Agent Evaluation v1: Deterministic Retrieval Baselines', '', f'Run: {cell(summary["run_id"])}', '',
        'One original-query engine request per applicable case and baseline; binary expected_sources labels.',
        f'top_k={top_k} limits returned chunks. Sources collapse in first-occurrence order; no refill requests.',
        'R@K/nDCG@K use the deduplicated source ranking; MRR uses that entire returned ranking.',
        'Ranking means exclude errors and no_retrieval. Successful empty rankings count as zero.',
        'Latency includes attempted requests, including failures; setup is separate in run_metadata.json.',
        'Every metric denominator is saved in summary.json.', '', '## Overall', '', *table(summary['overall'])]
    for task, groups in summary['by_task_type'].items():
        lines += ['', f'## {task}', '', *table(groups)]
    lines += ['', '## Comparison with Agent outcomes', '',
        'Baseline source recall can be compared with Agent trajectory unique-source recall on identical cases,',
        'snapshot and defined denominators, with the one-request versus accumulated-evidence budget stated.',
        'Agent task success also checks tool/read/turn constraints; it is not a baseline ranking metric.',
        'MRR/nDCG apply only to baseline rankings. No synthetic Agent ranking or baseline task success is created.',
        'Agent tool calls, turns, token usage and answer quality are not inferred from these retrieval results.']
    if summary.get('knowledge_changed'):
        lines += ['', 'Knowledge inputs changed during the run; this is not a valid fixed-input control group.']
    lines += ['', '## Retrieval errors', '']
    lines += [f'- {cell(r["case_id"])} / {r["baseline"]}: {cell(r["error"]["type"])} — {cell(r["error"]["message"])}'
              for r in summary['failed_cases']] or ['No retrieval errors.']
    return '\n'.join(lines) + '\n'
