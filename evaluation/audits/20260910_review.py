"""Read-only inventory and deterministic replay for the 2026-09-10 eval review.

Run from the repository root with .venv/bin/python. Prints JSON to stdout.
Does not call models, open an index, alter historical artifacts, or judge answers.
Historical observations are replayed with the current v1 scoring contract.
"""

from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
from statistics import mean, median
import subprocess

from arkb.agent.state import AgentToolTrace, AgentTrace
from arkb.evaluation.agent_metrics import evaluate_case, summarize_agent_results
from arkb.evaluation.baselines import source_metrics
from arkb.evaluation.datasets import load_agent_eval_dataset, source_hashes


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / 'evaluation/data/agent_v1.jsonl'
FORMAL = ROOT / 'evaluation/results/agent_model_ablation/20260909-phase1-formal'
BASELINE = ROOT / 'evaluation/results/deterministic-baselines-20260909'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(value):
    return json.loads(json.dumps(value))


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def assert_summary_fields(actual, saved):
    """Historical ablation summaries also contain separate behavioral analyses."""
    if isinstance(actual, dict):
        for key, value in actual.items():
            assert_summary_fields(value, saved[key])
    else:
        assert normalized(actual) == saved, (actual, saved)


def main():
    cases = load_agent_eval_dataset(DATASET, notes_dir=ROOT / 'example_notes')
    by_id = {case.id: case for case in cases}
    notes = sorted((ROOT / 'example_notes').glob('*.md'))
    lengths = [len(p.read_text()) for p in notes]
    positives = Counter(source for case in cases for source in case.expected_sources)
    current_hashes = source_hashes(ROOT / 'src/arkb')
    live_hashes = {p.name: sha(p) for p in notes}
    report = {
        'audit_date': '2026-09-10',
        'source_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        'audit_script_sha256': sha(Path(__file__)),
        'evaluator_source_hashes': {k: v for k, v in current_hashes.items() if k.startswith('evaluation/')},
        'scope': 'Offline v1 score replay; no live inference, answer grading, or index/vector verification.',
        'dataset': {
            'sha256': sha(DATASET), 'cases': len(cases),
            'by_task_type': dict(Counter(c.task_type for c in cases)),
            'queries_containing_han': sum(bool(re.search(r'[\u4e00-\u9fff]', c.query)) for c in cases),
            'positive_labeled_documents': len(positives),
            'documents_without_positive_labels': [p.name for p in notes if p.name not in positives],
            'positive_label_frequency': dict(positives.most_common()),
        },
        'corpus': {'documents': len(notes), 'raw_markdown_characters': sum(lengths),
                   'raw_character_length_min_median_max': [min(lengths), median(lengths), max(lengths)],
                   'documents_containing_han': sum(bool(re.search(r'[\u4e00-\u9fff]', p.read_text())) for p in notes)},
        'agent': {}, 'baseline': {}, 'inputs': {},
    }
    snapshots = []
    def verify_inputs(directory):
        metadata = json.loads((directory / 'run_metadata.json').read_text())
        assert metadata['dataset_sha256'] == sha(DATASET)
        assert sha(directory / 'cases.jsonl') == sha(DATASET)
        assert metadata['knowledge_before'] == metadata['knowledge_after']
        assert metadata['knowledge_before']['notes_sha256'] == live_hashes
        snapshot = metadata['knowledge_before']['snapshot']
        snapshots.append(snapshot)
        report['inputs'][str(directory.relative_to(ROOT))] = {
            'results_sha256': sha(directory / 'results.jsonl'),
            'run_metadata_sha256': sha(directory / 'run_metadata.json'),
            'dataset_matches': True, 'live_corpus_hashes_match': True,
            'historical_boundary_fingerprints_match': True,
            'snapshot_id': snapshot['manifest']['index_version'],
            'snapshot_chunks': len(snapshot['corpus']),
            'historical_paths_changed_or_moved': [k for k, v in metadata['source_hashes'].items()
                                                 if current_hashes.get(k) != v],
        }

    semantic_tasks = {'semantic_discovery', 'exploratory_retrieval', 'knowledge_qa'}
    for model in ('4b', '9b', '27b'):
        directory = FORMAL / f'qwen3.5-{model}'
        verify_inputs(directory)
        entries = rows(directory / 'results.jsonl')
        assert {(r['case']['id'], r['trial']) for r in entries} == {(c.id, t) for c in cases for t in range(3)}
        assert len(entries) == len(cases) * 3
        scored, grouped = [], defaultdict(list)
        for row in entries:
            case = by_id[row['case']['id']]
            assert normalized(asdict(case)) == row['case']
            payload = row['trace']
            trace = AgentTrace(**{**payload, 'tool_calls': [AgentToolTrace(**c) for c in payload['tool_calls']]}) if payload else None
            metric = evaluate_case(case, trace)
            assert normalized(asdict(metric)) == row['metrics'], (model, case.id, row['trial'])
            scored.append(metric)
            grouped[case.id].append(row)
        summary = summarize_agent_results(scored)
        saved = json.loads((directory / 'summary.json').read_text())
        assert_summary_fields(summary, saved)
        identical = sum(len({json.dumps({'metrics': r['metrics'], 'calls': [
            {'name': c['name'], 'arguments': c['arguments'], 'turn': c['turn']} for c in r['trace']['tool_calls']]}, sort_keys=True)
                            for r in rs}) == 1 for rs in grouped.values())
        report['agent'][model] = {
            'replayed_rows': len(entries), 'all_case_metrics_and_summary_match': True,
            'summary': summary, 'cases_with_identical_metrics_and_call_arguments': identical,
            'semantic_explore_qa_recall': mean(m.source_recall for m in scored if m.task_type in semantic_tasks),
            'mean_elapsed_ms_all_tasks': mean(r['runtime_metadata']['elapsed_ms'] for r in entries),
            'failed_case_ids': sorted({m.case_id for m in scored if not m.success}),
        }

    verify_inputs(BASELINE)
    entries = rows(BASELINE / 'results.jsonl')
    methods = {'bm25', 'semantic', 'hybrid', 'hybrid_rerank'}
    assert len(entries) == len(cases) * len(methods)
    assert {(r['case_id'], r['baseline']) for r in entries} == {(c.id, m) for c in cases for m in methods}
    saved = json.loads((BASELINE / 'summary.json').read_text())
    for row in entries:
        case = by_id[row['case_id']]
        assert row['query'] == case.query and row['expected_sources'] == list(case.expected_sources)
        assert row['task_type'] == case.task_type
        if row['status'] == 'ok':
            sources = [h['source'] for h in row['response']['results']]
            ranked = list(dict.fromkeys(sources))
            assert row['retrieved_sources'] == sources and row['ranked_sources'] == ranked
            metrics = source_metrics(case.expected_sources, ranked, ks=(1, 3, 5, 10))
            assert row['metrics'] == metrics
        else:
            assert row['status'] == 'not_applicable' and case.task_type == 'no_retrieval'
            assert row['response'] is None and all(v is None for v in row['metrics'].values())
    for method in sorted(methods):
        subset = [r for r in entries if r['baseline'] == method and r['status'] == 'ok']
        averages = {k: mean(r['metrics'][k] for r in subset) for k in subset[0]['metrics']}
        assert all(abs(v - saved['overall'][method]['metrics'][k]) < 1e-12 for k, v in averages.items())
        report['baseline'][method] = {
            'scored_cases': len(subset), 'not_applicable': 6, 'error_cases': 0,
            'all_row_metrics_and_overall_metric_means_match': True, 'metrics': averages,
            'semantic_explore_qa_recall': mean(r['metrics']['recall_at_10'] for r in subset if r['task_type'] in semantic_tasks),
            'unique_sources_min_mean_max': [min(len(r['ranked_sources']) for r in subset), mean(len(r['ranked_sources']) for r in subset), max(len(r['ranked_sources']) for r in subset)],
            'mean_latency_ms': mean(r['latency_ms'] for r in subset),
        }
    assert all(snapshot == snapshots[0] for snapshot in snapshots)
    report['all_four_run_metadata_snapshots_match'] = True
    report['score_replay_total_rows'] = 520
    # Synthetic metric-contract probes, never included in benchmark aggregates.
    probes = {}
    for case_id, name, observation in (
        ('qa_001', 'search', {'results': [{'source': by_id['qa_001'].expected_sources[0], 'content': 'irrelevant text'}]}),
        ('read_001', 'read', {'result': {'source': by_id['read_001'].expected_sources[0], 'content': 'irrelevant text'}}),
    ):
        case = by_id[case_id]
        trace = AgentTrace(case.query, 2, [AgentToolTrace(1, name, {}, observation)], 'deliberately unsupported answer', 'final')
        probes[case_id] = {'synthetic_only': True, 'v1_success': evaluate_case(case, trace).success,
                           'meaning': 'v1 verifies source membership and behavior, not evidence/answer semantics'}
    report['synthetic_metric_contract_probes'] = probes
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
