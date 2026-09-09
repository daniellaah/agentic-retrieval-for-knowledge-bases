from dataclasses import asdict, replace
import json

import pytest

from arkb.agent.state import AgentToolTrace, AgentTrace
from arkb.evaluation.agent import evaluate_case, extract_retrieved_sources, summarize_agent_results
from arkb.evaluation.models import AgentEvalCase


def case(task_type='semantic_discovery', **options):
    return AgentEvalCase(**{'id': 'case', 'query': 'Find evidence', 'task_type': task_type,
                           'expected_sources': ('a.md', 'b.md'), **options})


def observation(name, *sources, turn=1, **arguments):
    evidence = [{'source': source, 'content': 'Evidence'} for source in sources]
    result = {'result': evidence[0]} if name == 'read' else {'results': evidence}
    return AgentToolTrace(turn, name, arguments, result)


def trace(*calls, reason='final', turns=2, query='Find evidence'):
    return AgentTrace(query, turns, list(calls), 'Answer' if reason == 'final' else None, reason)


@pytest.mark.parametrize('sources, recall, success', [
    (('a.md', 'b.md'), 1, True), (('a.md',), .5, True), (('other.md',), 0, False),
    ((), 0, False),
])
def test_perfect_partial_and_zero_source_recall(sources, recall, success):
    result = evaluate_case(case(min_source_recall=.5), trace(observation('search', *sources)))
    assert result.source_recall == recall
    assert result.success is success
    assert result.tool_counts == {'match': 0, 'read': 0, 'search': 1}


def test_exact_lookup_requires_every_expected_source_but_does_not_require_match():
    label = case('exact_lookup')
    assert not evaluate_case(label, trace(observation('match', 'a.md'))).success
    assert evaluate_case(label, trace(observation('search', 'a.md', 'b.md'))).success


def test_source_extraction_uses_only_known_successful_observations_and_never_mutates_trace():
    run = trace(
        observation('search', 'b.md', 'a.md', 'a.md'),
        observation('read', 'a.md', document_id='returned-id'),
        AgentToolTrace(2, 'read', {'source': 'never.md'}, None),
        AgentToolTrace(2, 'search', {}, {'error': 'failed', 'results': [{'source': 'error.md'}]}),
        observation('unknown', 'unknown.md'),
        AgentToolTrace(2, 'match', {}, {'results': [None, {'source': ''}, {'source': 4}]}),
        AgentToolTrace(2, 'search', {}, {'results': {'source': 'bad-shape.md'}}),
        AgentToolTrace(2, 'read', {'source': 'argument.md'}, {}),
    )
    run = replace(run, final_response='I used never.md and citation.md')
    before = asdict(run)
    assert extract_retrieved_sources(run) == ('a.md', 'b.md')
    assert extract_retrieved_sources(run, tool_names=('read',)) == ('a.md',)
    result = evaluate_case(case(), run)
    assert result.source_recall == 1
    assert result.tool_call_count == 8  # Includes requested but unobserved calls.
    assert result.tool_counts['unknown'] == 1
    assert asdict(run) == before
    assert json.loads(json.dumps(asdict(result)))['retrieved_sources'] == ['a.md', 'b.md']


def test_no_retrieval_success_has_undefined_recall_and_no_tools():
    result = evaluate_case(case('no_retrieval', expected_sources=()), trace(turns=1))
    assert result.success and result.source_recall is None
    assert result.unnecessary_retrieval is False
    assert result.tool_call_count == 0 and result.turn_count == 1


@pytest.mark.parametrize('name', ['match', 'search', 'read'])
def test_no_retrieval_request_is_unnecessary_even_if_unexecuted_or_not_explicitly_forbidden(name):
    run = trace(AgentToolTrace(1, name, {}, None))
    result = evaluate_case(case('no_retrieval', expected_sources=()), run)
    assert not result.success and result.unnecessary_retrieval is True
    assert result.forbidden_tool_violations == {}


def test_forbidden_allowed_and_max_calls_constraints_count_each_request():
    run = trace(observation('match', 'a.md'), observation('match', 'b.md'), observation('read', 'a.md'))
    result = evaluate_case(case(forbidden_tools=('match',), allowed_tools=('read',), max_tool_calls=2), run)
    assert result.forbidden_tool_violations == {'match': 2}
    assert result.disallowed_tool_violations == {'match': 2}
    assert result.max_tool_calls_exceeded and not result.success
    assert evaluate_case(case(max_tool_calls=3), run).success
    assert not evaluate_case(case(allowed_tools=()), run).success


@pytest.mark.parametrize('reason', ['max_turns', 'error', 'unexpected_future_reason'])
def test_abnormal_stop_never_succeeds_even_with_perfect_evidence(reason):
    result = evaluate_case(case(), trace(observation('search', 'a.md', 'b.md'), reason=reason, turns=8))
    assert not result.success and result.source_recall == 1
    assert result.stop_reason == reason and result.turn_count == 8
    assert result.max_turn_failure is (reason == 'max_turns')
    group = 'other' if reason == 'unexpected_future_reason' else reason
    assert summarize_agent_results([result])['stop_reason_distribution'][group] == 1


def test_direct_read_needs_observed_targets_not_search_hits_or_read_arguments():
    label = case('direct_read')
    run = trace(observation('search', 'a.md', 'b.md'),
                AgentToolTrace(1, 'read', {'source': 'b.md'}, None))
    result = evaluate_case(label, run)
    assert result.source_recall == 1 and not result.success
    assert 'target_not_read' in result.failure_reasons
    run = trace(observation('read', 'a.md', document_id='id-a'),
                observation('read', 'b.md', start_char=1, end_char=10))
    assert evaluate_case(label, run).success


@pytest.mark.parametrize('calls', [
    [observation('search', 'a.md'), observation('read', 'a.md'),
     observation('search', 'b.md', turn=2), observation('read', 'b.md', turn=3)],
    [observation('match', 'b.md', 'a.md'), observation('read', 'b.md'), observation('read', 'a.md')],
    [observation('read', 'a.md'), observation('read', 'b.md')],
])
def test_multiple_valid_exploratory_trajectories(calls):
    result = evaluate_case(case('exploratory_retrieval', min_read_sources=2), trace(*calls, turns=4))
    assert result.success
    assert result.turn_count == 4  # Uses trace.turns, not inferred tool batches.


def test_repeated_or_unrelated_reads_do_not_satisfy_exploration_read_constraint():
    run = trace(observation('search', 'a.md', 'b.md'), observation('read', 'a.md'),
                observation('read', 'a.md'), observation('read', 'other.md'))
    result = evaluate_case(case('exploratory_retrieval', min_read_sources=2), run)
    assert result.source_recall == 1 and not result.success
    assert result.failure_reasons == ('insufficient_read_sources',)


def test_knowledge_qa_checks_evidence_and_final_without_scoring_prose():
    run = trace(observation('match', 'a.md', 'b.md'))
    assert evaluate_case(case('knowledge_qa'), replace(run, final_response='Arbitrary wording')).success
    assert not evaluate_case(case('knowledge_qa'), trace()).success


def test_missing_trace_is_failed_with_unavailable_behavior_and_mismatched_trace_is_rejected():
    result = evaluate_case(case(), None)
    assert not result.success and result.stop_reason is None
    assert result.source_recall is result.turn_count is result.tool_call_count is None
    assert result.retrieved_sources is result.tool_calls is None
    with pytest.raises(ValueError, match='query'):
        evaluate_case(case(), trace(query='Different case'))


def test_aggregate_counts_trials_including_failures_and_exposes_every_denominator():
    semantic = case()
    direct = case('no_retrieval', id='none', expected_sources=())
    results = [
        evaluate_case(semantic, trace(observation('search', 'a.md', 'b.md'))),
        evaluate_case(semantic, trace(observation('search', 'a.md'), reason='max_turns', turns=3)),
        evaluate_case(direct, trace(turns=1)),
        evaluate_case(direct, trace(observation('read', 'a.md'))),
        evaluate_case(direct, None),
    ]
    summary = summarize_agent_results(results)
    assert summary['total_cases'] == 2 and summary['total_trials'] == 5
    assert summary['task_success_rate'] == .4
    assert summary['average_source_recall'] == .75 and summary['source_recall_defined_trials'] == 2
    assert summary['average_tool_calls'] == .75 and summary['average_turns'] == 2
    assert summary['tool_call_count_defined_trials'] == 4
    assert summary['max_turn_failure_rate'] == .25 and summary['max_turn_failure_defined_trials'] == 4
    assert summary['unnecessary_retrieval_rate'] == .5
    assert summary['unnecessary_retrieval_defined_trials'] == 2 and summary['no_retrieval_trials'] == 3
    assert summary['missing_trace_trials'] == 1
    assert summary['tool_usage_distribution'] == {'match': 0, 'read': 1, 'search': 2}
    assert summary['stop_reason_distribution'] == {'error': 0, 'final': 3, 'max_turns': 1, 'other': 0, 'unavailable': 1}
    assert summary['by_task_type']['semantic_discovery']['task_success_rate'] == .5
    assert summary['by_task_type']['no_retrieval']['average_source_recall'] is None
    assert summary['by_task_type']['no_retrieval']['task_success_rate'] == pytest.approx(1 / 3)


def test_aggregate_without_defined_observations_keeps_nulls():
    summary = summarize_agent_results([evaluate_case(case(), None)])
    assert summary['task_success_rate'] == 0
    assert summary['average_source_recall'] is summary['average_turns'] is None
    assert summary['unnecessary_retrieval_rate'] is summary['max_turn_failure_rate'] is None
    assert summary['source_recall_defined_trials'] == 0
    with pytest.raises(ValueError, match='at least one'):
        summarize_agent_results([])
