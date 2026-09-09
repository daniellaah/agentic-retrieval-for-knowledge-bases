"""Pure deterministic agent evidence, behavior, and aggregate metrics."""

from collections import Counter
from collections.abc import Iterable, Sequence
from statistics import mean

from arkb.agent.state import AgentTrace
from arkb.evaluation.models import AgentEvalCase, AgentEvalResult, KNOWLEDGE_TOOLS


def extract_retrieved_sources(
    trace: AgentTrace, *, tool_names: Iterable[str] = KNOWLEDGE_TOOLS,
) -> tuple[str, ...]:
    """Union exact source identities from successful known tool observations.

    match/search return results[], read returns result. Unobserved calls, error
    payloads, arguments, prose/citations and unknown tools contribute nothing.
    A source is counted once, regardless of chunks, repeats, or read selectors.
    """
    names = set(tool_names) & KNOWLEDGE_TOOLS
    sources = set()
    for call in trace.tool_calls:
        observation = call.result
        if (call.name not in names or not isinstance(observation, dict)
                or 'error' in observation):
            continue
        evidence = ([observation.get('result')] if call.name == 'read'
                    else observation.get('results'))
        if not isinstance(evidence, list):
            continue
        for item in evidence:
            if isinstance(item, dict):
                source = item.get('source')
                if isinstance(source, str) and source.strip():
                    sources.add(source)
    return tuple(sorted(sources))


def evaluate_case(case: AgentEvalCase, trace: AgentTrace | None) -> AgentEvalResult:
    """Score outcome and explicit constraints, with no model or engine calls.

    All tasks require a normal final stop and obey declared tool/call limits.
    Direct reads require every target to be returned by read. Other retrieval
    tasks require the annotated recall and distinct expected-read count.
    no_retrieval forbids all knowledge tools even if annotations omit them.

    A runtime failure without a partial trace is still a failed trial, but its
    unobserved behavior remains null rather than inventing an empty trajectory.
    """
    if trace is None:
        return AgentEvalResult(
            case_id=case.id, task_type=case.task_type, success=False,
            expected_sources=case.expected_sources, source_recall=None,
            retrieved_sources=None, read_sources=None, tool_calls=None, tool_counts=None,
            tool_call_count=None, turn_count=None, forbidden_tool_violations=None,
            disallowed_tool_violations=None, max_tool_calls_exceeded=None,
            max_turn_failure=None, unnecessary_retrieval=None, stop_reason=None,
            failure_reasons=('trace_unavailable',),
        )
    if trace.query != case.query:
        raise ValueError('Trace query does not match the evaluation case.')
    retrieved = extract_retrieved_sources(trace)
    read = extract_retrieved_sources(trace, tool_names=('read',))
    expected = set(case.expected_sources)
    recall = len(expected.intersection(retrieved)) / len(expected) if expected else None
    sequence = tuple(call.name for call in trace.tool_calls)
    requested = Counter(sequence)
    counts = {name: requested[name] for name in sorted(set(requested) | KNOWLEDGE_TOOLS)}
    forbidden = {name: count for name, count in counts.items() if count and name in case.forbidden_tools}
    disallowed = {name: count for name, count in counts.items()
                  if count and case.allowed_tools is not None and name not in case.allowed_tools}
    exceeded = case.max_tool_calls is not None and len(sequence) > case.max_tool_calls
    unnecessary = (any(name in KNOWLEDGE_TOOLS for name in sequence)
                   if case.task_type == 'no_retrieval' else None)
    failures = []
    if trace.stop_reason != 'final':
        failures.append('not_final')
    if forbidden:
        failures.append('forbidden_tool')
    if disallowed:
        failures.append('disallowed_tool')
    if exceeded:
        failures.append('max_tool_calls_exceeded')
    if unnecessary:
        failures.append('unnecessary_retrieval')
    if expected and recall < case.min_source_recall:
        failures.append('source_recall_below_threshold')
    if case.task_type == 'direct_read' and not expected.issubset(read):
        failures.append('target_not_read')
    if len(expected.intersection(read)) < case.min_read_sources:
        failures.append('insufficient_read_sources')
    return AgentEvalResult(
        case_id=case.id, task_type=case.task_type, success=not failures,
        expected_sources=case.expected_sources, source_recall=recall,
        retrieved_sources=retrieved, read_sources=read, tool_calls=sequence,
        tool_counts=counts, tool_call_count=len(sequence), turn_count=trace.turns,
        forbidden_tool_violations=forbidden, disallowed_tool_violations=disallowed,
        max_tool_calls_exceeded=exceeded, max_turn_failure=trace.stop_reason == 'max_turns',
        unnecessary_retrieval=unnecessary, stop_reason=trace.stop_reason,
        failure_reasons=tuple(failures),
    )


def _summary(results: Sequence[AgentEvalResult]) -> dict:
    summary = {
        'total_cases': len({r.case_id for r in results}),
        'total_trials': len(results),
        'success_count': sum(r.success for r in results),
        'failure_count': sum(not r.success for r in results),
        'task_success_rate': mean(r.success for r in results),
        'no_retrieval_trials': sum(r.task_type == 'no_retrieval' for r in results),
        'missing_trace_trials': sum(r.stop_reason is None for r in results),
    }
    for metric, name in (
        ('source_recall', 'average_source_recall'),
        ('tool_call_count', 'average_tool_calls'), ('turn_count', 'average_turns'),
        ('max_turn_failure', 'max_turn_failure_rate'),
        ('unnecessary_retrieval', 'unnecessary_retrieval_rate'),
    ):
        values = [getattr(r, metric) for r in results if getattr(r, metric) is not None]
        summary[name] = mean(values) if values else None
        summary[metric + '_defined_trials'] = len(values)
    usage = Counter({name: 0 for name in KNOWLEDGE_TOOLS})
    stops = Counter({name: 0 for name in ('final', 'max_turns', 'error', 'other', 'unavailable')})
    for result in results:
        usage.update(result.tool_counts or {})
        reason = result.stop_reason
        stops['unavailable' if reason is None else reason if reason in ('final', 'max_turns', 'error') else 'other'] += 1
    summary['tool_usage_distribution'] = dict(sorted(usage.items()))
    summary['stop_reason_distribution'] = dict(sorted(stops.items()))
    return summary


def summarize_agent_results(results: Sequence[AgentEvalResult]) -> dict:
    """Trial-weighted macro means; undefined values excluded with denominators.

    Failures count in success rate. Recall excludes no_retrieval and unavailable
    traces; unnecessary retrieval uses only observed no_retrieval trials. This
    is neither pass-at-k nor all-trials-success. Tool usage counts requests.
    """
    if not results:
        raise ValueError('Agent summary requires at least one result.')
    types = {r.task_type for r in results}
    return {**_summary(results), 'by_task_type': {
        task_type: _summary([r for r in results if r.task_type == task_type])
        for task_type in sorted(types)
    }}
