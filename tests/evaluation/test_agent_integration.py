"""Cross-module evaluation contracts with real tools and scripted models.

These deterministic integration checks need no model or external service, so
they deliberately do not use the repository's external-service integration marker.
"""

import pytest

from arkb.evaluation.datasets import load_agent_eval_dataset
from arkb.evaluation.agent import evaluate_case, summarize_agent_results
from arkb.knowledge.documents import DocumentAccess
from arkb.retrieval.exact import ExactRetriever
from arkb.runtime import Runtime
from tests.agent.helpers import ScriptedModel, reply, tool_call
from tests.evaluation.test_agent_dataset import DATASET, NOTES


def test_all_curated_sources_are_readable_by_real_document_access():
    cases = load_agent_eval_dataset(DATASET, notes_dir=NOTES)
    documents = DocumentAccess(NOTES, vault_id='eval')
    for source in sorted({s for c in cases for s in c.expected_sources}):
        result = documents.read(source=source)
        assert result.chunk.source == source
        assert result.chunk.content


@pytest.mark.parametrize('index', range(6))
def test_exact_labels_equal_real_body_occurrences(index):
    case = [c for c in load_agent_eval_dataset(DATASET) if c.task_type == 'exact_lookup'][index]
    term = case.query.split('“')[1].split('”')[0]
    exact = ExactRetriever(DocumentAccess(NOTES, vault_id='eval'))
    response = exact.search(term, top_k=10000)
    assert {hit.source for hit in response.results} == set(case.expected_sources)


@pytest.mark.parametrize('index', range(6))
def test_direct_read_dataset_runs_through_real_runtime_and_trace(tmp_path, index):
    case = [c for c in load_agent_eval_dataset(DATASET) if c.task_type == 'direct_read'][index]
    client = ScriptedModel(reply(calls=[tool_call('read', source=case.expected_sources[0])]), reply('Done'))
    with Runtime() as runtime:
        result = runtime.ask(case.query, db=tmp_path / 'absent.sqlite', notes_dir=NOTES,
                             model='scripted', client=client)
    trace = result.trace
    assert trace.query == case.query
    assert trace.turns == 2 and trace.stop_reason == 'final'
    assert trace.tool_calls[0].result['result']['source'] == case.expected_sources[0]
    assert evaluate_case(case, trace).success
    assert not (tmp_path / 'absent.sqlite').exists()


def test_real_search_read_search_read_and_partial_tool_error_feed_metrics():
    from arkb.evaluation.models import AgentEvalCase
    from arkb.retrieval.bm25 import BM25Retriever
    from arkb.retrieval.engine import RetrievalEngine

    documents = DocumentAccess(NOTES, vault_id='eval')
    engine = RetrievalEngine(bm25=BM25Retriever(list(documents.records())))
    case = AgentEvalCase(id='explore', query='Find memory and compaction material',
                         task_type='exploratory_retrieval', min_read_sources=2,
                         expected_sources=('06_context_compaction.md', '34_agent_memory_lifecycle.md'))

    def read_first_hit(messages):
        import json
        hit = json.loads(messages[-1]['content'])['results'][0]
        return reply(calls=[tool_call('read', document_id=hit['document_id'])])

    client = ScriptedModel(
        reply(calls=[tool_call('search', query='LangMem', mode='bm25')]), read_first_hit,
        reply(calls=[tool_call('search', query='compaction', mode='bm25', source='06_context_compaction.md')]),
        read_first_hit, reply('Done'),
    )
    with Runtime() as runtime:
        tools = runtime.agent_tools(engine=engine, directory=NOTES, vault_id='eval', mode='bm25')
        result = runtime.run_agent(case.query, tools=tools, client=client, model='scripted')
        metrics = evaluate_case(case, result.trace)
        assert metrics.success and metrics.source_recall == 1
        assert metrics.tool_calls == ('search', 'read', 'search', 'read')
        assert metrics.turn_count == 5
        broken = ScriptedModel(reply(calls=[
            tool_call('read', source=case.expected_sources[0]),
            tool_call('read', source='absent.md'),
            tool_call('read', source=case.expected_sources[1]),
        ]))
        with pytest.raises(LookupError) as raised:
            runtime.run_agent(case.query, tools=tools, client=broken, model='scripted')
        failed = evaluate_case(case, raised.value.agent_result.trace)
    assert not failed.success and failed.stop_reason == 'error'
    assert failed.source_recall == .5
    assert failed.tool_call_count == 3
    assert failed.read_sources == (case.expected_sources[0],)
    assert summarize_agent_results([metrics, failed])['task_success_rate'] == .5
