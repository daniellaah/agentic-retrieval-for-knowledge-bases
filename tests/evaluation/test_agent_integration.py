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


def test_complete_runner_loop_uses_published_index_real_runtime_and_all_task_types(
    tmp_path, qdrant, qdrant_config,
):
    from dataclasses import asdict
    import json
    from unittest.mock import Mock

    from ollama import Client, EmbedResponse
    from tokenizers import Tokenizer, models

    from arkb.evaluation.agent_runner import run_agent_evaluation
    from arkb.evaluation.models import AgentEvalCase, AgentEvalConfig
    from arkb.knowledge.documents import load_notes
    from arkb.knowledge.indexing import build_index
    from arkb.knowledge.models import EmbeddingSpec
    from arkb.knowledge.sqlite import SQLiteStorage

    wanted = ('exact_002', 'semantic_006', 'read_001', 'explore_003', 'qa_004', 'none_001')
    cases = [c for c in load_agent_eval_dataset(DATASET) if c.id in wanted]
    cases.append(AgentEvalCase(id='failed_read', query='Read the memory note', task_type='direct_read',
                               expected_sources=('34_agent_memory_lifecycle.md',)))
    sources = {s for c in cases for s in c.expected_sources}
    notes = tmp_path / 'notes'
    notes.mkdir()
    for source in sources:
        (notes / source).write_bytes((NOTES / source).read_bytes())
    embedding = Mock(spec=Client)
    embedding.embed.side_effect = lambda **kw: EmbedResponse(embeddings=[[1., 0.] for _ in kw['input']])
    tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0}, unk_token='[UNK]'))
    spec = EmbeddingSpec(model='fixture', model_revision='fixed', dimensions=2,
                         document_template='title-body-v1')
    db = tmp_path / 'index.sqlite'
    with SQLiteStorage(db) as storage:
        indexed = build_index(storage, load_notes(notes), spec=spec, vault_id='eval',
                              client=embedding, tokenizer=tokenizer, max_input_tokens=100,
                              chunking='none', qdrant_client=qdrant, qdrant_config=qdrant_config,
                              source_scope=str(notes.resolve()))
    embedding.reset_mock()
    dataset = tmp_path / 'cases.jsonl'
    dataset.write_text(''.join(json.dumps(asdict(c), ensure_ascii=False) + '\n' for c in cases), encoding='utf-8')
    steps = []
    for case in cases:
        for trial in range(2):
            if case.id == 'exact_002':
                steps.append(reply(calls=[tool_call('match', query='Reciprocal Rank Fusion')]))
            elif case.id == 'semantic_006':
                steps.append(reply(calls=[tool_call('search', query='cross-encoder', mode='bm25')]))
            elif case.id == 'read_001':
                steps.append(reply(calls=[tool_call('read', source=case.expected_sources[0])]))
            elif case.id == 'explore_003':
                steps.append(reply(calls=[tool_call('search', query='graph', mode='bm25')]))
                steps.extend(reply(calls=[tool_call('read', source=s)]) for s in case.expected_sources)
            elif case.id == 'qa_004':
                steps.append(reply(calls=[tool_call('search', query='trials', mode='bm25')]))
            elif case.id == 'failed_read':
                steps.append(reply(calls=[tool_call('read', source='missing.md')]))
                continue
            steps.append(reply('Done'))
    client = ScriptedModel(*steps)
    with Runtime() as runtime:
        run = run_agent_evaluation(AgentEvalConfig(
            dataset_path=dataset, output_dir=tmp_path / 'report', notes_dir=notes,
            db=db, vault_id='eval', model='scripted', num_trials=2,
        ), runtime=runtime, client=client)
        # Evaluation leaves an injected runtime open.
        assert runtime.ask('hello', db=db, vault_id='eval', notes_dir=notes,
                           client=ScriptedModel(reply('Hello')), model='scripted').stop_reason == 'final'
    assert run.summary['total_cases'] == 7 and run.summary['total_trials'] == 14
    assert run.summary['success_count'] == 12 and run.summary['failure_count'] == 2
    assert len(run.summary['by_task_type']) == 6
    assert sum(len(r['messages']) == 2 for r in client.requests) == 14
    assert [r.metrics.stop_reason for r in run.results[-2:]] == ['error', 'error']
    assert all(r.trace.tool_calls[0].result is None for r in run.results[-2:])
    metadata = json.loads((run.output_dir / 'run_metadata.json').read_text())
    snapshot = metadata['knowledge_before']['snapshot']
    assert snapshot['manifest']['index_version'] == indexed.manifest.index_version
    assert len(snapshot['corpus']) == len(sources)
    assert metadata['knowledge_changed'] is False
    rows = [json.loads(line) for line in (run.output_dir / 'results.jsonl').read_text().splitlines()]
    assert len(rows) == 14 and rows[-1]['error']['type'] == 'LookupError'
    assert 'failed_read / trial 1' in (run.output_dir / 'report.md').read_text()
    embedding.embed.assert_not_called()
