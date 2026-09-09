"""Cross-module evaluation contracts with real tools and scripted models.

These deterministic integration checks need no model or external service, so
they deliberately do not use the repository's external-service integration marker.
"""

import pytest

from arkb.evaluation.datasets import load_agent_eval_dataset
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
    assert not (tmp_path / 'absent.sqlite').exists()
