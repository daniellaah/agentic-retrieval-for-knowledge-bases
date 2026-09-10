import pytest
from arkb.knowledge.models import Chunk, Note, ChunkRecord
from tests.evaluation.helpers import citation_fixture

def test_context_evaluation_measures_packed_span_union_and_coverage_retention():
    from arkb.generation.models import ContextConfig, GenerationCounter
    from arkb.generation.context import build_context
    from arkb.evaluation.generation import evaluate_context
    from arkb.retrieval.semantic import snapshot_result
    note = Note('Title', 'x' * 300, 'a.md')
    hits = []
    for index, (start, end) in enumerate([(0, 200), (150, 300)]):
        chunk = Chunk(note.content[start:end], note.title, note.source, index, start, end)
        record = ChunkRecord.from_note(chunk, note=note, vault_id='v')
        hits.append(snapshot_result(record, .9 - index / 10, 'snapshot'))
    counter = GenerationCounter('test', 'test-count', lambda messages: 10 + sum(len(m['content']) for m in messages))
    limit = counter(build_context('Q?', hits).messages)
    config = ContextConfig(limit + 20, 20, 0)
    case = {'required_source_groups': [['a.md']],
            'evidence_anchors': [{'source': 'a.md', 'body_start_char': 0, 'body_end_char': 300}]}
    result = evaluate_context('Q?', hits, case, config=config, counter=counter)
    built = result['metrics']
    assert built['section_coverage'] == built['section_coverage_retention'] == 1
    assert built['duplicate_span_fraction'] == 0
    assert built['fits_budget'] and built['prompt_tokens'] == limit
    assert built['body_characters'] == 300 and built['block_count'] == 1
    assert [(s['source_id'], s['source'], s['content']) for s in result['context']['citation_sources']] == [('S1', 'a.md', note.content)]
    from dataclasses import replace
    with pytest.raises(ValueError, match='one snapshot'):
        evaluate_context('Q?', [hits[0], replace(hits[1], metadata={**hits[1].metadata, 'index_version': 'other'})], case, config=config, counter=counter)


def test_citation_evaluation_retains_failed_raw_output_and_counts_failures():
    from unittest.mock import Mock
    from ollama import Client, ChatResponse, Message
    from arkb.evaluation.generation import evaluate_citation_context
    from arkb.evaluation.metrics import summarize_citations
    client = Mock(spec=Client)
    client.chat.return_value = ChatResponse(message=Message(role='assistant', content='{'), done_reason='length')
    row = evaluate_citation_context(citation_fixture(), client=client)
    assert row['success'] is False and row['raw_response'] == '{'
    assert row['error']['code'] == 'truncated_output'
    assert row['context']['citation_sources'][0]['content'] == 'Only small datasets were faster.'
    summary = summarize_citations([row])
    assert summary['error_count'] == 1 and summary['structure_valid'] == 0
    assert summary['supported_claim_rate'] is None
    assert summary['citation_id_validity_defined_cases'] == 0


def test_citation_case_records_prompt_budget_failure_without_calling_model():
    from unittest.mock import Mock
    from arkb.generation.models import ContextConfig
    from arkb.evaluation.generation import evaluate_citation_case
    context = citation_fixture()
    client = Mock()
    from tests.generation.helpers import source_hit
    row = evaluate_citation_case('Question?', [source_hit(0, 8)],
                                 config=ContextConfig(30, 10, 0), counter=context.counter, client=client)
    assert row['error']['code'] == 'context_budget' and row['context'] is None
    client.chat.assert_not_called()
