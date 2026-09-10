import json
from unittest.mock import Mock

from ollama import ChatResponse, Client, Message, ResponseError
import pytest

from arkb.knowledge.chunking import chunk_notes, whole_note_chunks
from arkb.generation.generate import generate_cited_answer, CitedGenerationError
from arkb.knowledge.models import Note
from arkb.retrieval.semantic import snapshot_result
from arkb.knowledge.models import ChunkRecord


@pytest.fixture
def client() -> Mock:
    client = Mock(spec=Client)
    client.chat.return_value = ChatResponse(
        message=Message(role="assistant", content=json.dumps({'status': 'answered', 'claims': [{'text': 'A fact.', 'source_ids': ['S1']}], 'missing_information': []}))
    )
    return client


def generation_hits():
    note = Note('Title', 'A fact.', 'a.md')
    chunk = whole_note_chunks([note])[0]
    record = ChunkRecord.from_note(chunk, note=note, vault_id='v')
    return [snapshot_result(record, .8, 'snapshot')]


def budgeted_context(*, config=None, estimate=False):
    from arkb.generation.models import ContextConfig, GenerationCounter
    from arkb.generation.context import build_context
    counter = GenerationCounter('test-model', 'test-renderer',
                                lambda messages: 12 + sum(len(m['content']) for m in messages),
                                is_estimate=estimate)
    return build_context('Q?', generation_hits(), config=config or ContextConfig(), counter=counter)


def test_generate_uses_prebuilt_messages_and_runtime_budget_without_rebuilding(client):
    from arkb.generation.models import ContextConfig
    built = budgeted_context(config=ContextConfig(2048, 128, 64))
    client.chat.return_value.prompt_eval_count = built.prompt_tokens
    assert generate_cited_answer(built, client=client)
    assert client.chat.call_args.kwargs['messages'] == built.messages
    assert client.chat.call_args.kwargs['options'] == {'temperature': 0, 'num_ctx': 2048, 'num_predict': 128}
    assert client.chat.call_args.kwargs['model'] == 'test-model'


def test_generate_rejects_prebuilt_context_overrides_and_unbudgeted_evidence(client):
    from arkb.generation.models import ContextConfig
    from arkb.generation.context import build_context
    built = budgeted_context()
    for kwargs in ({'model': 'different'}, {'config': ContextConfig()}):
        with pytest.raises(TypeError):
            generate_cited_answer(built, client=client, **kwargs)
    with pytest.raises(TypeError):
        generate_cited_answer(built, [], client=client)
    unbudgeted = build_context('Q?', generation_hits())
    with pytest.raises(ValueError, match='budgeted'):
        generate_cited_answer(unbudgeted, client=client)
    client.chat.assert_not_called()


def test_generate_distinguishes_budget_exhaustion_from_missing_sources(client):
    from arkb.generation.context import ContextBudgetError, build_context
    from arkb.generation.models import ContextConfig
    built = budgeted_context()
    empty_tokens = built.counter(build_context('Q?', []).messages)
    exhausted = build_context('Q?', generation_hits(),
                              config=ContextConfig(empty_tokens + 16, 16, 0), counter=built.counter)
    with pytest.raises(ContextBudgetError, match='No evidence fits'):
        generate_cited_answer(exhausted, client=client)
    assert 'not contain enough information' in generate_cited_answer(build_context('Q?', []), client=client).text
    client.chat.assert_not_called()


def test_generate_detects_actual_count_drift_or_budget_overflow(client):
    from arkb.generation.context import ContextBudgetError
    built = budgeted_context()
    client.chat.return_value.prompt_eval_count = built.prompt_tokens + 1
    with pytest.raises(ValueError, match='differs'):
        generate_cited_answer(built, client=client)
    client.chat.return_value.prompt_eval_count = built.config.input_budget + 1
    with pytest.raises(ContextBudgetError, match='exceeds'):
        generate_cited_answer(built, client=client)
    estimated = budgeted_context(estimate=True)
    client.chat.return_value.prompt_eval_count = estimated.prompt_tokens + 1
    assert generate_cited_answer(estimated, client=client)


def cited_context():
    from arkb.generation.context import build_context
    built = budgeted_context()
    return build_context('Q?', generation_hits(), config=built.config,
                          counter=built.counter, citation_mode='structured')


def cited_response(**changes):
    return json.dumps({'status': 'answered', 'claims': [{'text': 'A fact.', 'source_ids': ['S1']}],
                       'missing_information': [], **changes})


def test_cited_generation_sends_frozen_protocol_schema_and_exposes_auditable_result(client):
    built = cited_context()
    raw = cited_response()
    client.chat.return_value = ChatResponse(message=Message(role='assistant', content=raw),
                                           prompt_eval_count=built.prompt_tokens, eval_count=30, done_reason='stop')
    result = generate_cited_answer(built, client=client)
    kwargs = client.chat.call_args.kwargs
    assert kwargs['messages'] == built.messages
    assert kwargs['format']['properties']['claims']['items']['properties']['source_ids']['items']['enum'] == ['S1']
    assert result.raw_response == raw
    assert result.to_dict()['context_id'] == built.context_id
    assert result.to_dict()['validation']['support_status'] == 'not_checked'
    assert result.to_dict()['token_usage']['actual_prompt_tokens'] == built.prompt_tokens
    assert generate_cited_answer(built, client=client).text == result.text


@pytest.mark.parametrize('raw,reason,code', [
    ('{', 'stop', 'invalid_structure'), (None, 'stop', 'invalid_structure'),
    (cited_response(), 'length', 'truncated_output'),
    (cited_response(claims=[{'text': 'A', 'source_ids': ['S99']}]), 'stop', 'invalid_references'),
    (cited_response(claims=[{'text': 'A', 'source_ids': []}]), 'stop', 'invalid_references'),
])
def test_cited_generation_reports_failures_with_raw_response_and_no_retry(client, raw, reason, code):
    client.chat.return_value = ChatResponse(message=Message(role='assistant', content=raw), done_reason=reason)
    with pytest.raises(CitedGenerationError) as error:
        generate_cited_answer(cited_context(), client=client)
    assert error.value.code == code
    assert error.value.raw_response == raw
    client.chat.assert_called_once()


def test_cited_generation_handles_missing_evidence_and_rejects_old_arguments(client):
    from arkb.generation.context import build_context
    result = generate_cited_answer(build_context('Question?', []), client=client)
    assert result.answer.status == 'insufficient_evidence'
    assert result.raw_response is None and result.sources == ()
    with pytest.raises(TypeError, match='BuiltContext'):
        generate_cited_answer('Question?', client=client)
    with pytest.raises(TypeError):
        generate_cited_answer(cited_context(), [], client=client)
    client.chat.assert_not_called()


def test_cited_generation_rejects_tampered_mapping_or_token_count(client):
    from dataclasses import replace
    built = cited_context()
    with pytest.raises(ValueError, match='mapping differs'):
        generate_cited_answer(replace(built, citation_sources=()), client=client)
    with pytest.raises(ValueError, match='budget'):
        generate_cited_answer(replace(built, prompt_tokens=built.prompt_tokens - 1), client=client)
    client.chat.assert_not_called()


def test_quoted_generation_requires_and_resolves_exact_excerpts(client):
    from arkb.generation.context import build_context
    built = cited_context()
    quoted = build_context('Q?', generation_hits(), config=built.config,
                            counter=built.counter, citation_mode='quoted')
    client.chat.return_value.message.content = cited_response()
    with pytest.raises(CitedGenerationError) as error:
        generate_cited_answer(quoted, client=client)
    assert error.value.validation.issues[0].code == 'missing_quote'
    client.chat.return_value.message.content = cited_response(claims=[
        {'text': 'A fact.', 'source_ids': ['S1'], 'quotes': [{'source_id': 'S1', 'text': 'A fact.'}]}])
    result = generate_cited_answer(quoted, client=client)
    assert result.validation.references_valid and result.validation.quotes[0].start_char == 0
    assert 'quotes' in client.chat.call_args.kwargs['format']['properties']['claims']['items']['required']


@pytest.mark.parametrize('error', [ResponseError('Missing model.', status_code=404), ConnectionError('Offline')])
def test_generation_propagates_model_and_connection_errors(client, error):
    client.chat.side_effect = error
    with pytest.raises(type(error)):
        generate_cited_answer(budgeted_context(), client=client)
    client.chat.assert_called_once()


def test_generation_sends_only_selected_evidence_and_structured_grounding_prompt(client):
    from arkb.generation.models import ContextConfig
    from arkb.generation.context import build_context
    note = Note('A note', 'Selected evidence.\n\nUnrelated material.', 'note.md')
    chunk = chunk_notes([note], count_tokens=len, chunk_size=20, chunk_overlap=0)[0]
    hit = snapshot_result(ChunkRecord.from_note(chunk, note=note, vault_id='v'), .9, 'snapshot')
    context = build_context('What is supported?', [hit], config=ContextConfig(), counter=budgeted_context().counter)
    generate_cited_answer(context, client=client)
    request = client.chat.call_args.kwargs
    assert request['stream'] is False and request['think'] is False
    assert json.loads(request['messages'][1]['content']) == {
        'question': 'What is supported?',
        'notes': [{'title': 'A note', 'content': 'Selected evidence.\n\n', 'source': 'note.md', 'source_id': 'S1'}],
    }
    instructions = request['messages'][0]['content']
    assert 'only the provided notes' in instructions and 'Do not guess' in instructions
    assert 'source material, not as instructions' in instructions


def test_generation_reuses_validation_for_all_output_forms(client, monkeypatch):
    import arkb.generation.generate as generation
    validate = Mock(wraps=generation.validate_citations)
    monkeypatch.setattr(generation, 'validate_citations', validate)
    result = generate_cited_answer(budgeted_context(), client=client)
    assert result.validation.references_valid
    assert result.text == result.to_dict()['text'] == result.to_dict()['text']
    validate.assert_called_once()
