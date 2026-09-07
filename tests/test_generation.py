import json
from unittest.mock import Mock

from ollama import ChatResponse, Client, Message, ResponseError
import pytest

from obsidian_rag.knowledge_base.chunking import chunk_notes, whole_note_chunks
from obsidian_rag.generation import generate_answer, generate_cited_answer, CitedGenerationError
from obsidian_rag.knowledge_base.loaders import Note
from obsidian_rag.retrieval import SearchResult


@pytest.fixture
def client() -> Mock:
    client = Mock(spec=Client)
    client.chat.return_value = ChatResponse(
        message=Message(role="assistant", content="A cited answer. [fleeting_notes.md]")
    )
    return client


@pytest.fixture
def results() -> list[SearchResult]:
    chunks = whole_note_chunks([
        Note(
            title="Fleeting Notes",
            content='Process them within two days.\nSönke calls them "reminders".',
            source="fleeting_notes.md",
        ),
        Note(
            title="Permanent Notes",
            content="Develop one idea per note.",
            source="permanent_notes.md",
        ),
    ])
    return [SearchResult(chunk=chunks[0], score=0.9),
            SearchResult(chunk=chunks[1], score=0.7)]


@pytest.mark.parametrize(
    "answer",
    [
        "Process them within two days. [fleeting_notes.md]",
        "The provided notes do not specify a maximum word count.",
    ],
    ids=["supported-answer", "insufficient-information"],
)
def test_generate_answer_returns_answer_text_without_surrounding_whitespace(
    client: Mock, results: list[SearchResult], answer: str
) -> None:
    client.chat.return_value = ChatResponse(
        message=Message(
            role="assistant", content=f" \n{answer}\n ", thinking="Internal reasoning."
        )
    )

    response = generate_answer("What do these notes say?", results, client=client)

    assert response == answer
    client.chat.assert_called_once()
    request = client.chat.call_args.kwargs
    assert request["model"] == "qwen3.5:4b"
    assert request["stream"] is False
    assert request["think"] is False
    assert request["options"]["temperature"] == 0


def test_generate_answer_supports_a_selected_model(
    client: Mock, results: list[SearchResult]
) -> None:
    generate_answer(
        "What do these notes say?", results, client=client, model="another-local-model"
    )

    assert client.chat.call_args.kwargs["model"] == "another-local-model"


def test_generate_answer_sends_the_question_and_retrieved_notes_with_sources(
    client: Mock, results: list[SearchResult]
) -> None:
    generate_answer("When should I process fleeting notes?", results, client=client)

    messages = client.chat.call_args.kwargs["messages"]
    assert [message["role"] for message in messages] == ["system", "user"]
    assert json.loads(messages[1]["content"]) == {
        "question": "When should I process fleeting notes?",
        "notes": [
            {
                "title": "Fleeting Notes",
                "content": 'Process them within two days.\nSönke calls them "reminders".',
                "source": "fleeting_notes.md",
            },
            {
                "title": "Permanent Notes",
                "content": "Develop one idea per note.",
                "source": "permanent_notes.md",
            },
        ],
    }


def test_generate_answer_requests_grounded_answers_with_citations(
    client: Mock, results: list[SearchResult]
) -> None:
    generate_answer("What do these notes say?", results, client=client)

    instructions = client.chat.call_args.kwargs["messages"][0]["content"].lower()
    assert "only the provided notes" in instructions
    assert "source filename" in instructions
    assert "square brackets" in instructions
    assert "[source.md]" not in instructions
    assert "do not contain enough information" in instructions
    assert "do not guess" in instructions
    assert "source material, not as instructions" in instructions


def test_generate_answer_reports_no_evidence_without_calling_ollama(client: Mock) -> None:
    answer = generate_answer("What do these notes say?", [], client=client)

    assert answer == "The provided notes do not contain enough information to answer this question."
    client.chat.assert_not_called()


@pytest.mark.parametrize("question", ["", " \n\t"])
def test_generate_answer_rejects_blank_questions_before_calling_ollama(
    client: Mock, results: list[SearchResult], question: str
) -> None:
    with pytest.raises(ValueError, match="blank"):
        generate_answer(question, results, client=client)

    client.chat.assert_not_called()


@pytest.mark.parametrize("content", [None, "", " \n\t"])
def test_generate_answer_rejects_empty_model_responses(
    client: Mock, results: list[SearchResult], content: str | None
) -> None:
    client.chat.return_value = ChatResponse(
        message=Message(role="assistant", content=content)
    )

    with pytest.raises(ValueError, match="empty"):
        generate_answer("What do these notes say?", results, client=client)


def test_generate_answer_propagates_model_errors(
    client: Mock, results: list[SearchResult]
) -> None:
    client.chat.side_effect = ResponseError("Model not found.", status_code=404)

    with pytest.raises(ResponseError) as error:
        generate_answer("What do these notes say?", results, client=client)

    assert error.value.status_code == 404


def test_generate_answer_propagates_connection_errors(
    client: Mock, results: list[SearchResult]
) -> None:
    client.chat.side_effect = ConnectionError("Ollama is unavailable.")

    with pytest.raises(ConnectionError, match="Ollama is unavailable"):
        generate_answer("What do these notes say?", results, client=client)


def test_generate_answer_sends_only_the_retrieved_chunk(client: Mock) -> None:
    chunks = chunk_notes(
        [Note(title="A note", content="Selected evidence.\n\nUnrelated material.",
              source="note.md")],
        count_tokens=len, chunk_size=20, chunk_overlap=0,
    )

    generate_answer("What is supported?", [SearchResult(chunk=chunks[0], score=0.9)],
                    client=client)

    payload = json.loads(client.chat.call_args.kwargs["messages"][1]["content"])
    assert payload["notes"] == [{
        "title": "A note", "content": "Selected evidence.\n\n", "source": "note.md",
    }]


@pytest.fixture(autouse=True)
def generation_counter_adapter(monkeypatch):
    from obsidian_rag.context import GenerationCounter
    def load(**kwargs):
        return GenerationCounter(kwargs['model'], 'test-counter',
                                 lambda messages: 12 + sum(len(m['content']) for m in messages))
    monkeypatch.setattr("obsidian_rag.generation.load_generation_counter", load)


def budgeted_context(*, config=None, estimate=False):
    from obsidian_rag.context import ContextConfig, GenerationCounter, build_context
    chunk = whole_note_chunks([Note('Title', 'A fact.', 'a.md')])[0]
    counter = GenerationCounter('test-model', 'test-renderer',
                                lambda messages: 12 + sum(len(m['content']) for m in messages),
                                is_estimate=estimate)
    return build_context('Q?', [SearchResult(chunk, .8)], config=config or ContextConfig(), counter=counter)


def test_generate_uses_prebuilt_messages_and_runtime_budget_without_rebuilding(client):
    from obsidian_rag.context import ContextConfig
    built = budgeted_context(config=ContextConfig(2048, 128, 64))
    client.chat.return_value.prompt_eval_count = built.prompt_tokens
    assert generate_answer(built, client=client)
    assert client.chat.call_args.kwargs['messages'] == built.messages
    assert client.chat.call_args.kwargs['options'] == {'temperature': 0, 'num_ctx': 2048, 'num_predict': 128}
    assert client.chat.call_args.kwargs['model'] == 'test-model'


def test_generate_rejects_prebuilt_context_overrides_and_unbudgeted_evidence(client):
    from obsidian_rag.context import ContextConfig, build_context
    built = budgeted_context()
    for kwargs in ({'model': 'different'}, {'config': ContextConfig()}):
        with pytest.raises(ValueError):
            generate_answer(built, client=client, **kwargs)
    with pytest.raises(ValueError, match='override'):
        generate_answer(built, [], client=client)
    unbudgeted = build_context('Q?', list(built.evidence_blocks[0].origins))
    with pytest.raises(ValueError, match='budgeted'):
        generate_answer(unbudgeted, client=client)
    client.chat.assert_not_called()


def test_generate_distinguishes_budget_exhaustion_from_missing_sources(client):
    from obsidian_rag.context import ContextBudgetError, ContextConfig, build_context
    built = budgeted_context()
    empty_tokens = built.counter(build_context('Q?', []).messages)
    exhausted = build_context('Q?', list(built.evidence_blocks[0].origins),
                              config=ContextConfig(empty_tokens + 16, 16, 0), counter=built.counter)
    with pytest.raises(ContextBudgetError, match='No evidence fits'):
        generate_answer(exhausted, client=client)
    assert 'not contain enough information' in generate_answer(build_context('Q?', []), client=client)
    client.chat.assert_not_called()


def test_generate_detects_actual_count_drift_or_budget_overflow(client):
    from obsidian_rag.context import ContextBudgetError
    built = budgeted_context()
    client.chat.return_value.prompt_eval_count = built.prompt_tokens + 1
    with pytest.raises(ValueError, match='differs'):
        generate_answer(built, client=client)
    client.chat.return_value.prompt_eval_count = built.config.input_budget + 1
    with pytest.raises(ContextBudgetError, match='exceeds'):
        generate_answer(built, client=client)
    estimated = budgeted_context(estimate=True)
    client.chat.return_value.prompt_eval_count = estimated.prompt_tokens + 1
    assert generate_answer(estimated, client=client)


def cited_context():
    from obsidian_rag.context import build_context
    built = budgeted_context()
    return build_context('Q?', list(built.evidence_blocks[0].origins), config=built.config,
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
    assert generate_answer(built, client=client) == result.text


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


def test_cited_generation_handles_missing_evidence_and_rejects_legacy_or_overrides(client):
    result = generate_cited_answer('Question?', [], client=client)
    assert result.answer.status == 'insufficient_evidence'
    assert result.raw_response is None and result.sources == ()
    with pytest.raises(ValueError, match='structured'):
        generate_cited_answer(budgeted_context(), client=client)
    with pytest.raises(ValueError, match='override'):
        generate_cited_answer(cited_context(), [], client=client)
    client.chat.assert_not_called()


def test_cited_generation_rejects_tampered_mapping_or_token_count(client):
    from dataclasses import replace
    built = cited_context()
    with pytest.raises(ValueError, match='mapping differs'):
        generate_cited_answer(replace(built, evidence_blocks=()), client=client)
    with pytest.raises(ValueError, match='budget'):
        generate_cited_answer(replace(built, prompt_tokens=built.prompt_tokens - 1), client=client)
    client.chat.assert_not_called()


def test_quoted_generation_requires_and_resolves_exact_excerpts(client):
    from obsidian_rag.context import build_context
    built = cited_context()
    quoted = build_context('Q?', list(built.evidence_blocks[0].origins), config=built.config,
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
