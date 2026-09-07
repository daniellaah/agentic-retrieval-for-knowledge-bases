import json
from unittest.mock import Mock

from ollama import ChatResponse, Client, Message, ResponseError
import pytest

from obsidian_rag.chunking import chunk_notes, whole_note_chunks
from obsidian_rag.generation import generate_answer
from obsidian_rag.loaders import Note
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
