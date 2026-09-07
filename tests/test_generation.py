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
