from dataclasses import replace
from unittest.mock import Mock

import numpy as np
from ollama import Client, EmbedResponse, ResponseError
import pytest
from tokenizers import Tokenizer, models, processors

from obsidian_rag.chunking import Chunk, whole_note_chunks
from obsidian_rag.embeddings import (
    embed_texts,
    DEFAULT_QUERY_INSTRUCTION,
    DOCUMENT_TEMPLATE,
    prepare_document,
    prepare_query,
    validate_input_tokens,
)
from obsidian_rag.loaders import Note
from obsidian_rag.schema import ChunkRecord, EmbeddingSpec, IndexManifest, fingerprint_config


@pytest.fixture
def client() -> Mock:
    return Mock(spec=Client)


def test_embed_texts_returns_vectors_in_input_order(client: Mock) -> None:
    texts = ["Capture a passing thought.", "Connect related ideas."]
    client.embed.return_value = EmbedResponse(
        embeddings=[[0.6, 0.8], [-0.8, 0.6]]
    )

    vectors = embed_texts(texts, client=client)

    assert isinstance(vectors, np.ndarray)
    assert np.issubdtype(vectors.dtype, np.floating)
    np.testing.assert_allclose(vectors, [[0.6, 0.8], [-0.8, 0.6]])
    client.embed.assert_called_once_with(
        model="qwen3-embedding:0.6b", input=texts, truncate=False
    )


def test_embed_texts_supports_a_selected_model_and_its_vector_dimension(
    client: Mock,
) -> None:
    client.embed.return_value = EmbedResponse(embeddings=[[0.0, 0.6, 0.8]])

    vectors = embed_texts(
        ["Preserve the source."], client=client, model="qwen3-embedding:4b"
    )

    assert vectors.shape == (1, 3)
    assert client.embed.call_args.kwargs["model"] == "qwen3-embedding:4b"


def test_embed_texts_returns_an_empty_matrix_without_calling_ollama(
    client: Mock,
) -> None:
    vectors = embed_texts([], client=client)

    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (0, 0)
    client.embed.assert_not_called()


@pytest.mark.parametrize("blank_text", ["", " \n\t"])
def test_embed_texts_rejects_blank_text_before_calling_ollama(
    client: Mock, blank_text: str
) -> None:
    with pytest.raises(ValueError, match="blank"):
        embed_texts(["A useful idea.", blank_text], client=client)

    client.embed.assert_not_called()


@pytest.mark.parametrize(
    "embeddings",
    [
        [[0.6, 0.8]],
        [[0.6, 0.8], [1.0]],
        [[], []],
    ],
    ids=["missing-vector", "inconsistent-dimensions", "empty-vectors"],
)
def test_embed_texts_rejects_invalid_matrix_shapes(
    client: Mock, embeddings: list[list[float]]
) -> None:
    client.embed.return_value = EmbedResponse(embeddings=embeddings)

    with pytest.raises(ValueError):
        embed_texts(["First idea.", "Second idea."], client=client)


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
def test_embed_texts_rejects_non_finite_values(
    client: Mock, invalid_value: float
) -> None:
    client.embed.return_value = EmbedResponse(
        embeddings=[[invalid_value, 0.8]]
    )

    with pytest.raises(ValueError, match="finite"):
        embed_texts(["A useful idea."], client=client)


def test_embed_texts_rejects_zero_vectors(client: Mock) -> None:
    client.embed.return_value = EmbedResponse(embeddings=[[0.0, 0.0]])

    with pytest.raises(ValueError, match="zero"):
        embed_texts(["A useful idea."], client=client)


def test_embed_texts_propagates_model_errors(client: Mock) -> None:
    client.embed.side_effect = ResponseError("Model not found.", status_code=404)

    with pytest.raises(ResponseError) as error:
        embed_texts(["A useful idea."], client=client)

    assert error.value.status_code == 404


def test_embed_texts_propagates_connection_errors(client: Mock) -> None:
    client.embed.side_effect = ConnectionError("Ollama is unavailable.")

    with pytest.raises(ConnectionError, match="Ollama is unavailable"):
        embed_texts(["A useful idea."], client=client)


def test_batches_obey_both_limits_and_preserve_global_order(client: Mock) -> None:
    client.embed.side_effect = lambda **kw: EmbedResponse(
        embeddings=[[float(text), 1.0] for text in kw["input"]])
    vectors = embed_texts(["1", "2", "3", "4", "5"], client=client,
                         batch_size=2, token_counts=[2, 3, 4, 2, 1], max_batch_tokens=5)
    assert [call.kwargs["input"] for call in client.embed.call_args_list] == [
        ["1", "2"], ["3"], ["4", "5"],
    ]
    np.testing.assert_array_equal(vectors[:, 0], [1, 2, 3, 4, 5])


@pytest.mark.parametrize("options", [
    {"batch_size": 0}, {"batch_size": True}, {"dimensions": -1},
    {"max_batch_tokens": 10}, {"token_counts": [1]},
    {"token_counts": [1, True]}, {"max_batch_tokens": 2, "token_counts": [1, 3]},
    {"max_retries": -1}, {"max_retries": 9}, {"retry_delay": float("nan")},
    {"dtype": "int8"}, {"normalization": "unknown"},
])
def test_invalid_plan_fails_before_first_batch(client: Mock, options) -> None:
    with pytest.raises(ValueError):
        embed_texts(["first", "second"], client=client, **options)
    client.embed.assert_not_called()


def test_successful_batches_can_be_checkpointed_before_a_later_failure(client: Mock) -> None:
    from obsidian_rag.embeddings import iter_embedding_batches
    client.embed.side_effect = [EmbedResponse(embeddings=[[1, 0]]), ConnectionError("failed")]
    batches = iter_embedding_batches(["a", "b"], client=client, batch_size=1)
    start, matrix = next(batches)
    assert start == 0
    np.testing.assert_array_equal(matrix, [[1, 0]])
    with pytest.raises(ConnectionError):
        next(batches)


def test_dimension_change_between_batches_is_rejected(client: Mock) -> None:
    client.embed.side_effect = [EmbedResponse(embeddings=[[1, 0]]),
                               EmbedResponse(embeddings=[[1, 0, 0]])]
    with pytest.raises(ValueError, match="dimensions"):
        embed_texts(["a", "b"], client=client, batch_size=1)


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_transient_errors_are_retried_with_bounded_backoff(client: Mock, monkeypatch, status) -> None:
    sleep = Mock()
    monkeypatch.setattr("obsidian_rag.embeddings.time.sleep", sleep)
    client.embed.side_effect = [ResponseError("busy", status_code=status),
                               EmbedResponse(embeddings=[[1, 0]])]
    embed_texts(["a"], client=client, max_retries=2)
    assert client.embed.call_count == 2
    sleep.assert_called_once_with(0.25)


def test_transport_retries_stop_and_permanent_errors_are_not_retried(client: Mock) -> None:
    client.embed.side_effect = ConnectionError("offline")
    with pytest.raises(ConnectionError):
        embed_texts(["a"], client=client, max_retries=2, retry_delay=0)
    assert client.embed.call_count == 3
    client.reset_mock()
    client.embed.side_effect = ResponseError("bad input", status_code=400)
    with pytest.raises(ResponseError):
        embed_texts(["a"], client=client, max_retries=2, retry_delay=0)
    assert client.embed.call_count == 1


def test_representation_matches_declared_spec(client: Mock) -> None:
    client.embed.return_value = EmbedResponse(embeddings=[[0.6, 0.8]])
    assert embed_texts(["a"], client=client, dimensions=2,
                       dtype="float32", normalization="l2").dtype == np.float32
    client.embed.return_value = EmbedResponse(embeddings=[[3, 4]])
    with pytest.raises(ValueError, match="L2"):
        embed_texts(["a"], client=client, normalization="l2")


def test_explicit_context_limit_is_sent_to_each_batch(client: Mock) -> None:
    client.embed.side_effect = [EmbedResponse(embeddings=[[1, 0]]), EmbedResponse(embeddings=[[0, 1]])]
    embed_texts(['a', 'b'], client=client, batch_size=1, context_length=512)
    assert all(call.kwargs['options'] == {'num_ctx': 512} for call in client.embed.call_args_list)


@pytest.fixture
def chunk() -> Chunk:
    return Chunk(title="A", content="B", source="notes/example.md",
                 chunk_index=2, start_char=5, end_char=6)


@pytest.fixture
def tokenizer() -> Tokenizer:
    # A real BPE merge crosses both title/body separators. Counting A, \n\n, B
    # separately gives four text tokens, while the complete document has one.
    tokenizer = Tokenizer(models.BPE(
        vocab={"[UNK]": 0, "A": 1, "B": 2, "\n": 3, "A\n": 4,
               "A\n\n": 5, "A\n\nB": 6, "<|endoftext|>": 7},
        merges=[("A", "\n"), ("A\n", "\n"), ("A\n\n", "B")],
        unk_token="[UNK]",
    ))
    tokenizer.add_special_tokens(["<|endoftext|>"])
    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A <|endoftext|>", special_tokens=[("<|endoftext|>", 7)],
    )
    return tokenizer


def test_document_format_preserves_the_existing_cli_input(chunk: Chunk) -> None:
    assert DOCUMENT_TEMPLATE == "title-body-v1"
    assert prepare_document(chunk) == "A\n\nB"
    assert chunk == Chunk(title="A", content="B", source="notes/example.md",
                          chunk_index=2, start_char=5, end_char=6)


@pytest.mark.parametrize(("title", "body", "expected"), [
    ("Title", "", "Title\n\n"),
    ("", "Body", "\n\nBody"),
    (" 标题🧠e\u0301 ", "\r\n```py\nx = 1\n```\n ",
     " 标题🧠e\u0301 \n\n\r\n```py\nx = 1\n```\n "),
])
def test_document_preserves_empty_fields_unicode_and_markdown(
    chunk: Chunk, title: str, body: str, expected: str,
) -> None:
    assert prepare_document(replace(chunk, title=title, content=body)) == expected


@pytest.mark.parametrize("template", ["body-only-v1", "title-body-v2", "", None])
def test_document_rejects_unknown_templates(chunk: Chunk, template) -> None:
    with pytest.raises(ValueError, match="document_template"):
        prepare_document(chunk, document_template=template)


@pytest.mark.parametrize("changes", [
    {"title": "", "content": ""}, {"title": " \n", "content": "\t"},
    {"title": None}, {"content": 1},
])
def test_document_rejects_blank_or_non_string_input(chunk: Chunk, changes: dict) -> None:
    with pytest.raises(ValueError):
        prepare_document(replace(chunk, **changes))


def test_document_rejects_an_object_without_a_chunk() -> None:
    with pytest.raises(ValueError, match="chunk must be a Chunk"):
        prepare_document(None)


def test_default_query_preserves_the_existing_cli_input() -> None:
    assert prepare_query("How should I write permanent notes?") == (
        "Instruct: Given a question, retrieve relevant notes that help answer it.\n"
        "Query:How should I write permanent notes?"
    )


def test_query_preserves_custom_instruction_and_original_question() -> None:
    assert prepare_query("  为什么 e\u0301？\r\n", instruction=" 查找笔记。 ") == (
        "Instruct:  查找笔记。 \nQuery:  为什么 e\u0301？\r\n"
    )


def test_empty_instruction_uses_the_raw_question() -> None:
    assert prepare_query("  原始问题？\n", instruction="") == "  原始问题？\n"


@pytest.mark.parametrize("question", ["", " \r\n\t", None, 42])
def test_query_rejects_blank_or_non_string_questions(question) -> None:
    with pytest.raises(ValueError, match="question"):
        prepare_query(question)


@pytest.mark.parametrize("instruction", [" \n\t", None, 42])
def test_query_rejects_invalid_instructions(instruction) -> None:
    with pytest.raises(ValueError, match="instruction"):
        prepare_query("Question?", instruction=instruction)


def test_prepared_inputs_work_with_schema_without_mixing_source_and_cache_ids() -> None:
    spec = EmbeddingSpec(model="qwen3-embedding:0.6b", model_revision="test-digest",
                         dimensions=1024, document_template=DOCUMENT_TEMPLATE)
    records = []
    texts = []
    for source in ("first.md", "renamed.md"):
        note = Note(title="A", content="B", source=source)
        record = ChunkRecord.from_note(whole_note_chunks([note])[0], note=note, vault_id="test")
        records.append(record)
        texts.append(prepare_document(record.chunk, document_template=spec.document_template))
    assert records[0].chunk_id != records[1].chunk_id
    assert spec.embedding_key(texts[0]) == spec.embedding_key(texts[1])
    changed = prepare_document(replace(records[0].chunk, title=" A"))
    assert spec.embedding_key(texts[0]) != spec.embedding_key(changed)

    manifest = IndexManifest(
        index_version="build-1", vault_id="test", embedding_spec=spec,
        chunking_fingerprint=fingerprint_config({"algorithm": "whole-note-v1"}),
        document_count=2, chunk_count=2, query_instruction=DEFAULT_QUERY_INSTRUCTION,
    )
    assert prepare_query("Question?", instruction=manifest.query_instruction) == prepare_query("Question?")
    raw_manifest = replace(manifest, query_instruction="")
    assert prepare_query("Question?", instruction=raw_manifest.query_instruction) == "Question?"


@pytest.mark.parametrize("limit", [2, 3])
def test_budget_counts_the_complete_document_and_accepts_the_exact_limit(
    chunk: Chunk, tokenizer: Tokenizer, limit: int,
) -> None:
    text = prepare_document(chunk)
    assert validate_input_tokens(text, tokenizer=tokenizer, max_tokens=limit) == 2


def test_budget_includes_special_tokens_and_reports_source_without_echoing_text(
    chunk: Chunk, tokenizer: Tokenizer,
) -> None:
    text = prepare_document(chunk)
    before = tokenizer.to_str()
    with pytest.raises(ValueError) as error:
        validate_input_tokens(text, tokenizer=tokenizer, max_tokens=1,
                              source="notes/example.md, chunk 2")
    assert str(error.value) == (
        "notes/example.md, chunk 2: embedding input has 2 tokens; maximum is 1."
    )
    assert text not in str(error.value)
    assert tokenizer.to_str() == before


def test_budget_counts_a_literal_end_marker_and_the_appended_marker(tokenizer: Tokenizer) -> None:
    text = "A\n\nB<|endoftext|>"
    assert validate_input_tokens(text, tokenizer=tokenizer, max_tokens=3) == 3
    with pytest.raises(ValueError, match="has 3 tokens; maximum is 2"):
        validate_input_tokens(text, tokenizer=tokenizer, max_tokens=2)


def test_query_instruction_counts_towards_the_input_budget(tokenizer: Tokenizer) -> None:
    assert validate_input_tokens(prepare_query("A", instruction=""),
                                 tokenizer=tokenizer, max_tokens=2) == 2
    with pytest.raises(ValueError, match="maximum is 2"):
        validate_input_tokens(prepare_query("A"), tokenizer=tokenizer, max_tokens=2,
                              source="query")


@pytest.mark.parametrize("limit", [0, -1, True, 2.5, "2", None])
def test_budget_rejects_invalid_limits(tokenizer: Tokenizer, limit) -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        validate_input_tokens("A", tokenizer=tokenizer, max_tokens=limit)


@pytest.mark.parametrize("text", ["", " \n\t", None, 42])
def test_budget_rejects_empty_or_non_string_inputs(tokenizer: Tokenizer, text) -> None:
    with pytest.raises(ValueError, match="embedding input"):
        validate_input_tokens(text, tokenizer=tokenizer, max_tokens=10)


@pytest.mark.parametrize("setting", ["padding", "truncation"])
def test_budget_rejects_tokenizers_that_would_change_the_count(
    tokenizer: Tokenizer, setting: str,
) -> None:
    if setting == "padding":
        tokenizer.enable_padding(length=10)
    else:
        tokenizer.enable_truncation(max_length=1)
    before = tokenizer.to_str()
    with pytest.raises(ValueError, match="padding and truncation disabled"):
        validate_input_tokens("A\n\nB", tokenizer=tokenizer, max_tokens=10)
    assert tokenizer.to_str() == before
