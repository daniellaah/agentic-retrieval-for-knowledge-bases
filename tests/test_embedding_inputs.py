from dataclasses import replace

import pytest
from tokenizers import Tokenizer, models, processors

from obsidian_rag.chunking import Chunk, whole_note_chunks
from obsidian_rag.embedding_inputs import (
    DEFAULT_QUERY_INSTRUCTION,
    DOCUMENT_TEMPLATE,
    prepare_document,
    prepare_query,
    validate_input_tokens,
)
from obsidian_rag.index_schema import ChunkRecord, EmbeddingSpec, IndexManifest, fingerprint_config
from obsidian_rag.notes import Note


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


def test_prepared_inputs_work_with_index_schema_without_mixing_source_and_cache_ids() -> None:
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
