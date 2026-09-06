"""Opt-in prepared-input checks against the cached Qwen tokenizer and Ollama."""

import os

import pytest
from ollama import Client

from obsidian_rag.chunking import whole_note_chunks
from obsidian_rag.embedding_inputs import prepare_document, prepare_query, validate_input_tokens
from obsidian_rag.notes import Note
from obsidian_rag.tokenization import load_tokenizer


pytestmark = pytest.mark.skipif(
    os.environ.get("OBSIDIAN_RAG_RUN_MODEL_TESTS") != "1",
    reason="Set OBSIDIAN_RAG_RUN_MODEL_TESTS=1 with Qwen cached and Ollama running.",
)


@pytest.mark.parametrize(("kind", "body"), [
    ("document", "正文 cafe\u0301 🧠\r\n```py\nx = 1\n```"),
    ("document", "Literal marker.<|endoftext|>"),
    ("query", "如何保留 e\u0301 和 Markdown？"),
    ("raw-query", "  原始问题？\r\n"),
])
def test_prepared_input_count_matches_the_embedding_request(kind: str, body: str) -> None:
    tokenizer = load_tokenizer(local_files_only=True)
    if kind == "document":
        note = Note(title="卡片盒笔记", content=body, source="example.md")
        text = prepare_document(whole_note_chunks([note])[0])
    elif kind == "query":
        text = prepare_query(body)
    else:
        text = prepare_query(body, instruction="")

    actual_count = validate_input_tokens(text, tokenizer=tokenizer, max_tokens=512)
    with Client(host="http://127.0.0.1:11434", timeout=60, trust_env=False) as client:
        response = client.embed(model="qwen3-embedding:0.6b", input=text,
                                truncate=False, options={"num_ctx": 512})

    assert actual_count == response.prompt_eval_count
    assert validate_input_tokens(text, tokenizer=tokenizer, max_tokens=actual_count) == actual_count
    with pytest.raises(ValueError, match="maximum is"):
        validate_input_tokens(text, tokenizer=tokenizer, max_tokens=actual_count - 1)
