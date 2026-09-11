"""Opt-in prepared-input checks against the cached Qwen tokenizer and Ollama."""

import os

from ollama import Client
import pytest

from arkb.knowledge.chunking import whole_note_chunks
from arkb.knowledge.embeddings import prepare_document, prepare_query, validate_input_tokens
from arkb.knowledge.models import Note
from arkb.knowledge.embeddings import load_tokenizer


pytestmark = pytest.mark.skipif(
    os.environ.get("ARKB_RUN_MODEL_TESTS") != "1",
    reason="Set ARKB_RUN_MODEL_TESTS=1 with Qwen cached and Ollama running.",
)
pytestmark = [pytest.mark.integration, pytestmark]


@pytest.mark.parametrize(("kind", "body"), [
    ("document", "Body cafe\u0301 🧠\r\n```py\nx = 1\n```"),
    ("document", "Literal marker.<|endoftext|>"),
    ("query", "How to preserve e\u0301 and Markdown?"),
    ("raw-query", "  Original question?\r\n"),
])
def test_prepared_input_count_matches_the_embedding_request(kind: str, body: str) -> None:
    tokenizer = load_tokenizer(local_files_only=True)
    if kind == "document":
        note = Note(title="Permanent notes", content=body, source="example.md")
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
