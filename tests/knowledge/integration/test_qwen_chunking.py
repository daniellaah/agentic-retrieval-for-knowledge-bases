"""Opt-in chunking checks with the cached Qwen tokenizer; no Ollama required."""

from functools import partial
import os

import pytest
from tokenizers import Tokenizer

from arkb.knowledge.chunking import chunk_notes
from arkb.knowledge.models import Note
from arkb.knowledge.embeddings import count_tokens, load_tokenizer


pytestmark = pytest.mark.skipif(
    os.environ.get("ARKB_RUN_MODEL_TESTS") != "1",
    reason="Set ARKB_RUN_MODEL_TESTS=1 with the pinned Qwen tokenizer cached.",
)
pytestmark = [pytest.mark.integration, pytestmark]


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return load_tokenizer(local_files_only=True)


@pytest.mark.parametrize(
    "text",
    [
        "Permanent notes explain one idea and link related concepts.\n\n" * 200,
        "## Retrieval\n\nRetrieve related notes before generating an answer. Use chunk_size=512.\n\n" * 100,
        "```python\r\ndef retrieve(query):\r\n    return notes[query]\r\n```\r\n\r\n" * 100,
        "cafe\u0301👩🏽\u200d💻éø" * 180,
        " a" * 1050,
    ],
    ids=["english", "markdown", "code-and-crlf", "unicode", "boundary"],
)
def test_qwen_chunks_obey_token_budgets_and_preserve_the_original(
    tokenizer: Tokenizer, text: str
) -> None:
    count = partial(count_tokens, tokenizer=tokenizer)
    note = Note(title="A long note", content=text, source="long.md")

    chunks = chunk_notes([note], count_tokens=count)

    assert len(chunks) > 1
    reconstructed = ""
    covered_end = 0
    for index, chunk in enumerate(chunks):
        assert (chunk.title, chunk.source, chunk.chunk_index) == (
            "A long note", "long.md", index,
        )
        assert chunk.content == text[chunk.start_char:chunk.end_char]
        assert count(chunk.content) <= 512
        assert chunk.start_char <= covered_end < chunk.end_char
        assert count(text[chunk.start_char:covered_end]) <= 64
        reconstructed += chunk.content[covered_end - chunk.start_char:]
        covered_end = chunk.end_char
    assert reconstructed == text
