"""Opt-in chunking checks with the cached Qwen tokenizer; no Ollama required."""

from functools import partial
import os

import pytest
from tokenizers import Tokenizer

from obsidian_rag.knowledge_base.chunking import chunk_notes
from obsidian_rag.knowledge_base.loaders import Note
from obsidian_rag.knowledge_base.tokenization import count_tokens, load_tokenizer


pytestmark = pytest.mark.skipif(
    os.environ.get("OBSIDIAN_RAG_RUN_MODEL_TESTS") != "1",
    reason="Set OBSIDIAN_RAG_RUN_MODEL_TESTS=1 with the pinned Qwen tokenizer cached.",
)


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return load_tokenizer(local_files_only=True)


@pytest.mark.parametrize(
    "text",
    [
        "卡片盒笔记需要保留来源，长文档按段落切分。\n\n" * 180,
        "Permanent notes explain one idea and link related concepts.\n\n" * 200,
        "## Retrieval\n\n先检索相关笔记，再生成答案。 Use chunk_size=512.\n\n" * 100,
        "```python\r\ndef retrieve(query):\r\n    return notes[query]\r\n```\r\n\r\n" * 100,
        "cafe\u0301👩🏽\u200d💻中文" * 180,
        " a" * 1050,
    ],
    ids=["chinese", "english", "mixed-markdown", "code-and-crlf", "unicode", "boundary"],
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
