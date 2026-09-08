"""Opt-in checks against the cached Qwen snapshot and local Ollama model."""

import os

from ollama import Client
import pytest
from tokenizers import Tokenizer

from obsidian_rag.tokenization import count_tokens, load_tokenizer


pytestmark = pytest.mark.skipif(
    os.environ.get("OBSIDIAN_RAG_RUN_MODEL_TESTS") != "1",
    reason="Set OBSIDIAN_RAG_RUN_MODEL_TESTS=1 with Qwen cached and Ollama running.",
)


@pytest.fixture(scope="module")
def tokenizer() -> Tokenizer:
    return load_tokenizer(local_files_only=True)


@pytest.fixture(scope="module")
def client() -> Client:
    return Client(host="http://127.0.0.1:11434", timeout=60, trust_env=False)


@pytest.mark.parametrize(
    ("text", "expected_text_tokens", "expected_input_tokens"),
    [
        pytest.param("Chunking keeps related ideas together.", 7, 8, id="english"),
        pytest.param(
            "卡片盒笔记包含临时笔记、文献笔记和永久笔记。", 13, 14, id="chinese"
        ),
        pytest.param(
            "Obsidian 使用 Markdown 保存 Zettelkasten 笔记，chunk_size=512。",
            22, 23, id="mixed",
        ),
        pytest.param(
            "# Chunking\n\n## 原则\n\n- 保留来源\n- 使用 64 tokens overlap\n",
            23, 24, id="markdown",
        ),
        pytest.param(
            "```python\ndef count_tokens(text: str) -> int:\n"
            "    return len(tokenizer.encode(text).ids)\n```",
            24, 25, id="code",
        ),
        pytest.param("学习笔记 🧠📚，程序员 👩🏽\u200d💻。", 15, 16, id="emoji"),
        pytest.param("café naïve résumé", 7, 8, id="composed-characters"),
        pytest.param(
            "cafe\u0301 nai\u0308ve re\u0301sume\u0301",
            12, 13, id="combining-characters",
        ),
        pytest.param(
            "卡片盒笔记需要保留来源，长文档按段落切分。\n\n" * 70,
            1050, 1051, id="long-chinese",
        ),
        pytest.param(" a" * 511, 511, 512, id="511-tokens"),
        pytest.param(" a" * 512, 512, 513, id="512-tokens"),
        pytest.param(" a" * 513, 513, 514, id="513-tokens"),
        pytest.param("A note.<|endoftext|>", 4, 5, id="literal-end-marker"),
        pytest.param(
            "卡片盒笔记\n\n" + " a" * 512,
            516, 517, id="title-and-body",
        ),
    ],
)
def test_counts_match_qwen_and_ollama(
    tokenizer: Tokenizer,
    client: Client,
    text: str,
    expected_text_tokens: int,
    expected_input_tokens: int,
) -> None:
    # These literal counts were independently observed with Ollama 0.33.2.
    assert count_tokens(text, tokenizer=tokenizer) == expected_text_tokens
    assert count_tokens(
        text, tokenizer=tokenizer, add_special_tokens=True
    ) == expected_input_tokens

    response = client.embed(model="qwen3-embedding:0.6b", input=text, truncate=False)

    assert response.prompt_eval_count == expected_input_tokens
