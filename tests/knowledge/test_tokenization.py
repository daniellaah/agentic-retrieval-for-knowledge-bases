from pathlib import Path
from unittest.mock import Mock

import httpx
from huggingface_hub.errors import LocalEntryNotFoundError
import pytest
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors

from arkb.knowledge.embeddings import count_tokens, load_tokenizer


@pytest.fixture
def tokenizer_cache(tmp_path: Path) -> Path:
    """A tiny character tokenizer in a real Hub cache, with Qwen-like defaults."""
    backend = Tokenizer(
        models.WordLevel(
            {
                "[UNK]": 0, "a": 1, "b": 2, "é": 3, "e": 4,
                "\u0301": 5, "中": 6, "文": 7, "<|endoftext|>": 8,
            },
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    backend.normalizer = normalizers.NFC()
    backend.add_special_tokens(["<|endoftext|>"])
    backend.post_processor = processors.TemplateProcessing(
        single="$A <|endoftext|>",
        special_tokens=[("<|endoftext|>", 8)],
    )
    snapshot = (
        tmp_path / "models--Qwen--Qwen3-Embedding-0.6B" / "snapshots"
        / "c54f2e6e80b2d7b7de06f51cec4959f6b3e03418"
    )
    snapshot.mkdir(parents=True)
    backend.save(str(snapshot / "tokenizer.json"))
    return tmp_path


def test_load_tokenizer_reads_the_pinned_snapshot_offline(tokenizer_cache: Path) -> None:
    tokenizer = load_tokenizer(cache_dir=tokenizer_cache, local_files_only=True)

    assert tokenizer.encode("ab", add_special_tokens=False).ids == [1, 2]


@pytest.mark.parametrize(
    ("text", "expected"), [("", 0), ("ab", 2), ("中文", 2), ("a" * 512, 512)],
    ids=["empty", "text", "chinese", "512-tokens"],
)
def test_count_tokens_measures_text_without_end_markers(
    tokenizer_cache: Path, text: str, expected: int
) -> None:
    tokenizer = load_tokenizer(cache_dir=tokenizer_cache, local_files_only=True)

    assert count_tokens(text, tokenizer=tokenizer) == expected


def test_count_tokens_preserves_combining_characters_as_ollama_does(
    tokenizer_cache: Path,
) -> None:
    tokenizer = load_tokenizer(cache_dir=tokenizer_cache, local_files_only=True)

    assert count_tokens("e\u0301", tokenizer=tokenizer) == 2
    assert count_tokens("é", tokenizer=tokenizer) == 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [("ab", 3), ("a" * 512, 513), ("ab<|endoftext|>", 4)],
    ids=["text", "512-tokens", "literal-end-marker"],
)
def test_count_tokens_can_include_the_embedding_end_marker(
    tokenizer_cache: Path, text: str, expected: int
) -> None:
    tokenizer = load_tokenizer(cache_dir=tokenizer_cache, local_files_only=True)

    assert count_tokens(text, tokenizer=tokenizer, add_special_tokens=True) == expected


def test_load_tokenizer_downloads_only_the_fixed_tokenizer_file(
    tokenizer_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = (
        tokenizer_cache / "models--Qwen--Qwen3-Embedding-0.6B" / "snapshots"
        / "c54f2e6e80b2d7b7de06f51cec4959f6b3e03418" / "tokenizer.json"
    )
    download = Mock(return_value=str(snapshot))
    monkeypatch.setattr("arkb.knowledge.embeddings.hf_hub_download", download)

    tokenizer = load_tokenizer(cache_dir=tokenizer_cache)

    assert count_tokens("ab", tokenizer=tokenizer) == 2
    download.assert_called_once_with(
        repo_id="Qwen/Qwen3-Embedding-0.6B",
        filename="tokenizer.json",
        revision="c54f2e6e80b2d7b7de06f51cec4959f6b3e03418",
        cache_dir=tokenizer_cache,
        local_files_only=False,
        token=False,
    )


def test_load_tokenizer_reports_an_offline_cache_miss_without_network_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def reject_network(*args, **kwargs):
        pytest.fail("Offline tokenizer loading must not make an HTTP request.")

    monkeypatch.setattr(httpx.Client, "request", reject_network)

    with pytest.raises(LocalEntryNotFoundError):
        load_tokenizer(cache_dir=tmp_path, local_files_only=True)


def test_load_tokenizer_propagates_download_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    download = Mock(side_effect=httpx.ConnectError("Download unavailable."))
    monkeypatch.setattr("arkb.knowledge.embeddings.hf_hub_download", download)

    with pytest.raises(httpx.ConnectError, match="Download unavailable"):
        load_tokenizer(cache_dir=tmp_path)
