"""Load the pinned Qwen3-Embedding-0.6B tokenizer for local token counting."""

import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer


TOKENIZER_REPO_ID = "Qwen/Qwen3-Embedding-0.6B"
TOKENIZER_REVISION = "c54f2e6e80b2d7b7de06f51cec4959f6b3e03418"


def load_tokenizer(
    *, cache_dir: Path | None = None, local_files_only: bool = False
) -> Tokenizer:
    """Load the Qwen tokenizer aligned with Ollama 0.33.2's 0.6b model.

    Only tokenizer.json is fetched from the fixed public Hub revision. By default
    the Hub's configured cache is used; cache_dir selects an explicit cache.
    Set local_files_only=True to prevent network access, including metadata
    requests. A missing offline snapshot raises LocalEntryNotFoundError.

    Load once and reuse the returned tokenizer for multiple counts. No model
    weights or Ollama connection are needed. Unicode normalization is disabled
    to preserve the input's combining characters, as the Ollama backend does.
    The pinned snapshot has no padding or truncation enabled.

    Hub download, filesystem, and tokenizer parsing errors propagate to callers.
    This configuration has not been validated for other embedding models.
    """
    path = hf_hub_download(
        repo_id=TOKENIZER_REPO_ID,
        filename="tokenizer.json",
        revision=TOKENIZER_REVISION,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        token=False,
    )
    tokenizer = Tokenizer.from_file(path)
    # Ollama 0.33.2 preserves combining characters rather than applying NFC.
    tokenizer.normalizer = None
    return tokenizer


def count_tokens(
    text: str, *, tokenizer: Tokenizer, add_special_tokens: bool = False
) -> int:
    """Count text using the tokenizer returned by load_tokenizer.

    The default excludes automatically added special tokens and returns zero for
    empty text. For a complete nonempty embedding input, pass the title and body
    together and set add_special_tokens=True to include the ending token. Literal
    special markers already present in text are counted in either mode.

    This function only measures text; it does not split, truncate, add a title,
    or check the embedding model's context limit. Reuse the tokenizer without
    enabling padding or truncation, which would change the measured length.
    """
    return len(tokenizer.encode(text, add_special_tokens=add_special_tokens).ids)


def tokenizer_fingerprint(tokenizer: Tokenizer) -> str:
    return hashlib.sha256(tokenizer.to_str().encode('utf-8')).hexdigest()
