"""Prepare embedding inputs and call the model in validated, retryable batches."""

from collections.abc import Iterator, Sequence
import math
import time

from httpx import TransportError
import numpy as np
from numpy.typing import NDArray
from ollama import Client, ResponseError
from tokenizers import Tokenizer

from obsidian_rag.knowledge_base.chunking import Chunk
from obsidian_rag.knowledge_base.vector_index.manifest import EmbeddingSpec
from obsidian_rag.knowledge_base.tokenization import count_tokens


DOCUMENT_TEMPLATE = "title-body-v1"
DEFAULT_QUERY_INSTRUCTION = (
    "Given a question, retrieve relevant notes that help answer it."
)


def prepare_document(
    chunk: Chunk, *, document_template: str = DOCUMENT_TEMPLATE,
) -> str:
    """Return title, two newlines, and the verbatim chunk body.

    Pass EmbeddingSpec.document_template when using an indexing specification.
    Unknown templates fail instead of silently producing mislabeled cache data.
    Source paths and positions are metadata and do not enter the model input.
    Empty bodies are allowed when the title supplies nonblank text, and empty
    titles are allowed when the body does. No source fields are modified.
    """
    if document_template != DOCUMENT_TEMPLATE:
        raise ValueError(f"Unsupported document_template: {document_template!r}.")
    if not isinstance(chunk, Chunk):
        raise ValueError("chunk must be a Chunk.")
    if not isinstance(chunk.title, str) or not isinstance(chunk.content, str):
        raise ValueError("chunk title and content must be strings.")
    text = f"{chunk.title}\n\n{chunk.content}"
    _require_text(text, "document input")
    return text


def prepare_query(
    question: str, *, instruction: str = DEFAULT_QUERY_INSTRUCTION,
) -> str:
    """Return a Qwen retrieval instruction followed by the original question.

    The default exactly preserves the existing CLI format, including no space
    after 'Query:'. Pass instruction='' to embed the raw question, or supply
    IndexManifest.query_instruction explicitly when querying a stored index.
    A whitespace-only instruction is invalid. Preserve all supplied whitespace
    and Unicode; callers still pass the original question to answer generation.
    This format is not a universal prompt for other embedding model families.
    """
    _require_text(question, "question")
    if instruction == "":
        return question
    _require_text(instruction, "instruction")
    return f"Instruct: {instruction}\nQuery:{question}"


def validate_input_tokens(
    text: str, *, tokenizer: Tokenizer, max_tokens: int,
    source: str = "embedding input",
) -> int:
    """Count one complete, nonblank input and reject it if over max_tokens.

    Count the assembled text once with add_special_tokens=True; counting its
    parts separately can change BPE boundaries and omit model markers. Return
    the full count, accepting equality with the limit. Errors identify source
    (for example 'notes/topic.md, chunk 2') and counts without echoing the text.

    The caller supplies a tokenizer matching the model/runtime and the active
    per-input context limit, not the chunk body budget or a batch token limit.
    Padding and truncation must be disabled; never truncate or mutate input or
    tokenizer configuration. Keep truncate=False on the actual model request.
    Tokenizer errors propagate; there is no fallback to character counting.
    """
    _require_text(text, "embedding input")
    _require_text(source, "source")
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer.")
    if tokenizer.truncation is not None or tokenizer.padding is not None:
        raise ValueError("Input tokenizer must have padding and truncation disabled.")
    token_count = count_tokens(text, tokenizer=tokenizer, add_special_tokens=True)
    if token_count > max_tokens:
        raise ValueError(
            f"{source}: embedding input has {token_count} tokens; maximum is {max_tokens}."
        )
    return token_count


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string.")


def resolve_embedding_spec(client: Client, model: str, *, context_length: int) -> EmbeddingSpec:
    """Resolve the installed, validated model artifact and its embedding space."""
    if model != 'qwen3-embedding:0.6b':
        raise ValueError('Persistent indexing currently requires the validated qwen3-embedding:0.6b tokenizer pairing.')
    matches = [entry for entry in client.list().models if entry.model == model]
    if len(matches) != 1 or not matches[0].digest:
        raise ValueError(f'Embedding model is not installed or its digest is unavailable: {model}.')
    info = client.show(model).modelinfo or {}
    dimensions = [value for key, value in info.items() if key.endswith('.embedding_length')]
    limits = [value for key, value in info.items() if key.endswith('.context_length')]
    if len(dimensions) != 1 or len(limits) != 1 or type(limits[0]) is not int:
        raise ValueError('Embedding model dimensions/context metadata are unavailable.')
    if context_length > limits[0]:
        raise ValueError('Requested context length exceeds the model context limit.')
    return EmbeddingSpec(model=model, model_revision=matches[0].digest, dimensions=dimensions[0],
                         document_template=DOCUMENT_TEMPLATE)


def validate_vectors(
    vectors, *, rows: int, dimensions: int | None = None,
    dtype: str = "float64", normalization: str = "none",
) -> NDArray:
    """Copy and validate a matrix without silently normalizing its vectors."""
    if dtype not in ("float32", "float64") or normalization not in ("none", "l2"):
        raise ValueError("Unsupported vector representation.")
    with np.errstate(over="ignore", invalid="ignore"):
        matrix = np.array(vectors, dtype=dtype, copy=True)
    if (matrix.ndim != 2 or matrix.shape[0] != rows or matrix.shape[1] == 0
            or (dimensions is not None and matrix.shape[1] != dimensions)):
        raise ValueError("Expected one nonempty embedding vector per input text with matching dimensions.")
    if not np.isfinite(matrix).all():
        raise ValueError("Embedding vectors must contain only finite values.")
    with np.errstate(over="ignore", under="ignore"):
        norms = np.linalg.norm(matrix.astype(np.float64), axis=1)
    if np.any(norms == 0):
        raise ValueError("Embedding vectors must not be zero vectors.")
    if not np.isfinite(norms).all():
        raise ValueError("Embedding vector norms must be finite.")
    if normalization == "l2" and not np.allclose(norms, 1.0, atol=1e-6, rtol=1e-5):
        raise ValueError("Expected L2-normalized embedding vectors.")
    return matrix


def iter_embedding_batches(
    texts: Sequence[str], *, client: Client, model: str = "qwen3-embedding:0.6b",
    batch_size: int = 32, token_counts: Sequence[int] | None = None,
    max_batch_tokens: int | None = None, dimensions: int | None = None,
    dtype: str = "float64", normalization: str = "none",
    max_retries: int = 0, retry_delay: float = 0.25, context_length: int | None = None,
) -> Iterator[tuple[int, NDArray]]:
    """Yield (start offset, vectors), letting callers checkpoint each batch.

    Validate the complete plan before the first request. Token counts must cover
    complete model inputs; oversized individual inputs fail before any work.
    Retries apply only to transport errors and HTTP 429/500/502/503/504, with
    exponential delays capped at five seconds. A failed response is never yielded.
    Consistent dimensions are enforced across batches. Truncation stays disabled.
    """
    texts = list(texts)
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Input texts must not be blank and must be strings.")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be nonblank.")
    for name, value in (("batch_size", batch_size), ("dimensions", dimensions),
                        ("max_batch_tokens", max_batch_tokens), ("context_length", context_length)):
        if value is None and name != "batch_size":
            continue
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if type(max_retries) is not int or not 0 <= max_retries <= 8:
        raise ValueError("max_retries must be an integer between 0 and 8.")
    if isinstance(retry_delay, bool) or not math.isfinite(retry_delay) or not 0 <= retry_delay <= 5:
        raise ValueError("retry_delay must be finite and between 0 and 5 seconds.")
    if dtype not in ("float32", "float64") or normalization not in ("none", "l2"):
        raise ValueError("Unsupported vector representation.")
    counts = list(token_counts) if token_counts is not None else None
    if max_batch_tokens is not None and counts is None:
        raise ValueError("max_batch_tokens requires complete input token_counts.")
    if counts is not None:
        if len(counts) != len(texts) or any(type(n) is not int or n <= 0 for n in counts):
            raise ValueError("Expected one positive token count per input text.")
        if max_batch_tokens is not None and any(n > max_batch_tokens for n in counts):
            raise ValueError("An individual input exceeds max_batch_tokens.")

    start = 0
    while start < len(texts):
        end = start
        tokens = 0
        while end < len(texts) and end - start < batch_size:
            count = counts[end] if counts is not None else 0
            if max_batch_tokens is not None and tokens + count > max_batch_tokens:
                break
            tokens += count
            end += 1
        for attempt in range(max_retries + 1):
            try:
                options = {"options": {"num_ctx": context_length}} if context_length is not None else {}
                response = client.embed(model=model, input=texts[start:end], truncate=False, **options)
                break
            except (ConnectionError, TransportError, ResponseError) as error:
                transient = not isinstance(error, ResponseError) or error.status_code in (
                    429, 500, 502, 503, 504,
                )
                if not transient or attempt == max_retries:
                    raise
                time.sleep(min(retry_delay * 2 ** attempt, 5.0))
        matrix = validate_vectors(response.embeddings, rows=end - start, dimensions=dimensions,
                                  dtype=dtype, normalization=normalization)
        dimensions = matrix.shape[1]
        yield start, matrix
        start = end


def embed_texts(
    texts: list[str], *, client: Client, model: str = "qwen3-embedding:0.6b",
    batch_size: int = 32, token_counts: Sequence[int] | None = None,
    max_batch_tokens: int | None = None, dimensions: int | None = None,
    dtype: str = "float64", normalization: str = "none",
    max_retries: int = 0, retry_delay: float = 0.25, context_length: int | None = None,
) -> NDArray:
    """Collect batches in input order; empty input returns a (0, 0) matrix.

    Defaults preserve float64 values and propagate errors without retries. Use
    iter_embedding_batches when successful batches need durable checkpoints.
    """
    batches = [vectors for _, vectors in iter_embedding_batches(
        texts, client=client, model=model, batch_size=batch_size, token_counts=token_counts,
        max_batch_tokens=max_batch_tokens, dimensions=dimensions, dtype=dtype,
        normalization=normalization, max_retries=max_retries, retry_delay=retry_delay,
        context_length=context_length,
    )]
    return np.concatenate(batches) if batches else np.empty((0, 0), dtype=dtype)
