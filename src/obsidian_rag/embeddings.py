"""Generate validated embeddings in bounded, retryable Ollama batches."""

from collections.abc import Iterator, Sequence
import math
import time

from httpx import TransportError
import numpy as np
from numpy.typing import NDArray
from ollama import Client, ResponseError


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
    max_retries: int = 0, retry_delay: float = 0.25,
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
                        ("max_batch_tokens", max_batch_tokens)):
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
                response = client.embed(model=model, input=texts[start:end], truncate=False)
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
    max_retries: int = 0, retry_delay: float = 0.25,
) -> NDArray:
    """Collect batches in input order; empty input returns a (0, 0) matrix.

    Defaults preserve float64 values and propagate errors without retries. Use
    iter_embedding_batches when successful batches need durable checkpoints.
    """
    batches = [vectors for _, vectors in iter_embedding_batches(
        texts, client=client, model=model, batch_size=batch_size, token_counts=token_counts,
        max_batch_tokens=max_batch_tokens, dimensions=dimensions, dtype=dtype,
        normalization=normalization, max_retries=max_retries, retry_delay=retry_delay,
    )]
    return np.concatenate(batches) if batches else np.empty((0, 0), dtype=dtype)
