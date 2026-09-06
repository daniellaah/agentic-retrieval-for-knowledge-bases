"""Prepare exact embedding texts and validate complete model-input budgets.

Formatting never strips or normalizes text, loads a tokenizer, or calls a model.
Use the returned document text for both EmbeddingSpec.embedding_key and the
embedding request so cache identity describes the actual input.
"""

from tokenizers import Tokenizer

from obsidian_rag.chunking import Chunk
from obsidian_rag.tokenization import count_tokens


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
