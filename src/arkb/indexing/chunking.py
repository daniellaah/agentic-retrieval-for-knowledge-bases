"""Split loaded notes into source-addressable text chunks."""

from collections import deque
from collections.abc import Callable, Sequence
import re

from arkb.schema import Chunk, Note


_SEPARATORS = tuple(re.compile(pattern) for pattern in (
    r"\r?\n[ \t]*\r?\n",
    r"\r?\n",
    r"[。！？]+[ \t]*|[.!?]+(?:[ \t]+|(?=$))",
    r"[ \t]+",
))


def whole_note_chunks(notes: Sequence[Note]) -> list[Chunk]:
    """Wrap each note as one chunk for the whole-note retrieval baseline."""
    return [
        Chunk(
            content=note.content,
            title=note.title,
            source=note.source,
            chunk_index=0,
            start_char=0,
            end_char=len(note.content),
        )
        for note in notes
    ]


def chunk_notes(
    notes: Sequence[Note],
    *,
    count_tokens: Callable[[str], int],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Chunk]:
    """Recursively split each note using a deterministic text-token counter.

    Prefer paragraphs, lines, sentence punctuation, then spaces; oversized units
    fall back to Python character boundaries. Delimiters and whitespace remain
    verbatim. Recount each merged slice rather than adding individual counts.

    chunk_size limits body tokens, excluding titles and added model markers.
    chunk_overlap is an upper target: retain whole trailing split units when
    they fit both budgets. Actual overlap may be smaller, including zero.

    Chunks preserve input note order, with zero-based indices restarting per note.
    All body characters are covered; [start_char, end_char) addresses Note.content,
    not raw file bytes or lines. An empty body remains one title-bearing chunk.

    count_tokens must return a nonnegative integer without padding or truncation.
    Invalid budgets, or a fallback character exceeding chunk_size, raise ValueError.
    Counter errors propagate. No model loading, indexing, or disk writes occur.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if (
        isinstance(chunk_overlap, bool)
        or not isinstance(chunk_overlap, int)
        or not 0 <= chunk_overlap < chunk_size
    ):
        raise ValueError("chunk_overlap must be an integer from zero to chunk_size - 1.")
    chunks = []
    for note in notes:
        spans = _chunk_spans(
            note.content, count_tokens=count_tokens,
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        )
        for index, (start, end) in enumerate(spans):
            chunks.append(Chunk(
                content=note.content[start:end],
                title=note.title,
                source=note.source,
                chunk_index=index,
                start_char=start,
                end_char=end,
            ))
    return chunks


def _chunk_spans(
    text: str, *, count_tokens: Callable[[str], int], chunk_size: int, chunk_overlap: int
) -> list[tuple[int, int]]:
    chunks = []
    pending: deque[tuple[int, int]] = deque()
    for start, end in _split_spans(
        text, 0, len(text), count_tokens=count_tokens, chunk_size=chunk_size
    ):
        if pending and count_tokens(text[pending[0][0]:end]) > chunk_size:
            chunks.append((pending[0][0], pending[-1][1]))
            while pending and (
                count_tokens(text[pending[0][0]:pending[-1][1]]) > chunk_overlap
                or count_tokens(text[pending[0][0]:end]) > chunk_size
            ):
                pending.popleft()
        pending.append((start, end))
    if pending:
        chunks.append((pending[0][0], pending[-1][1]))
    return chunks


def _split_spans(
    text: str,
    start: int,
    end: int,
    *,
    count_tokens: Callable[[str], int],
    chunk_size: int,
    level: int = 0,
) -> list[tuple[int, int]]:
    if count_tokens(text[start:end]) <= chunk_size:
        return [(start, end)]

    for index in range(level, len(_SEPARATORS)):
        cuts = [
            match.end() for match in _SEPARATORS[index].finditer(text, start, end)
            if match.end() < end
        ]
        if cuts:
            spans = []
            cursor = start
            for stop in [*cuts, end]:
                spans.extend(_split_spans(
                    text, cursor, stop,
                    count_tokens=count_tokens,
                    chunk_size=chunk_size,
                    level=index + 1,
                ))
                cursor = stop
            return spans
    spans = []
    for index in range(start, end):
        if count_tokens(text[index:index + 1]) > chunk_size:
            raise ValueError(
                f"chunk_size cannot fit a single character at body offset {index}."
            )
        spans.append((index, index + 1))
    return spans
