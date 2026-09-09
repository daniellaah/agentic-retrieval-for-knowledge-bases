"""Deterministic Markdown chunking over shared note records, with no I/O.

Public operations: chunk_notes and whole_note_chunks. Budgets measure body text
using a caller-supplied deterministic counter (len also works for characters).
"""

from collections import defaultdict, deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
import re

from arkb.schema import Chunk, Note, _digest

__all__ = ["chunk_notes", "whole_note_chunks"]

_ATX = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*)|[ \t]*)$")
_SETEXT = re.compile(r"^ {0,3}(=+|-+)[ \t]*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_LIST = re.compile(r"^( {0,3})(?:[-+*]|\d{1,9}[.)])[ \t]+", re.MULTILINE)
_SEPARATORS = tuple(re.compile(pattern) for pattern in (
    r"\r?\n[ \t]*\r?\n",                 # Paragraphs, including blank lines.
    r"\r?\n",                           # Lines, including their terminators.
    r"[。！？]+[ \t]*|[.!?]+(?:[ \t]+|(?=$))",
    r"[ \t]+",
))


@dataclass(frozen=True)
class _Block:
    start: int
    end: int
    kind: str = "paragraph"
    level: int = 0
    heading: str = ""


@dataclass(frozen=True)
class _Section:
    section_id: str
    heading_path: tuple[str, ...]
    blocks: list[_Block]


def whole_note_chunks(notes: Sequence[Note]) -> list[Chunk]:
    """Wrap each note in one verbatim chunk, including empty bodies."""
    return [_make_chunk(note, _root_section(note), 0, len(note.content), 0, 0)
            for note in notes]


def chunk_notes(
    notes: Sequence[Note], *, count_tokens: Callable[[str], int],
    chunk_size: int = 512, chunk_overlap: int = 64,
) -> list[Chunk]:
    """Split at Markdown headings, then pack blocks within each section.

    Recognizes ATX/Setext headings, paragraphs, lists, and backtick/tilde fences.
    Headings inside fenced or indented code do not create sections. Oversized
    blocks split recursively at smaller boundaries, ultimately characters.
    Fences are never synthesized: a split code block is still a source slice.

    Overlap retains whole trailing units within a section, up to the overlap
    budget; it may be zero. Combined slices are recounted because token counts
    need not be additive. Titles and other downstream formatting are outside
    the body budget. The counter must return nonnegative integers without
    padding or truncation; counter errors propagate.

    Output follows input order, with indices restarting for each note. All
    ranges address loaded Note.content, not raw files. Empty bodies yield one
    chunk. IDs exclude absolute positions and unrelated section content.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if (isinstance(chunk_overlap, bool) or not isinstance(chunk_overlap, int)
            or not 0 <= chunk_overlap < chunk_size):
        raise ValueError("chunk_overlap must be an integer from zero to chunk_size - 1.")

    chunks = []
    for note in notes:
        index = 0
        for section in _sections(note):
            occurrences: dict[str, int] = defaultdict(int)
            for start, end in _chunk_spans(
                note.content, section.blocks, count_tokens=count_tokens,
                chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            ):
                content = note.content[start:end]
                chunks.append(_make_chunk(note, section, start, end, index, occurrences[content]))
                occurrences[content] += 1
                index += 1
    return chunks


def _root_section(note: Note) -> _Section:
    return _Section(_digest("note-section", {"note_id": note.note_id}), (),
                    [_Block(0, len(note.content))])


def _make_chunk(note: Note, section: _Section, start: int, end: int,
                index: int, occurrence: int) -> Chunk:
    return Chunk(
        content=note.content[start:end], title=note.title, source=note.source,
        chunk_index=index, start_char=start, end_char=end,
        heading_path=section.heading_path, section_id=section.section_id,
        section_start_char=section.blocks[0].start,
        section_end_char=section.blocks[-1].end, occurrence=occurrence,
    )


def _sections(note: Note) -> Iterator[_Section]:
    root = _root_section(note)
    section = _Section(root.section_id, (), [])
    stack: list[tuple[int, str, str]] = []
    occurrences: dict[tuple[str, int, str], int] = defaultdict(int)
    for block in _markdown_blocks(note.content):
        if block.kind == "heading":
            if section.blocks:
                yield section
            while stack and stack[-1][0] >= block.level:
                stack.pop()
            parent = stack[-1][2] if stack else note.note_id
            key = (parent, block.level, block.heading)
            section_id = _digest("heading-section", {
                "parent_id": parent, "level": block.level,
                "heading": block.heading, "occurrence": occurrences[key],
            })
            occurrences[key] += 1
            stack.append((block.level, block.heading, section_id))
            section = _Section(section_id, tuple(heading for _, heading, _ in stack), [])
        section.blocks.append(block)
    yield section if section.blocks else root


def _opening_fence(line: str) -> str | None:
    match = _FENCE.match(line)
    if match and (match[1][0] != "`" or "`" not in match[2]):
        return match[1]
    return None


def _indented(line: str) -> bool:
    return line.startswith(("    ", "\t"))


def _starts_block(line: str) -> bool:
    return bool(_ATX.match(line) or _opening_fence(line) or _LIST.match(line)
                or _indented(line))


def _markdown_blocks(text: str) -> Iterator[_Block]:
    """Scan source lines without normalizing or rendering Markdown.

    This is a small block recognizer, not a full CommonMark parser. Obsidian
    links, tags, callouts, and other inline syntax remain opaque source text.
    Blank separators belong to the preceding block so no text is discarded.
    """
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    bodies = [line.rstrip("\r\n") for line in lines]
    i = 0
    while i < len(lines):
        start = i
        line = bodies[i]
        kind, level, heading = "paragraph", 0, ""
        fence = _opening_fence(line)
        atx = _ATX.match(line)
        frontmatter_end = None
        if i == 0 and line == "---":
            frontmatter_end = next((j for j in range(1, len(lines))
                                    if bodies[j] in ("---", "...")), None)
        if frontmatter_end is not None:
            # Obsidian YAML is opaque metadata, not a Setext heading.
            kind, i = "metadata", frontmatter_end + 1
        elif fence:
            kind = "code"
            closing = re.compile(r"^ {0,3}" + re.escape(fence[0]) +
                                 "{" + str(len(fence)) + r",}[ \t]*$")
            i += 1
            while i < len(lines):
                i += 1
                if closing.match(bodies[i - 1]):
                    break
        elif atx:
            kind, level = "heading", len(atx[1])
            heading = re.sub(r"(?:^|[ \t]+)#+[ \t]*$", "", atx[2] or "").strip()
            i += 1
        elif not line.strip():
            i += 1
        elif _indented(line):
            kind = "code"
            i += 1
            while i < len(lines) and (not bodies[i].strip() or _indented(bodies[i])):
                i += 1
        elif _LIST.match(line):
            kind = "list"
            i += 1
            while i < len(lines):
                current = bodies[i]
                if _ATX.match(current) or _opening_fence(current):
                    break
                if (i > start and not bodies[i - 1].strip() and current.strip()
                        and not (_LIST.match(current) or current.startswith(("  ", "\t")))):
                    break
                i += 1
        else:
            i += 1
            while i < len(lines) and bodies[i].strip():
                underline = _SETEXT.match(bodies[i])
                if underline:
                    kind, level = "heading", 1 if underline[1][0] == "=" else 2
                    heading = " ".join(part.strip() for part in bodies[start:i])
                    i += 1
                    break
                if _starts_block(bodies[i]):
                    break
                i += 1
        while i < len(lines) and not bodies[i].strip():
            i += 1
        yield _Block(offsets[start], offsets[i], kind, level, heading)


def _block_spans(text: str, block: _Block, *, count_tokens: Callable[[str], int],
                 chunk_size: int) -> list[tuple[int, int]]:
    if count_tokens(text[block.start:block.end]) <= chunk_size:
        return [(block.start, block.end)]
    cuts = []
    if block.kind == "list":
        markers = list(_LIST.finditer(text, block.start, block.end))
        if markers:
            indent = min(len(marker[1]) for marker in markers)
            cuts = [marker.start() for marker in markers
                    if len(marker[1]) == indent and marker.start() > block.start]
    spans = []
    cursor = block.start
    for stop in [*cuts, block.end]:
        spans.extend(_split_spans(text, cursor, stop, count_tokens=count_tokens,
                                  chunk_size=chunk_size, level=1 if block.kind == "code" else 0))
        cursor = stop
    return spans


def _chunk_spans(
    text: str, blocks: Sequence[_Block], *, count_tokens: Callable[[str], int],
    chunk_size: int, chunk_overlap: int,
) -> list[tuple[int, int]]:
    chunks = []
    pending: deque[tuple[int, int]] = deque()
    for block in blocks:
        for start, end in _block_spans(text, block, count_tokens=count_tokens, chunk_size=chunk_size):
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
    text: str, start: int, end: int, *, count_tokens: Callable[[str], int],
    chunk_size: int, level: int = 0,
) -> list[tuple[int, int]]:
    if count_tokens(text[start:end]) <= chunk_size:
        return [(start, end)]
    for index in range(level, len(_SEPARATORS)):
        cuts = [match.end() for match in _SEPARATORS[index].finditer(text, start, end)
                if match.end() < end]
        if cuts:
            spans = []
            cursor = start
            for stop in [*cuts, end]:
                spans.extend(_split_spans(text, cursor, stop, count_tokens=count_tokens,
                                          chunk_size=chunk_size, level=index + 1))
                cursor = stop
            return spans
    spans = []
    for index in range(start, end):
        if count_tokens(text[index:index + 1]) > chunk_size:
            raise ValueError(f"chunk_size cannot fit a single character at body offset {index}.")
        spans.append((index, index + 1))
    return spans
