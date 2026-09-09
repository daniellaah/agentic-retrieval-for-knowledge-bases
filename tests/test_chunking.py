from dataclasses import FrozenInstanceError
from functools import partial

import pytest
from tokenizers import Tokenizer, models

from arkb.indexing.chunking import chunk_notes, whole_note_chunks
from arkb.indexing.loaders import Note
from arkb.tokenization import count_tokens


def test_whole_note_chunks_preserves_each_note_and_its_origin() -> None:
    notes = [
        Note(title="First", content="First idea.", source="first.md"),
        Note(title="Second", content="Second idea.", source="second.md"),
        Note(title="Title only", content="", source="empty.md"),
    ]

    chunks = whole_note_chunks(notes)

    assert [
        (chunk.title, chunk.content, chunk.source, chunk.chunk_index,
         chunk.start_char, chunk.end_char)
        for chunk in chunks
    ] == [
        ("First", "First idea.", "first.md", 0, 0, 11),
        ("Second", "Second idea.", "second.md", 0, 0, 12),
        ("Title only", "", "empty.md", 0, 0, 0),
    ]


def test_chunk_notes_keeps_short_notes_separate_and_complete() -> None:
    notes = [
        Note(title="A", content="  idea  ", source="a.md"),
        Note(title="B", content="第二条笔记。", source="b.md"),
    ]

    chunks = chunk_notes(notes, count_tokens=len, chunk_size=8, chunk_overlap=0)

    assert [(chunk.content, chunk.source, chunk.chunk_index) for chunk in chunks] == [
        ("  idea  ", "a.md", 0),
        ("第二条笔记。", "b.md", 0),
    ]


@pytest.mark.parametrize(
    ("size", "overlap"),
    [(0, 0), (-1, 0), (True, 0), (4.5, 0),
     (4, -1), (4, True), (4, 1.5), (4, 4), (4, 5)],
)
def test_chunk_notes_rejects_invalid_budgets_even_without_notes(size, overlap) -> None:
    with pytest.raises(ValueError):
        chunk_notes([], count_tokens=len, chunk_size=size, chunk_overlap=overlap)


def test_chunk_notes_splits_long_unbroken_text_and_records_exact_ranges() -> None:
    note = Note(title="Long", content="abcdefghij", source="long.md")

    chunks = chunk_notes([note], count_tokens=len, chunk_size=4, chunk_overlap=0)

    assert [
        (chunk.content, chunk.chunk_index, chunk.start_char, chunk.end_char)
        for chunk in chunks
    ] == [("abcd", 0, 0, 4), ("efgh", 1, 4, 8), ("ij", 2, 8, 10)]
    assert all(chunk.title == "Long" and chunk.source == "long.md" for chunk in chunks)


@pytest.mark.parametrize(
    ("text", "size", "expected"),
    [
        ("aa\n\nbbbb\n\ncc", 8, ["aa\n\n", "bbbb\n\ncc"]),
        ("aa\nbbbb\ncc", 6, ["aa\n", "bbbb\n", "cc"]),
        ("One. Two words. End.", 12, ["One. ", "Two words. ", "End."]),
        ("先记下。然后整理内容。最后连接。", 8, ["先记下。", "然后整理内容。", "最后连接。"]),
        ("aaa bbb cc", 7, ["aaa ", "bbb cc"]),
    ],
    ids=["paragraphs", "lines", "english-sentences", "chinese-sentences", "words"],
)
def test_chunk_notes_prefers_natural_boundaries(
    text: str, size: int, expected: list[str]
) -> None:
    note = Note(title="Boundaries", content=text, source="boundaries.md")

    chunks = chunk_notes([note], count_tokens=len, chunk_size=size, chunk_overlap=0)

    assert [chunk.content for chunk in chunks] == expected
    assert "".join(chunk.content for chunk in chunks) == text


@pytest.mark.parametrize(
    ("text", "size", "overlap", "expected"),
    [
        ("abcdefghij", 4, 1, [("abcd", 0, 4), ("defg", 3, 7), ("ghij", 6, 10)]),
        ("ab ab ab ab ", 6, 4,
         [("ab ab ", 0, 6), ("ab ab ", 3, 9), ("ab ab ", 6, 12)]),
    ],
    ids=["character-fallback", "repeated-words-with-natural-overlap"],
)
def test_chunk_notes_overlaps_without_losing_original_positions(
    text: str, size: int, overlap: int, expected: list[tuple[str, int, int]]
) -> None:
    note = Note(title="Repeated", content=text, source="repeated.md")

    chunks = chunk_notes(
        [note], count_tokens=len, chunk_size=size, chunk_overlap=overlap
    )

    assert [(c.content, c.start_char, c.end_char) for c in chunks] == expected
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]


def test_chunk_notes_reports_a_budget_too_small_for_one_character() -> None:
    note = Note(title="Unicode", content="🧠a", source="unicode.md")

    with pytest.raises(ValueError, match="single character"):
        chunk_notes(
            [note], count_tokens=lambda text: len(text.encode("utf-8")),
            chunk_size=3, chunk_overlap=0,
        )


def test_chunks_are_immutable() -> None:
    chunk = whole_note_chunks([Note(title="A", content="a", source="a.md")])[0]

    with pytest.raises(FrozenInstanceError):
        chunk.start_char = 10


def test_empty_collections_produce_no_chunks() -> None:
    assert whole_note_chunks([]) == []
    assert chunk_notes([], count_tokens=len) == []


def test_empty_body_preserves_the_note_title_and_source() -> None:
    note = Note(title="Title only", content="", source="empty.md")

    chunks = chunk_notes([note], count_tokens=len)

    assert [(c.title, c.content, c.source, c.start_char, c.end_char) for c in chunks] == [
        ("Title only", "", "empty.md", 0, 0),
    ]


def test_chunk_indices_and_overlap_restart_at_each_note() -> None:
    notes = [
        Note(title="A", content="abcd", source="a.md"),
        Note(title="B", content="efgh", source="b.md"),
    ]

    chunks = chunk_notes(notes, count_tokens=len, chunk_size=2, chunk_overlap=1)

    assert [(c.content, c.source, c.chunk_index) for c in chunks] == [
        ("ab", "a.md", 0), ("bc", "a.md", 1), ("cd", "a.md", 2),
        ("ef", "b.md", 0), ("fg", "b.md", 1), ("gh", "b.md", 2),
    ]


def test_chunk_notes_measures_merged_text_with_the_supplied_tokenizer() -> None:
    # This real BPE tokenizer merges two characters into one token.
    tokenizer = Tokenizer(models.BPE(
        vocab={"a": 0, "b": 1, "ab": 2}, merges=[("a", "b")],
    ))
    note = Note(title="BPE", content="ababab", source="bpe.md")

    chunks = chunk_notes(
        [note], count_tokens=partial(count_tokens, tokenizer=tokenizer),
        chunk_size=1, chunk_overlap=0,
    )

    assert [(c.content, c.start_char, c.end_char) for c in chunks] == [
        ("ab", 0, 2), ("ab", 2, 4), ("ab", 4, 6),
    ]


def test_chunk_notes_defaults_to_512_tokens_and_64_overlap() -> None:
    note = Note(title="Defaults", content="a" * 1000, source="defaults.md")

    chunks = chunk_notes([note], count_tokens=len)

    assert [(c.start_char, c.end_char) for c in chunks] == [
        (0, 512), (448, 960), (896, 1000),
    ]


@pytest.mark.parametrize(
    "text",
    [
        "Hi\n\nabcdefghi\n\nBye",
        "# Header\r\n\r\n- one\r\n- two\r\n\r\n```py\r\na = 1\r\n```",
        "e\u0301👩🏽\u200d💻中文" * 8,
    ],
    ids=["oversized-paragraph", "markdown-and-crlf", "combining-characters-and-emoji"],
)
def test_chunk_notes_preserves_all_original_text_with_bounded_overlap(text: str) -> None:
    note = Note(title="Lossless", content=text, source="lossless.md")

    chunks = chunk_notes([note], count_tokens=len, chunk_size=8, chunk_overlap=2)

    reconstructed = ""
    covered_end = 0
    for chunk in chunks:
        assert chunk.content == text[chunk.start_char:chunk.end_char]
        assert len(chunk.content) <= 8
        assert chunk.start_char <= covered_end < chunk.end_char
        assert covered_end - chunk.start_char <= 2
        reconstructed += chunk.content[covered_end - chunk.start_char:]
        covered_end = chunk.end_char
    assert reconstructed == text
