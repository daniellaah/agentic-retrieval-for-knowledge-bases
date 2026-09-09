from dataclasses import FrozenInstanceError, asdict, replace
from functools import partial
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import pytest
from tokenizers import Tokenizer, models

from arkb.chunking import chunk_notes, whole_note_chunks
from arkb.schema import Chunk, ChunkRecord, Note
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


def assert_lossless(note: Note, chunks: list[Chunk], size: int, overlap: int) -> None:
    covered = 0
    rebuilt = ""
    for chunk in chunks:
        assert chunk.content == note.content[chunk.start_char:chunk.end_char]
        assert len(chunk.content) <= size
        assert chunk.start_char <= covered <= chunk.end_char
        assert covered - chunk.start_char <= overlap
        if note.content:
            assert chunk.end_char > covered
        rebuilt += chunk.content[covered - chunk.start_char:]
        covered = chunk.end_char
    assert rebuilt == note.content
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_heading_boundaries_and_ancestor_paths_are_preserved_even_with_a_large_budget() -> None:
    text = "Preamble\n\n# Main\nIntro\n\n### Detail ###\nBody\n\n## Next\nNext body\n# Other\nEnd"
    note = Note("Title", text, "folder/note.md")
    chunks = chunk_notes([note], count_tokens=len, chunk_size=1000)

    assert [c.heading_path for c in chunks] == [
        (), ("Main",), ("Main", "Detail"), ("Main", "Next"), ("Other",),
    ]
    assert [c.content for c in chunks] == [
        "Preamble\n\n", "# Main\nIntro\n\n", "### Detail ###\nBody\n\n",
        "## Next\nNext body\n", "# Other\nEnd",
    ]
    for chunk in chunks:
        assert chunk.note_id == note.note_id
        assert chunk.path == note.path == "folder/note.md"
        assert chunk.title == "Title"
        assert chunk.parent_id == chunk.section_id
        assert (chunk.section_start_char, chunk.section_end_char) == (chunk.start_char, chunk.end_char)
    assert_lossless(note, chunks, 1000, 0)


def test_setext_headings_and_repeated_titles_keep_distinct_sections() -> None:
    text = "Main\n====\nIntro\n\nSame\n----\nBody\n\nSame\n----\nBody\n"
    note = Note("Title", text, "note.md")
    chunks = chunk_notes([note], count_tokens=len)

    assert [c.heading_path for c in chunks] == [("Main",), ("Main", "Same"), ("Main", "Same")]
    assert len({c.section_id for c in chunks}) == 3
    assert_lossless(note, chunks, 512, 0)


@pytest.mark.parametrize("fence", ["```python", "~~~~", "````"])
def test_fenced_code_is_atomic_when_it_fits_and_inner_headings_are_opaque(fence: str) -> None:
    marker = fence.rstrip("python")
    code = fence + "\n# not a heading\n\nx = 1\n```\n" + marker + "\n\n"
    # For triple backticks, the first ``` closes the fence; avoid a second opener.
    if marker == "```":
        code = fence + "\n# not a heading\n\nx = 1\n```\n\n"
    text = "# Section\n\nIntro paragraph.\n\n" + code + "After.\n\n# Next\nEnd"
    note = Note("Title", text, "code.md")
    chunks = chunk_notes([note], count_tokens=len, chunk_size=len(code), chunk_overlap=0)

    assert any(code in chunk.content for chunk in chunks)
    assert {c.heading_path for c in chunks} == {("Section",), ("Next",)}
    assert_lossless(note, chunks, len(code), 0)


def test_paragraphs_and_loose_nested_task_lists_stay_whole_when_they_fit() -> None:
    paragraph = "A short paragraph.\n\n"
    items = "- [ ] First\n  continuation\n  - nested\n\n- [x] Second\n\n"
    text = paragraph * 2 + items + paragraph
    note = Note("Tasks", text, "tasks.md")
    chunks = chunk_notes([note], count_tokens=len, chunk_size=len(items), chunk_overlap=0)

    assert [c.content for c in chunks] == [paragraph * 2, items, paragraph]
    assert_lossless(note, chunks, len(items), 0)


def test_oversized_list_splits_at_items_before_wrapped_lines() -> None:
    items = [f"{i}. Item\n   continuation\n\n" for i in range(1, 5)]
    note = Note("List", "".join(items), "list.md")
    chunks = chunk_notes([note], count_tokens=len, chunk_size=len(items[0]) + 2, chunk_overlap=0)
    assert [c.content for c in chunks] == items
    assert_lossless(note, chunks, len(items[0]) + 2, 0)


@pytest.mark.parametrize("text, paths", [
    ("```\n# Fake\ntext\n## Still code", [()]),
    ("~~~\n# Fake\n```\n## Still code\n~~~\n# Real", [(), ("Real",)]),
    ("````\n# Fake\n```\n## Still code\n`````\n# Real", [(), ("Real",)]),
    ("    # Indented code\n    ## More code\n\n# Real", [(), ("Real",)]),
    ("#Not a heading\n####### Neither\n\n# Real", [(), ("Real",)]),
    ("---\ntitle: YAML\n# A YAML comment\n---\n\n# Real", [(), ("Real",)]),
    ("Title on\ntwo lines\n===\nBody", [("Title on two lines",)]),
    ("#\n## ###\ntext", [("",), ("", "")]),
])
def test_markdown_heading_edge_cases(text: str, paths: list[tuple[str, ...]]) -> None:
    note = Note("Title", text, "edge.md")
    chunks = chunk_notes([note], count_tokens=len)
    assert [c.heading_path for c in chunks] == paths
    assert_lossless(note, chunks, 512, 0)


def test_crlf_unicode_links_and_source_ranges_survive_json_roundtrip() -> None:
    text = "## 中文\r\n\r\n[[笔记#标题|alias]] #tag\r\n\r\n👩🏽‍💻 café\r\n\r\n## Next\r\n![[image.png]]"
    note = Note("My note", text, "folder/笔记.md")
    chunks = chunk_notes([note], count_tokens=len, chunk_size=24, chunk_overlap=4)

    for chunk in chunks:
        restored = Chunk(**json.loads(json.dumps(asdict(chunk))))
        assert restored == chunk
        assert restored.chunk_id == chunk.chunk_id
        assert isinstance(restored.heading_path, tuple)
        assert text[chunk.section_start_char:chunk.section_end_char].startswith("## ")
        assert chunk.section_start_char <= chunk.start_char <= chunk.end_char <= chunk.section_end_char
    assert_lossless(note, chunks, 24, 4)


def test_overlap_never_crosses_a_section_boundary() -> None:
    note = Note("Sections", "# A\nabcdefghij\n# B\nklmnopqrst", "sections.md")
    chunks = chunk_notes([note], count_tokens=len, chunk_size=8, chunk_overlap=3)
    boundary = note.content.index("# B")

    assert len(chunks) > 2
    assert all(c.end_char <= boundary if c.heading_path == ("A",) else c.start_char >= boundary
               for c in chunks)
    assert len({c.section_id for c in chunks}) == 2
    assert_lossless(note, chunks, 8, 3)


def test_unchanged_section_ids_survive_edits_insertions_and_reordering_elsewhere() -> None:
    note = Note("Title", "# Intro\nFirst\n\n# Keep\nStable\n\n# Last\nEnd\n\n", "note.md")
    edited = replace(note, content="# Last\nChanged\n\n# Added\nNew\n\n# Intro\nLonger introduction\n\n# Keep\nStable\n\n")
    original = chunk_notes([note], count_tokens=len)
    updated = chunk_notes([edited], count_tokens=len)
    first = next(c for c in original if c.heading_path == ("Keep",))
    second = next(c for c in updated if c.heading_path == ("Keep",))

    assert first.chunk_index != second.chunk_index
    assert first.start_char != second.start_char
    assert (first.note_id, first.section_id, first.chunk_id) == (second.note_id, second.section_id, second.chunk_id)
    records = [ChunkRecord.from_note(c, note=n, vault_id="vault") for c, n in ((first, note), (second, edited))]
    assert records[0].document_revision != records[1].document_revision
    assert records[0].chunk_id == records[1].chunk_id
    assert replace(records[0], vault_id="another-vault").chunk_id != records[0].chunk_id


def test_unchanged_chunk_id_survives_an_edit_earlier_in_the_same_section() -> None:
    note = Note("Title", "first!!\n\nstable!!\n\nlast!!!", "note.md")
    edited = replace(note, content=note.content.replace("first!!", "changed!"))
    first, second = [chunk_notes([n], count_tokens=len, chunk_size=12, chunk_overlap=0)[1]
                     for n in (note, edited)]
    assert first.content == second.content == "stable!!\n\n"
    assert first.start_char != second.start_char
    assert first.chunk_id == second.chunk_id


def test_repeated_headings_and_content_have_unique_reproducible_occurrences() -> None:
    note = Note("Repeat", ("# Same\n\n" + "abcdefgh" * 5 + "\n") * 3, "repeat.md")
    chunks = chunk_notes([note], count_tokens=len, chunk_size=8, chunk_overlap=0)
    assert chunks == chunk_notes([note], count_tokens=len, chunk_size=8, chunk_overlap=0)
    assert len({c.section_id for c in chunks}) == 3
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    repeated = [c for c in chunks if c.content == "abcdefgh"]
    assert len(repeated) == 15
    assert [c.occurrence for c in repeated] == list(range(5)) * 3
    renamed = replace(note, source="other.md")
    assert {c.chunk_id for c in chunks}.isdisjoint(
        c.chunk_id for c in chunk_notes([renamed], count_tokens=len, chunk_size=8, chunk_overlap=0))


@pytest.mark.parametrize("text", ["", "a", " ", "\r\n\r\n", "# H", "## H\n\n"])
def test_empty_and_tiny_notes_have_complete_parent_metadata(text: str) -> None:
    note = Note("Title", text, "tiny.md")
    for chunks in (chunk_notes([note], count_tokens=len), whole_note_chunks([note])):
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.note_id == note.note_id
        assert chunk.section_id is not None
        assert (chunk.section_start_char, chunk.section_end_char) == (0, len(text))
        assert ChunkRecord.from_note(chunk, note=note, vault_id="vault").chunk == chunk
        assert_lossless(note, chunks, 512, 0)


@pytest.mark.parametrize("text", [
    "# Large\n\n" + "Paragraph with a [[link]].\n\n" * 400,
    "# Code\n\n```python\n" + "print('hello')\n" * 800 + "```\n",
    "# Long line\n\n" + "x" * 10000,
])
def test_very_large_sections_and_code_blocks_remain_lossless_and_bounded(text: str) -> None:
    note = Note("Large", text, "large.md")
    chunks = chunk_notes([note], count_tokens=len, chunk_size=128, chunk_overlap=24)
    assert len(chunks) > 50
    assert len({c.section_id for c in chunks}) == 1
    assert_lossless(note, chunks, 128, 24)


def test_mixed_markdown_is_lossless_across_many_small_budgets() -> None:
    rng = random.Random(731)
    fragments = ["# Heading\n", "## Child\n", "text\n\n", "- item\n  wrapped\n\n",
                 "```py\n# Code\n\nprint(1)\n```\n", "~~~\nx\n~~~\n", "\n",
                 "Title\n===\n", "中文👩🏽‍💻", "[[note]]", "\r\n", "---\n"]
    for _ in range(80):
        text = "".join(rng.choices(fragments, k=20))
        size = rng.randrange(1, 70)
        overlap = rng.randrange(size)
        note = Note("Mixed", text, "mixed.md")
        chunks = chunk_notes([note], count_tokens=len, chunk_size=size, chunk_overlap=overlap)
        assert_lossless(note, chunks, size, overlap)


def test_chunking_is_stable_across_processes_and_imports_without_optional_dependencies() -> None:
    script = '''
import json, sys
from arkb.chunking import chunk_notes
from arkb.schema import Note, ChunkRecord
note = Note("Title", "# Same\\nabcabc\\n# Same\\nabcabc", "folder/note.md")
chunks = chunk_notes([note], count_tokens=len, chunk_size=6, chunk_overlap=1)
assert not any(name.startswith(("arkb.indexing", "arkb.retrieval", "ollama", "qdrant_client", "tokenizers")) for name in sys.modules)
print(json.dumps([(c.note_id, c.section_id, c.chunk_id, ChunkRecord.from_note(c, note=note, vault_id="vault").chunk_id) for c in chunks]))
'''
    outputs = [subprocess.check_output(
        [sys.executable, "-S", "-B", "-c", script], text=True,
        env={**os.environ, "PYTHONHASHSEED": seed,
             "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
    ) for seed in ("1", "99")]
    assert outputs[0] == outputs[1]
    identities = json.loads(outputs[0])
    assert len({row[2] for row in identities}) == len(identities)


def test_original_import_path_keeps_the_same_public_functions() -> None:
    from arkb.indexing import chunking
    assert chunking.chunk_notes is chunk_notes
    assert chunking.whole_note_chunks is whole_note_chunks
