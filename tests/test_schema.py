from dataclasses import FrozenInstanceError, replace
import os
from pathlib import Path
import subprocess
import sys

import pytest

from arkb.indexing.chunking import Chunk, chunk_notes, whole_note_chunks
from arkb.indexing.loaders import Note
from arkb.schema import (
    SCHEMA_VERSION,
    ChunkRecord,
    EmbeddingSpec,
    IndexManifest,
    fingerprint_config,
)


@pytest.fixture
def spec() -> EmbeddingSpec:
    return EmbeddingSpec(
        model="qwen3-embedding:0.6b", model_revision="test-model-digest",
        dimensions=1024, document_template="title-body-v1",
    )


@pytest.fixture
def note() -> Note:
    return Note(title="重复片段", content="ab ab ab", source="notes/repeated.md")


@pytest.fixture
def manifest(spec: EmbeddingSpec) -> IndexManifest:
    return IndexManifest(
        index_version="build-1", vault_id="personal", embedding_spec=spec,
        chunking_fingerprint=fingerprint_config({
            "algorithm": "recursive-v1", "chunk_size": 512, "chunk_overlap": 64,
            "tokenizer": {"revision": "test-revision", "normalizer": None},
        }),
        document_count=1, chunk_count=3, query_instruction="Retrieve relevant notes.",
    )


def test_config_fingerprint_is_independent_of_nested_key_order() -> None:
    assert fingerprint_config({"b": [1, True, None], "a": {"x": "中文", "y": 2}}) == (
        fingerprint_config({"a": {"y": 2, "x": "中文"}, "b": [1, True, None]})
    )


@pytest.mark.parametrize("left, right", [
    ({"parts": ["a", "b"]}, {"parts": ["b", "a"]}),
    ({"parts": ["a|b", "c"]}, {"parts": ["a", "b|c"]}),
    ({"text": "é"}, {"text": "e\u0301"}),
    ({"text": "idea"}, {"text": "idea\n"}),
    ({"size": 512}, {"size": 256}),
    ({"tokenizer": {"normalizer": None}}, {"tokenizer": {"normalizer": "NFC"}}),
])
def test_config_fingerprint_preserves_meaningful_differences(left, right) -> None:
    assert fingerprint_config(left) != fingerprint_config(right)


@pytest.mark.parametrize("config", [
    [], {1: "value"}, {"nested": {1: "value"}}, {"values": (1, 2)},
    {"path": Path("notes")}, {"value": float("nan")}, {"value": float("inf")},
    {"values": [{"number": float("-inf")}]},
])
def test_config_rejects_ambiguous_or_non_json_values(config) -> None:
    with pytest.raises(ValueError):
        fingerprint_config(config)


def test_ids_are_stable_across_processes_and_python_hash_seeds(note: Note) -> None:
    # Decode a legacy chunk without the new optional section fields.
    chunk = Chunk(note.content, note.title, note.source, 0, 0, len(note.content))
    record = ChunkRecord.from_note(chunk, note=note, vault_id="personal")
    # These are persisted-format fixtures: changing them requires a schema
    # version/migration decision, even if IDs remain deterministic in one run.
    assert [record.document_id, record.document_revision, record.chunk_id] == [
        "2f46502aa584138f3967787d1df39b71abec7ceefb88be824bbb339abb393cb6",
        "94e3551b6a3f6636c1dc6c392b4dda631665eb881bf72411a5f2ebb6cf957cd1",
        "0c77bf8396a483e0586ec8b01d18db9f006e69cf9ec8107ca00f005a8bab6159",
    ]
    script = """
from arkb.schema import Chunk, ChunkRecord
from arkb.indexing.loaders import Note
note = Note(title="重复片段", content="ab ab ab", source="notes/repeated.md")
chunk = Chunk(note.content, note.title, note.source, 0, 0, len(note.content))
record = ChunkRecord.from_note(chunk, note=note, vault_id="personal")
print(record.document_id, record.document_revision, record.chunk_id)
"""
    for seed in ("1", "2"):
        output = subprocess.check_output(
            [sys.executable, "-B", "-c", script], text=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        assert output.strip().split() == [record.document_id, record.document_revision, record.chunk_id]


def test_repeated_text_keeps_each_source_occurrence(note: Note) -> None:
    note = replace(note, content="ababab")
    chunks = chunk_notes([note], count_tokens=len, chunk_size=2, chunk_overlap=0)
    records = [ChunkRecord.from_note(c, note=note, vault_id="personal") for c in chunks]
    repeated = [record for record in records if record.chunk.content == "ab"]

    assert len(repeated) == 3
    assert len({r.chunk_id for r in repeated}) == 3
    assert len({r.document_id for r in records}) == 1
    assert len({r.document_revision for r in records}) == 1
    assert all(record.chunk is chunk for record, chunk in zip(records, chunks))


def test_identical_notes_have_distinct_ids_but_share_embedding_keys(
    note: Note, spec: EmbeddingSpec,
) -> None:
    renamed = replace(note, source="other.md")
    first = ChunkRecord.from_note(whole_note_chunks([note])[0], note=note, vault_id="personal")
    second = ChunkRecord.from_note(whole_note_chunks([renamed])[0], note=renamed, vault_id="personal")
    other_vault = replace(first, vault_id="work")

    assert len({r.document_id for r in (first, second, other_vault)}) == 3
    assert len({r.chunk_id for r in (first, second, other_vault)}) == 3
    assert first.document_revision == second.document_revision == other_vault.document_revision
    inputs = [f"{r.chunk.title}\n\n{r.chunk.content}" for r in (first, second, other_vault)]
    assert len({spec.embedding_key(text) for text in inputs}) == 1


def test_edit_outside_a_chunk_changes_revision_but_preserves_its_embedding(
    note: Note, spec: EmbeddingSpec,
) -> None:
    chunk = Chunk(content="ab", title=note.title, source=note.source,
                  chunk_index=0, start_char=0, end_char=2)
    edited = replace(note, content=note.content + " more")
    first = ChunkRecord.from_note(chunk, note=note, vault_id="personal")
    second = ChunkRecord.from_note(chunk, note=edited, vault_id="personal")

    assert first.document_id == second.document_id
    assert first.document_revision != second.document_revision
    assert first.chunk_id != second.chunk_id
    assert spec.embedding_key(f"{first.chunk.title}\n\n{first.chunk.content}") == (
        spec.embedding_key(f"{second.chunk.title}\n\n{second.chunk.content}")
    )


def test_title_change_invalidates_document_revision_and_embedding(note: Note, spec: EmbeddingSpec) -> None:
    edited = replace(note, title="New title")
    records = [
        ChunkRecord.from_note(whole_note_chunks([n])[0], note=n, vault_id="personal")
        for n in (note, edited)
    ]
    assert records[0].document_id == records[1].document_id
    assert records[0].document_revision != records[1].document_revision
    assert spec.embedding_key(f"{note.title}\n\n{note.content}") != (
        spec.embedding_key(f"{edited.title}\n\n{edited.content}")
    )


@pytest.mark.parametrize("body", ["", "中文🧠e\u0301\r\n"])
def test_records_preserve_empty_and_unicode_note_coordinates(body: str) -> None:
    note = Note(title="Title", content=body, source="unicode.md")
    chunk = whole_note_chunks([note])[0]
    record = ChunkRecord.from_note(chunk, note=note, vault_id="personal")
    assert record.chunk is chunk
    assert record.chunk.end_char == len(body)


@pytest.mark.parametrize("changes", [
    {"title": "Wrong title"}, {"source": "wrong.md"}, {"content": "xy xy xy"},
    {"chunk_index": -1}, {"chunk_index": True}, {"start_char": -1},
    {"start_char": 0.0}, {"end_char": 0}, {"start_char": 1, "end_char": 9},
])
def test_record_rejects_invalid_or_mismatched_chunks(note: Note, changes: dict) -> None:
    chunk = replace(whole_note_chunks([note])[0], **changes)
    with pytest.raises(ValueError):
        ChunkRecord.from_note(chunk, note=note, vault_id="personal")


def test_empty_span_cannot_point_beyond_source_end() -> None:
    note = Note(title="Empty", content="", source="empty.md")
    chunk = replace(whole_note_chunks([note])[0], start_char=5, end_char=5, section_end_char=5)
    with pytest.raises(ValueError, match="source note"):
        ChunkRecord.from_note(chunk, note=note, vault_id="personal")


@pytest.mark.parametrize("source", [
    "", "/notes/a.md", "../a.md", "notes/../a.md", "./a.md", "notes//a.md",
    "notes/", "C:/notes/a.md", "notes\\a.md",
])
def test_record_rejects_noncanonical_sources(note: Note, source: str) -> None:
    note = replace(note, source=source)
    with pytest.raises(ValueError, match="source"):
        ChunkRecord.from_note(whole_note_chunks([note])[0], note=note, vault_id="personal")


def test_record_validates_stored_revision_and_vault(note: Note) -> None:
    record = ChunkRecord.from_note(whole_note_chunks([note])[0], note=note, vault_id="personal")
    for changes in ({"document_revision": "latest"}, {"vault_id": "  "}):
        with pytest.raises(ValueError):
            replace(record, **changes)


@pytest.mark.parametrize("changes", [
    {"model": "other-model"}, {"model_revision": "new-digest"}, {"dimensions": 256},
    {"document_template": "body-only-v1"}, {"provider": "other-runtime"},
    {"normalization": "none"}, {"dtype": "float32"},
])
def test_embedding_spec_changes_invalidate_reuse(spec: EmbeddingSpec, changes: dict) -> None:
    changed = replace(spec, **changes)
    assert not spec.is_compatible_with(changed)
    assert spec.fingerprint != changed.fingerprint
    assert spec.embedding_key("Title\n\nBody") != changed.embedding_key("Title\n\nBody")


def test_embedding_spec_accepts_identical_configuration(spec: EmbeddingSpec) -> None:
    assert spec.is_compatible_with(replace(spec))
    assert spec.fingerprint == replace(spec).fingerprint
    assert spec.embedding_key("Text") != spec.embedding_key("Text\n")


@pytest.mark.parametrize("changes", [
    {"model": ""}, {"model_revision": " "}, {"document_template": ""}, {"provider": ""},
    {"dimensions": 0}, {"dimensions": -1}, {"dimensions": True}, {"dimensions": 2.0},
    {"normalization": "cosine"}, {"dtype": "int8"},
])
def test_embedding_spec_rejects_invalid_configuration(spec: EmbeddingSpec, changes: dict) -> None:
    with pytest.raises(ValueError):
        replace(spec, **changes)


@pytest.mark.parametrize("text", ["", " \n\t", None])
def test_embedding_key_rejects_blank_inputs(spec: EmbeddingSpec, text) -> None:
    with pytest.raises(ValueError, match="embedding text"):
        spec.embedding_key(text)


def test_manifest_separates_build_identity_query_and_document_settings(manifest: IndexManifest) -> None:
    new_build = replace(manifest, index_version="build-2", status="ready", chunk_count=4)
    new_query = replace(manifest, query_instruction="Retrieve code examples.")
    new_chunks = replace(manifest, chunking_fingerprint=fingerprint_config({"algorithm": "whole-note-v1"}))
    new_model = replace(manifest, embedding_spec=replace(manifest.embedding_spec, model_revision="v2"))

    assert manifest.schema_version == SCHEMA_VERSION
    assert new_build.configuration_fingerprint == manifest.configuration_fingerprint
    assert len({m.configuration_fingerprint for m in (manifest, new_query, new_chunks, new_model)}) == 4
    assert new_query.embedding_spec.embedding_key("Text") == manifest.embedding_spec.embedding_key("Text")
    assert new_chunks.embedding_spec.embedding_key("Text") == manifest.embedding_spec.embedding_key("Text")


@pytest.mark.parametrize("status", ["building", "ready", "failed"])
def test_manifest_can_describe_empty_snapshots(manifest: IndexManifest, status: str) -> None:
    empty = replace(manifest, document_count=0, chunk_count=0, status=status)
    assert empty.document_count == empty.chunk_count == 0


@pytest.mark.parametrize("changes", [
    {"index_version": ""}, {"vault_id": ""}, {"embedding_spec": {}},
    {"chunking_fingerprint": "invalid"}, {"document_count": -1}, {"chunk_count": True},
    {"document_count": 0}, {"chunk_count": 0}, {"document_count": 4},
    {"query_instruction": None}, {"status": "published"},
    {"schema_version": True}, {"schema_version": 1.0}, {"schema_version": 999},
])
def test_manifest_rejects_invalid_metadata(manifest: IndexManifest, changes: dict) -> None:
    with pytest.raises(ValueError):
        replace(manifest, **changes)


def test_records_and_configuration_are_immutable(
    spec: EmbeddingSpec, note: Note, manifest: IndexManifest,
) -> None:
    record = ChunkRecord.from_note(whole_note_chunks([note])[0], note=note, vault_id="personal")
    for item, attribute, value in (
        (spec, "dimensions", 256), (record, "vault_id", "other"),
        (record.chunk, "content", "changed"), (manifest, "status", "ready"),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(item, attribute, value)


@pytest.mark.parametrize('changes', [
    {'section_id': 'invalid'}, {'section_start_char': -1},
    {'section_end_char': True}, {'section_end_char': 1},
    {'section_start_char': 1}, {'section_end_char': 999},
    {'occurrence': -1}, {'occurrence': True}, {'heading_path': (1,)}, {'heading_path': 'Section'},
    {'section_id': None},
])
def test_record_validates_section_provenance(note: Note, changes: dict) -> None:
    chunk = replace(whole_note_chunks([note])[0], **changes)
    with pytest.raises(ValueError):
        ChunkRecord.from_note(chunk, note=note, vault_id='vault')
