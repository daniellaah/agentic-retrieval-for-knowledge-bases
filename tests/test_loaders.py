from pathlib import Path

import pytest

from obsidian_rag.loaders import load_notes


def test_load_notes_reads_title_content_and_source(tmp_path: Path) -> None:
    (tmp_path / "reading.md").write_text(
        "# Reading Notes\n\n"
        "Keep the author's meaning.\n\n"
        "## Source\n\n"
        "Sönke Ahrens.\n",
        encoding="utf-8",
    )

    notes = load_notes(tmp_path)

    assert [(note.title, note.content, note.source) for note in notes] == [
        (
            "Reading Notes",
            "Keep the author's meaning.\n\n## Source\n\nSönke Ahrens.",
            "reading.md",
        )
    ]


def test_load_notes_returns_notes_in_filename_order(tmp_path: Path) -> None:
    for filename in ["zeta.md", "alpha.md", "Beta.md"]:
        (tmp_path / filename).write_text("# A Note\n\nAn idea.\n", encoding="utf-8")

    notes = load_notes(tmp_path)

    assert [note.source for note in notes] == ["Beta.md", "alpha.md", "zeta.md"]


def test_load_notes_uses_filename_when_no_level_one_heading_exists(
    tmp_path: Path,
) -> None:
    (tmp_path / "reading_notes.md").write_text(
        "## Reading\n\nA useful passage.\n", encoding="utf-8"
    )

    notes = load_notes(tmp_path)

    assert [(note.title, note.content) for note in notes] == [
        ("reading_notes", "## Reading\n\nA useful passage.")
    ]


def test_load_notes_uses_first_level_one_heading_and_preserves_other_content(
    tmp_path: Path,
) -> None:
    (tmp_path / "ideas.md").write_text(
        "Introductory text.\n"
        "# Main Idea\n\n"
        "Develop one idea.\n\n"
        "# Another Heading\n\n"
        "Keep this section.\n",
        encoding="utf-8",
    )

    notes = load_notes(tmp_path)

    assert [(note.title, note.content) for note in notes] == [
        (
            "Main Idea",
            "Introductory text.\n\nDevelop one idea.\n\n"
            "# Another Heading\n\nKeep this section.",
        )
    ]


def test_load_notes_reports_a_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_notes(tmp_path / "missing")


def test_load_notes_reads_only_markdown_files_in_the_given_directory(
    tmp_path: Path,
) -> None:
    (tmp_path / "current.md").write_text("# Current\n\nAn idea.\n", encoding="utf-8")
    (tmp_path / "draft.txt").write_text("An unfinished draft.\n", encoding="utf-8")
    archive = tmp_path / "archive.md"
    archive.mkdir()
    (archive / "old.md").write_text("# Old\n\nAn earlier idea.\n", encoding="utf-8")

    notes = load_notes(tmp_path)

    assert [note.source for note in notes] == ["current.md"]


def test_load_notes_returns_an_empty_list_for_an_empty_directory(tmp_path: Path) -> None:
    assert load_notes(tmp_path) == []
