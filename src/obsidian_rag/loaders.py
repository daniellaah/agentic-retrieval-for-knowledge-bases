"""Read Markdown notes from a directory."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Note:
    """A Markdown note with a title, body, and source filename."""

    title: str
    content: str
    source: str


def load_notes(directory: Path) -> list[Note]:
    """Read UTF-8 .md files in filename order without visiting subdirectories.

    Use the first line starting with "# " as the title, or the filename stem if
    there is no such line. Remove the title line from the body and strip leading
    and trailing whitespace. Keep source references as filenames.

    Filesystem and decoding errors propagate to the caller.
    """
    notes = []
    for path in sorted(directory.iterdir()):
        if path.suffix != ".md" or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
        title = path.stem
        content = text
        for index, line in enumerate(lines):
            if line.startswith("# "):
                title = line.removeprefix("# ").strip()
                content = "".join(lines[:index] + lines[index + 1 :])
                break
        notes.append(
            Note(
                title=title,
                content=content.strip(),
                source=path.name,
            )
        )
    return notes
