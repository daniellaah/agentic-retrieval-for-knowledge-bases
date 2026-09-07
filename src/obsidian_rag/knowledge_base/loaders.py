"""Read Markdown notes from a directory."""

from .models import Note
from pathlib import Path


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


def scan_notes(directory: Path) -> list[Note]:
    """Read the existing flat Markdown scope; fail if it changes during scanning."""
    def inventory():
        result = {}
        for path in sorted(directory.iterdir()):
            if path.suffix == '.md' and path.is_file():
                stat = path.stat()
                result[path.name] = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        return result
    before = inventory()
    notes = load_notes(directory)
    if before != inventory() or {note.source for note in notes} != set(before):
        raise ValueError('Notes changed during scanning; rerun the index command.')
    return notes
