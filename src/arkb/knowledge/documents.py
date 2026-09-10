"""Load and access current Markdown documents in the existing flat directory scope."""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from arkb.knowledge.chunking import _sections, whole_note_chunks
from arkb.knowledge.models import ChunkRecord, Note, _document_id, _require_digest, _require_text


@dataclass(frozen=True)
class DocumentSlice:
    """Current document text in body coordinates, without an indexed chunk identity."""

    document_id: str
    document_revision: str
    source: str
    title: str
    content: str
    start_char: int
    end_char: int
    section_id: str | None = None
    heading_path: tuple[str, ...] = ()


def _load_note(path: Path) -> Note:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    title = path.stem
    content = text
    for index, line in enumerate(lines):
        if line.startswith("# "):
            title = line.removeprefix("# ").strip()
            content = "".join(lines[:index] + lines[index + 1:])
            break
    return Note(title=title, content=content.strip(), source=path.name)


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
        notes.append(_load_note(path))
    return notes


class DocumentAccess:
    """Resolve existing document IDs and read live files without retaining content.

    The directory and vault are supplied by the application, never a tool call.
    IDs use the same vault/path namespace as ChunkRecord.document_id. File edits
    are visible on the next call; deleting or renaming a file removes its old ID.
    Symlinks outside the knowledge root are excluded from document access.
    """

    def __init__(self, directory: Path, *, vault_id: str):
        _require_text(vault_id, 'vault_id')
        self.directory = Path(directory).resolve()
        self.vault_id = vault_id

    def _paths(self, source: str | None = None) -> Iterator[Path]:
        if source is not None:
            _require_text(source, 'source')
        for path in sorted(self.directory.iterdir()):
            if (path.suffix == '.md' and (source is None or path.name == source)
                    and path.resolve().is_relative_to(self.directory) and path.is_file()):
                yield path

    def records(self, *, source: str | None = None) -> Iterator[ChunkRecord]:
        """Yield current complete bodies in filename order, filtering before I/O.

        These are source slices represented with existing records, not indexed
        chunks; consumers must not advertise their synthetic chunk IDs.
        """
        for path in self._paths(source):
            note = _load_note(path)
            yield ChunkRecord.from_note(whole_note_chunks([note])[0], note=note,
                                        vault_id=self.vault_id)

    def read(self, document_id: str | None = None, *, source: str | None = None,
             section_id: str | None = None,
             start_char: int | None = None, end_char: int | None = None) -> DocumentSlice:
        """Read a full body, Markdown section, or end-exclusive character range.

        Supply a document ID or exact source filename; if both are provided,
        they must identify the same document. Source lookup
        uses the same flat directory scope and symlink exclusions as ID lookup.
        A missing range endpoint means the corresponding document boundary.
        Sections include their heading and direct body, up to the next heading.
        Section IDs always refer to current Markdown sections. Use an unqualified
        read for a whole document, including hits from whole-note indexes.
        """
        if document_id is None and source is None:
            raise ValueError('Supply document_id or source.')
        if document_id is not None:
            _require_digest(document_id, 'document_id')
        if source is not None:
            _require_text(source, 'source')
        if section_id is not None:
            _require_digest(section_id, 'section_id')
            if start_char is not None or end_char is not None:
                raise ValueError('section_id and character range are mutually exclusive.')
        for name, value in (('start_char', start_char), ('end_char', end_char)):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f'{name} must be a nonnegative integer.')
        if start_char is not None and end_char is not None and start_char > end_char:
            raise ValueError('start_char must not exceed end_char.')

        # Resolve by path identity without loading unrelated document bodies.
        path = next((path for path in self._paths(source) if document_id is None
                     or _document_id(self.vault_id, path.name) == document_id), None)
        if path is None:
            raise LookupError(f'No document matches document_id={document_id!r}, source={source!r}.')
        try:
            note = _load_note(path)
        except FileNotFoundError as error:
            raise LookupError(f'Document no longer exists: {path.name}.') from error
        heading_path = ()
        if section_id is not None:
            section = next((s for s in _sections(note) if s.section_id == section_id), None)
            if section is None:
                raise LookupError(f'Unknown section: {section_id}.')
            start, end = section.blocks[0].start, section.blocks[-1].end
            heading_path = section.heading_path
        else:
            start = 0 if start_char is None else start_char
            end = len(note.content) if end_char is None else end_char
            if not 0 <= start <= end <= len(note.content):
                raise ValueError('Character range is outside the current document body.')
        return DocumentSlice(_document_id(self.vault_id, note.source), note.document_revision,
                             note.source, note.title, note.content[start:end], start, end,
                             section_id, heading_path)


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
