"""JSONL experiment inputs and reproducibility fingerprints."""

import hashlib
import json
from pathlib import Path


def load_cases(path: Path) -> tuple[bytes, list[dict]]:
    """Retain the exact input bytes alongside parsed, nonblank JSONL rows."""
    raw = path.read_bytes()
    return raw, [json.loads(line) for line in raw.splitlines() if line.strip()]


def source_hashes(package: Path) -> dict[str, str]:
    """Use package-relative paths so identical basenames stay distinct."""
    return {p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(package.rglob('*.py'))}


def corpus_manifest(records) -> list[dict]:
    """Record the identity, coordinates and text digest of each indexed chunk."""
    return [{'chunk_id': r.chunk_id, 'document_id': r.document_id, 'document_revision': r.document_revision,
             'source': r.chunk.source, 'chunk_index': r.chunk.chunk_index,
             'start_char': r.chunk.start_char, 'end_char': r.chunk.end_char,
             'text_sha256': hashlib.sha256((r.chunk.title + '\n\n' + r.chunk.content).encode()).hexdigest()}
            for r in records]
