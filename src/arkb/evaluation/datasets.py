"""JSONL experiment inputs and reproducibility fingerprints."""

import hashlib
import json
from pathlib import Path

from arkb.evaluation.models import AgentEvalCase


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON field: {key}.')
        result[key] = value
    return result


def parse_agent_eval_dataset(raw: bytes, *, notes_dir: Path | None = None) -> list[AgentEvalCase]:
    """Validate frozen UTF-8 JSONL bytes; optionally check live source existence.

    No index, model, or retrieval engine is loaded. Source checks mirror the
    current flat document scope and exclude symlinks outside the notes root.
    """
    root = Path(notes_dir).resolve() if notes_dir is not None else None
    if root is not None and not root.is_dir():
        raise ValueError(f'notes_dir is not a directory: {root}.')
    cases, ids = [], set()
    for number, line in enumerate(raw.decode('utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, object_pairs_hook=_unique_object)
            if not isinstance(row, dict):
                raise ValueError('case must be a JSON object.')
            case = AgentEvalCase(**row)
            if case.id in ids:
                raise ValueError(f'duplicate case id: {case.id}.')
            if root is not None:
                for source in case.expected_sources:
                    path = root / source
                    if not path.is_file() or not path.resolve().is_relative_to(root):
                        raise ValueError(f'expected source does not exist in notes_dir: {source}.')
        except (ValueError, TypeError) as error:
            raise ValueError(f'Agent dataset line {number}: {error}') from error
        ids.add(case.id)
        cases.append(case)
    if not cases:
        raise ValueError('Agent dataset must contain at least one case.')
    return cases


def load_agent_eval_dataset(path: Path, *, notes_dir: Path | None = None) -> list[AgentEvalCase]:
    """Load cases in file order and fail before execution on invalid annotations."""
    return parse_agent_eval_dataset(Path(path).read_bytes(), notes_dir=notes_dir)


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
