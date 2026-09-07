"""Frozen BrowseComp-Plus experiments over the existing RAG pipeline."""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from obsidian_rag.browsecomp import (
    AnswerLabels, BenchmarkQuestion, document_note, file_sha256, read_cases,
    read_corpus, read_jsonl,
)
from obsidian_rag.chunking import whole_note_chunks
from obsidian_rag.schema import ChunkRecord


def _write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def _write_rows(path, rows):
    with path.open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def _hash_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def prepare_benchmark(corpus_path: Path, cases_path: Path, output: Path, *,
                      dataset_revision: str, query_ids=None) -> dict:
    """Freeze a selected query set against the entire supplied corpus.

    Corpus text is referenced by file hash, not copied. Relevance labels never
    select the runtime corpus. A reduced input corpus remains a custom subset.
    """
    if output.exists():
        raise FileExistsError(output)
    if not isinstance(dataset_revision, str) or not dataset_revision.strip():
        raise ValueError('An identified dataset_revision is required.')
    before = {'corpus': file_sha256(corpus_path), 'cases': file_sha256(cases_path)}
    provenance = None
    download = corpus_path.parent / 'download.json'
    if download.exists():
        provenance = json.loads(download.read_text(encoding='utf-8'))
        if (provenance.get('status') != 'complete'
                or provenance.get('corpus.jsonl_sha256') != before['corpus']
                or provenance.get('cases.jsonl_sha256') != before['cases']):
            raise ValueError('Incomplete or mismatched download hashes.')
    questions, labels = read_cases(cases_path)
    by_id = {q.query_id: (q, label) for q, label in zip(questions, labels)}
    selected = list(query_ids) if query_ids is not None else list(by_id)
    if not selected or len(set(selected)) != len(selected) or any(qid not in by_id for qid in selected):
        raise ValueError('Select nonempty, unique, known query IDs.')
    sources, docids, text_chars, largest = {}, set(), 0, 0
    for doc in read_corpus(corpus_path):
        note = document_note(doc)
        record = ChunkRecord.from_note(whole_note_chunks([note])[0], note=note,
                                       vault_id='browsecomp-' + before['corpus'][:16])
        sources[note.source] = {'docid': doc.docid, 'url': doc.url, 'title': note.title,
                                'text_sha256': _hash_text(doc.text), 'characters': len(doc.text),
                                'document_revision': record.document_revision}
        docids.add(doc.docid)
        text_chars += len(doc.text)
        largest = max(largest, len(doc.text))
    for qid in selected:
        label = by_id[qid][1]
        if not set(label.evidence_docids + label.gold_docids) <= docids:
            raise ValueError('Selected question references evidence missing from the supplied corpus.')
    if before != {'corpus': file_sha256(corpus_path), 'cases': file_sha256(cases_path)}:
        raise ValueError('Inputs changed during preparation.')
    output.mkdir(parents=True, exist_ok=False)
    _write_rows(output / 'questions.jsonl', (asdict(by_id[qid][0]) for qid in selected))
    _write_rows(output / 'labels.jsonl', (asdict(by_id[qid][1]) for qid in selected))
    _write_json(output / 'sources.json', sources)
    manifest = {'format_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
                'dataset_revision': dataset_revision, 'download_provenance': provenance,
                'corpus_scope': 'downloaded_full' if provenance else 'custom',
                'corpus_path': str(corpus_path.resolve()), 'input_hashes': before,
                'corpus_count': len(docids), 'corpus_text_characters': text_chars,
                'largest_document_characters': largest, 'query_ids': selected,
                'vault_id': 'browsecomp-' + before['corpus'][:16],
                'source_scope': 'browsecomp:' + before['corpus'],
                'artifact_hashes': {name: file_sha256(output / name)
                                   for name in ('questions.jsonl', 'labels.jsonl', 'sources.json')}}
    _write_json(output / 'manifest.json', manifest)
    return manifest


def load_prepared(directory: Path) -> tuple[dict, tuple[BenchmarkQuestion, ...], dict]:
    """Validate frozen artifacts and return runtime questions, never answer labels."""
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('format_version') != 1:
        raise ValueError('Unsupported benchmark manifest.')
    for name in ('questions.jsonl', 'labels.jsonl', 'sources.json'):
        if file_sha256(directory / name) != manifest['artifact_hashes'].get(name):
            raise ValueError(f'Prepared artifact hash mismatch: {name}.')
    questions = tuple(BenchmarkQuestion(**row) for row in read_jsonl(directory / 'questions.jsonl'))
    if [q.query_id for q in questions] != manifest['query_ids']:
        raise ValueError('Question IDs differ from the frozen selection.')
    sources = json.loads((directory / 'sources.json').read_text(encoding='utf-8'))
    return manifest, questions, sources


def load_labels(directory: Path) -> dict[str, AnswerLabels]:
    manifest, _, _ = load_prepared(directory)
    labels = [AnswerLabels(**{**row, 'evidence_docids': tuple(row['evidence_docids']),
                             'gold_docids': tuple(row['gold_docids'])})
              for row in read_jsonl(directory / 'labels.jsonl')]
    if [label.query_id for label in labels] != manifest['query_ids']:
        raise ValueError('Labels differ from the frozen selection.')
    return {label.query_id: label for label in labels}


def build_benchmark_index(prepared: Path, *, storage, client, tokenizer, spec,
                          max_input_tokens: int, **settings):
    """Use the existing index lifecycle with a corpus-bound benchmark vault."""
    from obsidian_rag.indexing import build_index
    manifest, _, _ = load_prepared(prepared)
    path = Path(manifest['corpus_path'])
    expected = manifest['input_hashes']['corpus']
    if file_sha256(path) != expected:
        raise ValueError('Prepared corpus hash has changed; prepare a new experiment.')
    notes = [document_note(doc) for doc in read_corpus(path)]
    if file_sha256(path) != expected:
        raise ValueError('Prepared corpus hash changed while loading.')
    return build_index(storage, notes, client=client, tokenizer=tokenizer, spec=spec,
                       vault_id=manifest['vault_id'], max_input_tokens=max_input_tokens,
                       source_scope=manifest['source_scope'], **settings)
