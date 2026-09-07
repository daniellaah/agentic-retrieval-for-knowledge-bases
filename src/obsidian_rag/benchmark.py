"""Frozen BrowseComp-Plus experiments over the existing RAG pipeline."""

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter

from obsidian_rag.browsecomp import (
    AnswerLabels, BenchmarkQuestion, document_note, file_sha256, read_cases,
    read_corpus, read_jsonl,
)
from obsidian_rag.chunking import whole_note_chunks
from obsidian_rag.schema import ChunkRecord
from obsidian_rag.evaluation import qrel_statistics, rank_documents


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


def _capture_snapshot(storage, manifest, sources, spec, index_version):
    active = storage.get_manifest(index_version) if index_version else storage.active_manifest(manifest['vault_id'])
    if active is None or active.status != 'ready' or active.vault_id != manifest['vault_id']:
        raise ValueError('A ready benchmark snapshot is required.')
    if active.embedding_spec != spec:
        raise ValueError('Runtime embedding specification differs from the snapshot.')
    metadata = storage.build_metadata(active.index_version)
    if metadata['backend'].get('source_scope') != manifest['source_scope']:
        raise ValueError('Snapshot source scope differs from the frozen corpus.')
    seen = set()
    for record in storage.snapshot_records(active.index_version):
        source = sources.get(record.chunk.source)
        if source is None or record.document_revision != source['document_revision']:
            raise ValueError('Snapshot document differs from the frozen corpus.')
        seen.add(record.chunk.source)
    if seen != set(sources):
        raise ValueError('Snapshot does not cover the frozen corpus.')
    return active, metadata


class _EmbeddingTimer:
    """Measure Ollama embedding request time without changing retrieval behavior."""
    def __init__(self, client):
        self.client = client
        self.milliseconds = 0.0

    def embed(self, **kwargs):
        started = perf_counter()
        try:
            return self.client.embed(**kwargs)
        finally:
            self.milliseconds += (perf_counter() - started) * 1000


def score_retrieval(prepared: Path, rows, *, doc_ks) -> dict:
    labels = load_labels(prepared)
    rows = list(rows)
    if [row['query_id'] for row in rows] != list(labels):
        raise ValueError('Results must include each frozen question exactly once, in order.')
    result = {}
    for group in ('evidence', 'gold'):
        result[group] = {}
        for k in doc_ks:
            values = [qrel_statistics([doc['docid'] for doc in row['documents']],
                       getattr(labels[row['query_id']], group + '_docids'), k=k) for row in rows]
            defined = [v for v in values if v['recall'] is not None]
            result[group][str(k)] = {key: sum(v[key] for v in defined) / len(defined) if defined else None
                                     for key in ('recall', 'ndcg')}
            result[group][str(k)]['defined_cases'] = len(defined)
    return result


def _code_identity():
    source = Path(__file__).parent
    git = subprocess.run(['git', '-C', str(source.parents[1]), 'rev-parse', 'HEAD'],
                         capture_output=True, text=True)
    return {'commit': git.stdout.strip() if git.returncode == 0 else None,
            'python': platform.python_version(),
            'source_hashes': {path.name: file_sha256(path) for path in sorted(source.glob('*.py'))}}


def run_benchmark(prepared: Path, *, storage, output: Path, client, tokenizer, spec,
                  chunk_top_k: int = 2, doc_ks=(5, 10, 100, 1000),
                  index_version=None, exact: bool = True, qdrant_client=None) -> dict:
    """Retrieve each frozen question once; retain failures and score them too.

    Each row is flushed before continuing. An interrupted run has no summary
    and cannot be mistaken for a complete evaluation. No answer labels enter
    retrieval, and scoring only reads them after all raw results are saved.
    """
    from obsidian_rag.retrieval import search_index
    if output.exists():
        raise FileExistsError(output)
    if type(chunk_top_k) is not int or chunk_top_k <= 0 or type(exact) is not bool:
        raise ValueError('Use a positive chunk_top_k and boolean exact.')
    doc_ks = tuple(doc_ks)
    if not doc_ks or any(type(k) is not int or k <= 0 for k in doc_ks) or len(set(doc_ks)) != len(doc_ks):
        raise ValueError('doc_ks must contain unique positive cutoffs.')
    manifest, questions, sources = load_prepared(prepared)
    active, metadata = _capture_snapshot(storage, manifest, sources, spec, index_version)
    mapping = {source: value['docid'] for source, value in sources.items()}
    output.mkdir(parents=True, exist_ok=False)
    run = {'format_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
           'prepared_path': str(prepared.resolve()), 'prepared_sha256': file_sha256(prepared / 'manifest.json'),
           'query_ids': manifest['query_ids'], 'snapshot': asdict(active), 'build_metadata': metadata,
           'chunk_top_k': chunk_top_k, 'doc_ks': doc_ks, 'exact': exact,
           'document_ranking': 'max retrieved chunk score, ties ascending docid',
           'code': _code_identity(),
           'timing_scope': 'query_embedding_ms measures embed requests; retrieval_ms includes query preparation, snapshot loading and search.'}
    _write_json(output / 'run.json', run)
    rows = []
    with (output / 'results.jsonl').open('x', encoding='utf-8') as stream:
        for question in questions:
            row = {'query_id': question.query_id, 'question': question.question,
                   'index_version': active.index_version, 'success': False,
                   'retrieval_success': False, 'error': None, 'hits': [], 'documents': []}
            timer = _EmbeddingTimer(client)
            started = perf_counter()
            try:
                hits = search_index(storage, question.question, vault_id=manifest['vault_id'],
                                    spec=spec, tokenizer=tokenizer, client=timer, top_k=chunk_top_k,
                                    exact=exact, index_version=active.index_version, qdrant_client=qdrant_client)
                row['documents'] = rank_documents(hits, mapping)
                row['hits'] = [{'record': asdict(hit.record), 'score': hit.score,
                                'docid': mapping[hit.chunk.source]} for hit in hits]
                row.update(success=True, retrieval_success=True)
            except Exception as error:
                # Per-query failures stay inspectable; interrupts still propagate.
                row['error'] = {'stage': 'retrieval', 'type': type(error).__name__, 'message': str(error)}
            row['timings'] = {'query_embedding_ms': timer.milliseconds,
                              'retrieval_ms': (perf_counter() - started) * 1000}
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            rows.append(row)
    summary = {'query_count': len(rows), 'success_count': sum(row['success'] for row in rows),
               'error_count': sum(not row['success'] for row in rows),
               'retrieval': score_retrieval(prepared, rows, doc_ks=doc_ks),
               'results_sha256': file_sha256(output / 'results.jsonl')}
    _write_json(output / 'summary.json', summary)
    return summary
