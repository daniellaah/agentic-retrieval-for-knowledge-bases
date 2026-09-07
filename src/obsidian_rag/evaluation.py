"""Reproducible retrieval comparisons; section coverage is not answer accuracy."""

import argparse
from contextlib import ExitStack, closing
from dataclasses import asdict
from functools import partial
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter
import numpy as np
from obsidian_rag.embeddings import validate_vectors
from obsidian_rag.retrieval import search_numpy


def recall_at_k(reference: list[str], candidate: list[str], k: int) -> float | None:
    """Set recall against exact neighbors; no reference neighbors means undefined."""
    if type(k) is not int or k <= 0:
        raise ValueError('k must be positive.')
    if len(set(reference)) != len(reference) or len(set(candidate)) != len(candidate):
        raise ValueError('Neighbor lists must not contain duplicate IDs.')
    expected = set(reference[:k])
    return len(expected & set(candidate[:k])) / len(expected) if expected else None


def evidence_statistics(chunks, case: dict) -> dict:
    """Measure source groups and union coverage of labeled Note.content spans.

    Labels must explicitly use body_start_char/body_end_char. Overlap is counted
    once; raw Markdown coordinates are never silently substituted. These are
    section-character metrics, not necessary-fact recall or generated-answer scores.
    """
    sources = {chunk.source for chunk in chunks}
    groups = case.get('required_source_groups', [])
    if not isinstance(groups, list) or any(not isinstance(g, list) or not g or
                                          any(not isinstance(s, str) or not s for s in g) for g in groups):
        raise ValueError('Expected nonempty source groups.')
    fractions = []
    for anchor in case.get('evidence_anchors', []):
        start, end = anchor.get('body_start_char'), anchor.get('body_end_char')
        if type(start) is not int or type(end) is not int or not 0 <= start < end:
            raise ValueError('Evidence requires valid body coordinates, with an exclusive end.')
        intervals = sorted((max(start, c.start_char), min(end, c.end_char)) for c in chunks
                           if c.source == anchor['source'] and c.start_char < end and c.end_char > start)
        covered, cursor = 0, start
        for left, right in intervals:
            covered += max(0, right - max(cursor, left))
            cursor = max(cursor, right)
        fractions.append(covered / (end - start))
    return {
        'source_group_recall': sum(any(s in sources for s in g) for g in groups) / len(groups) if groups else None,
        'section_coverage': float(np.mean(fractions)) if fractions else None,
        'all_sections_complete': all(f == 1 for f in fractions) if fractions else None,
    }


def compare_retrieval(records, vectors, query_vectors, cases: list[dict], *, spec,
                      vault_id: str, backends: dict | None = None, top_k: int = 2) -> dict:
    """Compare identical vectors/queries; timings cover search only, including backend I/O.

    Additional backends map mode names to (search callable, exact flag).
    Each callable receives a query vector, top_k and exact; callers bind the
    already validated snapshot, embedding spec and vault when preparing it. Mode order
    rotates between cases to reduce a fixed warm-order advantage. Exact ranking
    ties may select different IDs; raw IDs/scores are retained for inspection.
    """
    if not cases or len({case['id'] for case in cases}) != len(cases):
        raise ValueError('Evaluation requires nonempty cases with unique IDs.')
    if type(top_k) is not int or top_k <= 0:
        raise ValueError('top_k must be positive.')
    query_vectors = validate_vectors(query_vectors, rows=len(cases), dimensions=spec.dimensions,
                                     dtype=spec.dtype, normalization=spec.normalization)
    reference = partial(search_numpy, records, vectors, spec=spec, vault_id=vault_id)
    modes = {'numpy_exact': (reference, True)}
    if backends and 'numpy_exact' in backends:
        raise ValueError('numpy_exact is reserved for the reference.')
    modes.update(backends or {})
    for search, exact in modes.values():
        if not callable(search) or type(exact) is not bool:
            raise ValueError('Evaluation requires search callables and boolean exact flags.')
    by_id = {r.chunk_id: r for r in records}
    for case in cases:
        if not isinstance(case.get('question'), str) or not case['question'].strip():
            raise ValueError('Evaluation questions must be nonblank.')
        evidence_statistics([], case)
        for anchor in case.get('evidence_anchors', []):
            end = max((r.chunk.end_char for r in records if r.chunk.source == anchor['source']), default=-1)
            if anchor['body_end_char'] > end:
                raise ValueError('Evidence anchor is outside the indexed corpus.')
    results = []
    names = list(modes)
    for i, (case, query) in enumerate(zip(cases, query_vectors)):
        row = {'id': case['id'], 'question': case['question'], 'modes': {}}
        order = names[i % len(names):] + names[:i % len(names)]
        for name in order:
            search, exact = modes[name]
            started = perf_counter()
            hits = search(query, top_k=top_k, exact=exact)
            elapsed = (perf_counter() - started) * 1000
            if len(hits) > top_k or any(h.chunk_id not in by_id for h in hits):
                raise ValueError('Evaluation backend returned unknown or excess hits.')
            chunks = [by_id[h.chunk_id].chunk for h in hits]
            row['modes'][name] = {'hits': [asdict(h) for h in hits], 'search_ms': elapsed,
                                  **evidence_statistics(chunks, case)}
        expected = row['modes']['numpy_exact']['hits']
        expected_ids = [h['chunk_id'] for h in expected]
        expected_scores = {h['chunk_id']: h['score'] for h in expected}
        for mode in row['modes'].values():
            actual_ids = [h['chunk_id'] for h in mode['hits']]
            mode['neighbor_recall_at_k'] = recall_at_k(expected_ids, actual_ids, top_k)
            mode['same_top_k_order'] = expected_ids == actual_ids
            deltas = [abs(h['score'] - expected_scores[h['chunk_id']]) for h in mode['hits'] if h['chunk_id'] in expected_scores]
            mode['max_shared_score_delta'] = max(deltas, default=None)
        results.append(row)
    summary = {}
    for name in names:
        entries = [row['modes'][name] for row in results]
        summary[name] = {}
        for key in ('neighbor_recall_at_k', 'source_group_recall', 'section_coverage', 'all_sections_complete'):
            values = [entry[key] for entry in entries if entry[key] is not None]
            summary[name][key] = float(np.mean(values)) if values else None
        times = [entry['search_ms'] for entry in entries]
        summary[name].update(search_ms_mean=float(np.mean(times)), search_ms_p50=float(np.median(times)),
                             search_ms_p95=float(np.percentile(times, 95)),
                             same_order_cases=sum(entry['same_top_k_order'] for entry in entries))
    return {'settings': {'top_k': top_k, 'case_count': len(cases), 'chunk_count': len(records),
                         'vector_bytes': int(np.asarray(vectors).nbytes)},
            'summary': summary, 'results': results}


def main(argv=None) -> int:
    """Evaluate an existing snapshot and save a new, non-overwriting artifact folder."""
    from importlib.metadata import version
    from ollama import Client
    from obsidian_rag.retrieval import connect_qdrant, search_qdrant
    from obsidian_rag.embeddings import resolve_embedding_spec
    from obsidian_rag.embeddings import prepare_query, validate_input_tokens
    from obsidian_rag.embeddings import embed_texts
    from obsidian_rag.indexing import QdrantIndex
    from obsidian_rag.tokenization import tokenizer_fingerprint
    from obsidian_rag.storage import SQLiteStorage
    from obsidian_rag.tokenization import load_tokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vault-id', default='default')
    parser.add_argument('--top-k', type=int, default=2)
    parser.add_argument('--host', default='http://127.0.0.1:11434')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--tokenizer-cache', type=Path)
    parser.add_argument('--qdrant-url')
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('--output must be a new directory; old evaluation artifacts are preserved')
    raw_cases = args.cases.read_bytes()
    cases = [json.loads(line) for line in raw_cases.decode('utf-8').splitlines() if line.strip()]
    with ExitStack() as resources, SQLiteStorage(args.db, read_only=True) as storage:
        manifest = storage.active_manifest(args.vault_id)
        if manifest is None:
            raise ValueError('No active snapshot to evaluate.')
        metadata = storage.build_metadata(manifest.index_version)
        _, records, vectors = storage.load_snapshot(manifest.index_version)
        tokenizer = load_tokenizer(cache_dir=args.tokenizer_cache, local_files_only=args.offline)
        if tokenizer_fingerprint(tokenizer) != metadata['backend']['input']['tokenizer']:
            raise ValueError('Evaluation tokenizer differs from the snapshot.')
        ollama = resources.enter_context(Client(host=args.host, timeout=180, trust_env=False))
        limit = metadata['backend']['input']['max_tokens']
        spec = resolve_embedding_spec(ollama, manifest.embedding_spec.model, context_length=limit)
        if spec != manifest.embedding_spec:
            raise ValueError('Evaluation model differs from the snapshot.')
        prepared = [prepare_query(case['question'], instruction=manifest.query_instruction) for case in cases]
        counts = [validate_input_tokens(text, tokenizer=tokenizer, max_tokens=limit, source=case['id'])
                  for case, text in zip(cases, prepared)]
        started = perf_counter()
        queries = embed_texts(prepared, client=ollama, model=spec.model, token_counts=counts,
                              dimensions=spec.dimensions, dtype=spec.dtype, normalization=spec.normalization,
                              context_length=limit)
        embedding_seconds = perf_counter() - started
        modes, server_info = {}, None
        if metadata['backend']['kind'] == 'qdrant':
            client = resources.enter_context(closing(connect_qdrant(args.qdrant_url or metadata['backend']['url'], 30)))
            store = QdrantIndex(client, metadata['backend']['collection'], spec, vault_id=args.vault_id)
            store.verify_snapshot(records, vectors)
            info = store.check_configuration()
            server_info = {'version': client.info().version, 'indexed_vectors': info.indexed_vectors_count,
                           'hnsw_config': info.config.hnsw_config.model_dump(mode='json')}
            search = partial(search_qdrant, client, store.collection, spec=spec, vault_id=args.vault_id)
            modes = {'qdrant_exact': (search, True), 'qdrant_ann': (search, False)}
        report = compare_retrieval(records, vectors, queries, cases, spec=spec, vault_id=args.vault_id,
                                   backends=modes, top_k=args.top_k)
        repo = Path(__file__).resolve().parents[2]
        git = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True)
        run = {'created_at': datetime.now(timezone.utc).isoformat(), 'manifest': asdict(manifest),
               'build_metadata': metadata, 'cases_sha256': hashlib.sha256(raw_cases).hexdigest(),
               'python': platform.python_version(), 'numpy': np.__version__, 'qdrant_client': version('qdrant-client'),
               'qdrant_server': server_info, 'source_commit': git.stdout.strip() if git.returncode == 0 else None,
               'source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
               'query_embedding_seconds': embedding_seconds,
               'sqlite_bytes': args.db.stat().st_size, 'sqlite_wal_bytes': Path(str(args.db) + '-wal').stat().st_size
               if Path(str(args.db) + '-wal').exists() else 0,
               'limits': 'Search timings include backend I/O, exclude embedding and snapshot load; small fixed corpus, no answer generation.'}
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / 'cases.jsonl').write_bytes(raw_cases)
        (args.output / 'results.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in report.pop('results')))
        (args.output / 'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        (args.output / 'run_metadata.json').write_text(json.dumps(run, ensure_ascii=False, indent=2) + '\n')
        corpus = [{'chunk_id': r.chunk_id, 'document_id': r.document_id, 'document_revision': r.document_revision,
                   'source': r.chunk.source, 'chunk_index': r.chunk.chunk_index,
                   'start_char': r.chunk.start_char, 'end_char': r.chunk.end_char,
                   'text_sha256': hashlib.sha256((r.chunk.title + '\n\n' + r.chunk.content).encode()).hexdigest()} for r in records]
        (args.output / 'corpus_manifest.json').write_text(json.dumps(corpus, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
