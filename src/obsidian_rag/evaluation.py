"""Reproducible retrieval comparisons; section coverage is not answer accuracy."""

import argparse
from contextlib import ExitStack, closing
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
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


def compare_contexts(question, results, case, *, config, counter) -> dict:
    """Compare the same frozen hits under raw and processed context policies.

    raw is the historical unbounded concatenation; raw_budgeted and built share
    the exact input budget and renderer. Metrics measure source spans/sections,
    not necessary-fact recall, claim support or generated-answer accuracy.
    """
    from obsidian_rag.context import build_context

    if any(hit.record is None for hit in results):
        raise ValueError('Context evaluation requires snapshot-identified hits.')
    identities = {(h.index_version, h.record.vault_id) for h in results}
    if len(identities) > 1:
        raise ValueError('Context evaluation requires one snapshot and vault.')
    modes = {}
    for mode, processing, budget in (('raw', False, None), ('raw_budgeted', False, config), ('built', True, config)):
        started = perf_counter()
        context = build_context(question, results, config=budget, counter=counter, process_evidence=processing)
        elapsed = (perf_counter() - started) * 1000
        blocks = context.evidence_blocks
        groups = {}
        for block in blocks:
            hit = block.origins[0]
            key = (hit.index_version, hit.record.document_id, hit.record.document_revision)
            groups.setdefault(key, []).append((block.start_char, block.end_char))
        unique_chars = 0
        for intervals in groups.values():
            cursor = 0
            for start, end in sorted(intervals):
                unique_chars += max(0, end - max(cursor, start))
                cursor = max(cursor, end)
        chars = sum(len(b.content) for b in blocks)
        modes[mode] = {
            'context': context.to_dict(),
            'metrics': {'prompt_tokens': context.prompt_tokens, 'block_count': len(blocks),
                        'body_characters': chars, 'unique_span_characters': unique_chars,
                        'duplicate_span_fraction': (chars - unique_chars) / chars if chars else 0.0,
                        'fits_budget': context.prompt_tokens <= config.input_budget,
                        'build_ms': elapsed, **evidence_statistics(blocks, case)},
        }
    reference = modes['raw']['metrics']['section_coverage']
    for entry in modes.values():
        coverage = entry['metrics']['section_coverage']
        entry['metrics']['section_coverage_retention'] = coverage / reference if reference else None
    return modes


def citation_statistics(raw_response, sources, *, review: dict | None = None) -> dict:
    """Reference membership and coverage, separately from supplied human/judge labels.

    Every model claim is treated as requiring evidence. This does not detect
    omitted answer facts or multiple facts hidden in one claim. Support labels
    judge the cited sources jointly; the rate uses reviewed claims only and is
    always accompanied by review coverage. No labels means no semantic score.
    """
    from obsidian_rag.citation import CitationParseError, parse_cited_answer, validate_citations
    metrics = {'structure_valid': False, 'references_valid': False, 'claim_count': None,
               'reference_count': None, 'valid_reference_count': None,
               'citation_id_validity': None, 'claim_reference_coverage': None,
               'support_review_coverage': None, 'supported_claim_rate': None,
               'answer_correct': None, 'answer_complete': None, 'issues': []}
    try:
        answer = parse_cited_answer(raw_response)
    except CitationParseError:
        metrics['issues'] = [{'code': 'invalid_structure'}]
        return metrics
    validation = validate_citations(answer, sources)
    known = {s.source_id for s in sources}
    references = [s for c in answer.claims for s in c.source_ids]
    valid = sum(s in known for s in references)
    count = len(answer.claims)
    metrics.update(structure_valid=True, references_valid=validation.references_valid,
                   claim_count=count, reference_count=len(references), valid_reference_count=valid,
                   citation_id_validity=valid / len(references) if references else None,
                   claim_reference_coverage=sum(any(s in known for s in c.source_ids) for c in answer.claims) / count if count else None,
                   issues=validation.to_dict()['issues'])
    if review is not None:
        labels = review.get('claim_support')
        if (not isinstance(review.get('reviewer'), str) or not review['reviewer'].strip()
                or not isinstance(labels, list) or len(labels) != count
                or any(label not in (None, 'supported', 'partial', 'contradicted', 'insufficient') for label in labels)):
            raise ValueError('Review requires an identified reviewer and one support label per claim.')
        # A supplied semantic score cannot bless missing or out-of-context references.
        for claim, label in zip(answer.claims, labels):
            if label == 'supported' and (not claim.source_ids or any(s not in known for s in claim.source_ids)):
                raise ValueError('A supported claim must have valid source references.')
        reviewed = [label for label in labels if label is not None]
        metrics['support_review_coverage'] = len(reviewed) / count if count else None
        metrics['supported_claim_rate'] = reviewed.count('supported') / len(reviewed) if reviewed else None
        for key in ('answer_correct', 'answer_complete'):
            value = review.get(key)
            if value is not None and type(value) is not bool:
                raise ValueError(f'{key} review must be boolean or null.')
            metrics[key] = value
    return metrics


def evaluate_citation_context(context, *, client) -> dict:
    """Run one frozen context once, preserving failures as well as successes."""
    from httpx import HTTPError
    from ollama import ResponseError
    from obsidian_rag.citation import citation_json_schema
    from obsidian_rag.generation import CitedGenerationError, generate_cited_answer
    context.verify_citation_mapping()
    sources = context.citation_sources
    row = {'context': context.to_dict(), 'response_schema': citation_json_schema([s.source_id for s in sources]),
           'success': False, 'result': None, 'raw_response': None, 'error': None}
    started = perf_counter()
    try:
        result = generate_cited_answer(context, client=client)
        payload = result.to_dict()
        payload.pop('raw_response')
        row.update(success=True, result=payload, raw_response=result.raw_response)
        # No-evidence short circuit has no model response; evaluate the application result.
        metric_input = result.raw_response if result.raw_response is not None else json.dumps(result.answer.to_dict())
    except (CitedGenerationError, ValueError, OSError, HTTPError, ResponseError) as error:
        row['raw_response'] = getattr(error, 'raw_response', None)
        row['error'] = {'code': getattr(error, 'code', type(error).__name__), 'message': str(error)}
        row['error']['token_usage'] = getattr(error, 'token_usage', None)
        metric_input = row['raw_response']
    row['generation_ms'] = (perf_counter() - started) * 1000
    row['metrics'] = citation_statistics(metric_input, sources)
    return row


def evaluate_citation_case(question, results, *, config, counter, client) -> dict:
    """Retain per-question budget failures instead of aborting a batch evaluation."""
    from obsidian_rag.context import ContextBudgetError, build_context
    try:
        context = build_context(question, results, config=config, counter=counter, citation_mode='structured')
    except ContextBudgetError as error:
        return {'context': None, 'response_schema': None, 'success': False, 'result': None,
                'raw_response': None, 'generation_ms': 0,
                'error': {'code': 'context_budget', 'message': str(error)},
                'metrics': citation_statistics(None, [])}
    return evaluate_citation_context(context, client=client)


def summarize_citations(rows) -> dict:
    """Macro rates on defined cases, with failures/counts reported alongside."""
    if not rows:
        raise ValueError('Citation summary requires at least one case.')
    summary = {'case_count': len(rows), 'success_count': sum(r['success'] for r in rows),
               'error_count': sum(not r['success'] for r in rows),
               'mean_generation_ms': float(np.mean([r['generation_ms'] for r in rows]))}
    for key in ('structure_valid', 'references_valid', 'citation_id_validity', 'claim_reference_coverage',
                'support_review_coverage', 'supported_claim_rate', 'answer_correct', 'answer_complete'):
        values = [r['metrics'][key] for r in rows if r['metrics'][key] is not None]
        summary[key] = float(np.mean(values)) if values else None
        summary[key + '_defined_cases'] = len(values)
    summary['claim_count'] = sum(r['metrics']['claim_count'] or 0 for r in rows)
    summary['reference_count'] = sum(r['metrics']['reference_count'] or 0 for r in rows)
    return summary

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
    parser.add_argument('--context', action='store_true', help='Also compare context policies on the same NumPy hits.')
    parser.add_argument('--citations', action='store_true', help='Generate and evaluate structured citations on the same NumPy hits.')
    parser.add_argument('--generation-model', default='qwen3.5:4b')
    parser.add_argument('--context-window', type=int, default=8192)
    parser.add_argument('--max-output-tokens', type=int, default=1024)
    parser.add_argument('--context-safety-margin', type=int, default=128)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('--output must be a new directory; old evaluation artifacts are preserved')
    context_config = None
    if args.context or args.citations:
        from obsidian_rag.context import ContextConfig, load_generation_counter
        try:
            context_config = ContextConfig(args.context_window, args.max_output_tokens, args.context_safety_margin)
        except ValueError as error:
            parser.error(str(error))
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
        context_rows, citation_rows = [], []
        if args.context or args.citations:
            from obsidian_rag.context import build_context
            from obsidian_rag.retrieval import SearchResult
            counter = load_generation_counter(client=ollama, model=args.generation_model,
                                              cache_dir=args.tokenizer_cache, local_files_only=args.offline)
            by_id = {record.chunk_id: record for record in records}
            for case, row in zip(cases, report['results']):
                hits = [SearchResult(by_id[h['chunk_id']].chunk, h['score'], by_id[h['chunk_id']], manifest.index_version)
                        for h in row['modes']['numpy_exact']['hits']]
                if args.context:
                    context_rows.append({'id': case['id'], 'question': case['question'],
                                         'modes': compare_contexts(case['question'], hits, case, config=context_config, counter=counter)})
                if args.citations:
                    citation_rows.append({'id': case['id'], 'question': case['question'],
                                          **evaluate_citation_case(case['question'], hits, config=context_config,
                                                                  counter=counter, client=ollama)})
        if citation_rows:
            report['citations'] = {'config': asdict(context_config), 'counter': counter.identity,
                                   'summary': summarize_citations(citation_rows),
                                   'limits': 'Source membership is checked; claim support and answer accuracy are not judged.'}
        if context_rows:
            report['context'] = {'config': asdict(context_config), 'counter': counter.identity,
                                 'generation_model': counter.model, 'is_estimate': counter.is_estimate,
                                 'summary': {}}
            for mode in ('raw', 'raw_budgeted', 'built'):
                entries = [row['modes'][mode]['metrics'] for row in context_rows]
                summary = {}
                for key in entries[0]:
                    values = [entry[key] for entry in entries if entry[key] is not None]
                    summary[key] = float(np.mean(values)) if values else None
                report['context']['summary'][mode] = summary
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
               'limits': 'Search timings exclude embedding and snapshot load. Citation timings include generation and validation when enabled. No semantic scoring without reviews.'}
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / 'cases.jsonl').write_bytes(raw_cases)
        if citation_rows:
            (args.output / 'citation_results.jsonl').write_text(
                ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in citation_rows))
        if context_rows:
            (args.output / 'context_results.jsonl').write_text(
                ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in context_rows))
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
