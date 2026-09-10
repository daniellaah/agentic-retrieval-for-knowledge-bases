"""Relevance baselines and exact/ANN comparisons on pinned knowledge snapshots."""

import argparse
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
from time import perf_counter

import numpy as np

from arkb.config import DEFAULT_DB, RuntimeConfig, RetrievalConfig, DEFAULT_GENERATION_MODEL
from arkb.knowledge.embeddings import validate_vectors
from arkb.runtime import Runtime
from arkb.evaluation.datasets import load_cases, source_hashes, corpus_manifest
from arkb.evaluation.metrics import (
    ranking_metrics, recall_at_k, evidence_statistics, summarize_citations,
)

def evaluate_retrievers(retrievers, cases: Sequence[dict], *, top_k: int = 10,
                        relevance_key: str = 'source') -> dict:
    """Compare fixed configurations on identical questions and judgments.

    For document labels, repeated document hits collapse in first-occurrence
    order after retrieval; raw chunk rankings remain in the report. MRR is thus
    bounded by the retrieved top_k chunks. Latency includes query embedding and
    all retrieval stages, excluding construction/model loading. Mode order rotates.
    """
    from dataclasses import asdict
    from time import perf_counter
    from arkb.retrieval.models import SearchResponse, validate_request
    if relevance_key not in ('source', 'source_id', 'chunk_id'):
        raise ValueError('relevance_key must be source, source_id, or chunk_id.')
    if not retrievers or not cases or len({case['id'] for case in cases}) != len(cases):
        raise ValueError('Evaluation requires retrievers and cases with unique IDs.')
    for case in cases:
        validate_request(case['question'], top_k, case.get('filters'))
        ranking_metrics(case['relevance'], [], k=top_k)
    names = list(retrievers)
    rows = []
    for i, case in enumerate(cases):
        row = {'id': case['id'], 'question': case['question'], 'modes': {}}
        for name in names[i % len(names):] + names[:i % len(names)]:
            started = perf_counter()
            response = retrievers[name].search(case['question'], top_k=top_k, filters=case.get('filters'))
            latency = (perf_counter() - started) * 1000
            if (not isinstance(response, SearchResponse) or response.query != case['question']
                    or len(response.results) > top_k
                    or len({hit.identity for hit in response.results}) != len(response.results)):
                raise ValueError('Retriever returned an invalid ranking for evaluation.')
            ids = list(dict.fromkeys(getattr(hit, relevance_key) for hit in response.results))
            metrics = ranking_metrics(case['relevance'], ids, k=top_k)
            row['modes'][name] = {'response': asdict(response), 'metrics': metrics, 'latency_ms': latency}
        rows.append(row)
    summary = {}
    for name in names:
        entries = [row['modes'][name] for row in rows]
        summary[name] = {'mean_latency_ms': sum(r['latency_ms'] for r in entries) / len(entries)}
        for metric in ('recall_at_k', 'mrr', 'ndcg_at_k'):
            values = [r['metrics'][metric] for r in entries if r['metrics'][metric] is not None]
            summary[name][metric] = sum(values) / len(values) if values else None
            summary[name][metric + '_defined_cases'] = len(values)
    return {'settings': {'top_k': top_k, 'relevance_key': relevance_key,
                         'mrr_depth': top_k, 'latency': 'query embedding and search; excludes setup'},
            'summary': summary, 'results': rows}


def evaluate_reranker(reranker, query, candidates, relevance, *, top_k=10,
                      relevance_key='source') -> dict:
    """Score one frozen candidate set, retaining input/output and rank movement."""
    from dataclasses import asdict
    from time import perf_counter
    if relevance_key not in ('source', 'source_id', 'chunk_id'):
        raise ValueError('Invalid relevance key.')
    candidates = tuple(candidates)
    def metrics(hits):
        ids = list(dict.fromkeys(getattr(hit, relevance_key) for hit in hits))
        return ranking_metrics(relevance, ids, k=top_k)
    before = metrics(candidates[:top_k])
    started = perf_counter()
    results = reranker.rerank(query, candidates, top_k=top_k)
    latency = (perf_counter() - started) * 1000
    ranks = {hit.identity: i for i, hit in enumerate(candidates, 1)}
    return {'query': query, 'before': before, 'after': metrics(results), 'latency_ms': latency,
            'candidates': [asdict(hit) for hit in candidates], 'results': [asdict(hit) for hit in results],
            'rank_changes': [{'identity': list(hit.identity), 'before': ranks[hit.identity], 'after': i}
                             for i, hit in enumerate(results, 1)]}


def compare_retrieval(records, vectors, query_vectors, cases: list[dict], *, spec,
                      search, top_k: int = 2) -> dict:
    """Compare identical vectors/queries; timings cover search only, including backend I/O.

    Compare Qdrant exact and ANN modes. The search callable receives a query
    vector, top_k and exact; callers bind the
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
    if not callable(search):
        raise ValueError('Evaluation requires a Qdrant search callable.')
    modes = {'qdrant_exact': True, 'qdrant_ann': False}
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
            started = perf_counter()
            hits = search(query, top_k=top_k, exact=modes[name])
            elapsed = (perf_counter() - started) * 1000
            if len(hits) > top_k or any(h.chunk_id not in by_id for h in hits):
                raise ValueError('Evaluation backend returned unknown or excess hits.')
            chunks = [by_id[h.chunk_id].chunk for h in hits]
            row['modes'][name] = {'hits': [asdict(h) for h in hits], 'search_ms': elapsed,
                                  **evidence_statistics(chunks, case)}
        expected = row['modes']['qdrant_exact']['hits']
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


def baseline_main(argv=None) -> int:
    """Run explicit retrieval configurations against one pinned production snapshot."""
    import argparse
    from dataclasses import asdict
    import hashlib
    import json
    from pathlib import Path
    import platform
    from arkb.knowledge.sqlite import SQLiteStorage

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, default=DEFAULT_DB)
    parser.add_argument('--vault-id', default='default')
    parser.add_argument('--index-version')
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New JSON file; never overwritten.')
    parser.add_argument('--modes', nargs='+', choices=('semantic', 'bm25', 'hybrid', 'hybrid_reranked'), default=['semantic', 'bm25', 'hybrid'])
    parser.add_argument('--candidate-k', type=int, default=20)
    parser.add_argument('--rerank-candidates', type=int, default=20)
    parser.add_argument('--reranker-cache')
    parser.add_argument('--reranker-max-length', type=int, default=512)
    parser.add_argument('--rrf-k', type=float, default=60)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--relevance-key', choices=('source', 'source_id', 'chunk_id'), default='source')
    parser.add_argument('--host', default=RuntimeConfig.host)
    parser.add_argument('--qdrant-url')
    parser.add_argument('--timeout', type=float, default=RuntimeConfig.timeout)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--tokenizer-cache', type=Path)
    parser.add_argument('--bm25-k1', type=float, default=1.2)
    parser.add_argument('--bm25-b', type=float, default=.75)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('--output already exists')
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout must be positive and finite')
    raw, cases = load_cases(args.cases)
    from arkb.retrieval.models import validate_request
    from arkb.retrieval.hybrid import rrf
    if not cases or len({case['id'] for case in cases}) != len(cases):
        parser.error('Cases must be nonempty with unique IDs')
    for case in cases:
        validate_request(case['question'], args.top_k, case.get('filters'))
        ranking_metrics(case['relevance'], [], k=args.top_k)
    if set(args.modes) & {'hybrid', 'hybrid_reranked'}:
        validate_request('configuration', args.candidate_k, None)
        rrf([], k=args.rrf_k)
        if args.top_k > args.candidate_k:
            parser.error('top-k cannot exceed hybrid candidate-k')
    if 'hybrid_reranked' in args.modes and not args.top_k <= args.rerank_candidates <= args.candidate_k:
        parser.error('Require top-k <= rerank-candidates <= candidate-k')
    runtime_config = RuntimeConfig(host=args.host, timeout=args.timeout, offline=args.offline,
                                   tokenizer_cache=args.tokenizer_cache, qdrant_url=args.qdrant_url)
    retrieval_config = RetrievalConfig(candidate_k=args.candidate_k, rrf_k=args.rrf_k,
        rerank_candidates=args.rerank_candidates, reranker_max_length=args.reranker_max_length,
        reranker_cache=args.reranker_cache, bm25_k1=args.bm25_k1, bm25_b=args.bm25_b)
    with Runtime(runtime_config) as runtime, SQLiteStorage(args.db, read_only=True) as storage:
        manifest = storage.get_manifest(args.index_version) if args.index_version else storage.active_manifest(args.vault_id)
        if manifest is None or manifest.status != 'ready' or manifest.vault_id != args.vault_id:
            raise ValueError('Evaluation requires a ready snapshot in the requested vault.')
        records = storage.snapshot_records(manifest.index_version)
        known = {getattr(r.chunk, 'source') if args.relevance_key == 'source' else
                 r.document_id if args.relevance_key == 'source_id' else r.chunk_id for r in records}
        for case in cases:
            if set(case['relevance']) - known:
                raise ValueError('Relevance labels refer to evidence outside the pinned snapshot.')
        engine = runtime.retrieval_engine(storage, manifest, modes=args.modes,
            settings=retrieval_config, exact=True, records=records)
        retrievers = {'semantic': engine.semantic, 'bm25': engine.bm25}
        if set(args.modes) & {'hybrid', 'hybrid_reranked'}:
            from arkb.retrieval.hybrid import HybridRetriever
            retrievers['hybrid'] = HybridRetriever(engine.bm25, engine.semantic,
                                                  candidate_k=args.candidate_k, rrf_k=args.rrf_k)
        if 'hybrid_reranked' in args.modes:
            from arkb.retrieval.rerank import RerankedRetriever
            retrievers['hybrid_reranked'] = RerankedRetriever(retrievers['hybrid'], engine.reranker,
                                                           candidate_k=args.rerank_candidates)
        retrievers = {name: retrievers[name] for name in dict.fromkeys(args.modes)}
        report = evaluate_retrievers(retrievers, cases, top_k=args.top_k, relevance_key=args.relevance_key)
        package = Path(__file__).resolve().parent.parent
        report['run'] = {'manifest': asdict(manifest), 'cases': cases,
                         'cases_sha256': hashlib.sha256(raw).hexdigest(),
                         'python': platform.python_version(),
                         'configuration': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                         'source_hashes': source_hashes(package)}
    with args.output.open('x') as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps(report['summary'], indent=2))
    return 0


def ann_main(argv=None) -> int:
    """Evaluate an existing snapshot and save a new, non-overwriting artifact folder."""
    from importlib.metadata import version
    from arkb.knowledge.qdrant import search_qdrant
    from arkb.knowledge.embeddings import resolve_embedding_spec
    from arkb.knowledge.embeddings import prepare_query, validate_input_tokens
    from arkb.knowledge.embeddings import embed_texts
    from arkb.knowledge.qdrant import QdrantIndex
    from arkb.knowledge.embeddings import tokenizer_fingerprint
    from arkb.knowledge.models import require_qdrant_backend
    from arkb.knowledge.sqlite import SQLiteStorage

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vault-id', default='default')
    parser.add_argument('--top-k', type=int, default=2)
    parser.add_argument('--host', default=RuntimeConfig.host)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--tokenizer-cache', type=Path)
    parser.add_argument('--qdrant-url')
    parser.add_argument('--context', action='store_true', help='Evaluate packed context on the same Qdrant exact hits.')
    parser.add_argument('--citations', action='store_true', help='Generate and evaluate structured citations on the same Qdrant exact hits.')
    parser.add_argument('--generation-model', default=DEFAULT_GENERATION_MODEL)
    parser.add_argument('--context-window', type=int, default=8192)
    parser.add_argument('--max-output-tokens', type=int, default=1024)
    parser.add_argument('--context-safety-margin', type=int, default=128)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('--output must be a new directory; old evaluation artifacts are preserved')
    context_config = None
    if args.context or args.citations:
        from arkb.generation.models import ContextConfig
        from arkb.generation.generate import load_generation_counter
        try:
            context_config = ContextConfig(args.context_window, args.max_output_tokens, args.context_safety_margin)
        except ValueError as error:
            parser.error(str(error))
    raw_cases, cases = load_cases(args.cases)
    runtime_config = RuntimeConfig(host=args.host, offline=args.offline,
        tokenizer_cache=args.tokenizer_cache, qdrant_url=args.qdrant_url, qdrant_timeout=30.0)
    with Runtime(runtime_config) as runtime, SQLiteStorage(args.db, read_only=True) as storage:
        manifest = storage.active_manifest(args.vault_id)
        if manifest is None:
            raise ValueError('No active snapshot to evaluate.')
        metadata = storage.build_metadata(manifest.index_version)
        require_qdrant_backend(metadata['backend'])
        _, records, vectors = storage.load_snapshot(manifest.index_version)
        tokenizer = runtime.tokenizer()
        if tokenizer_fingerprint(tokenizer) != metadata['backend']['input']['tokenizer']:
            raise ValueError('Evaluation tokenizer differs from the snapshot.')
        ollama = runtime.model_client()
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
        client = runtime.qdrant_client(args.qdrant_url or metadata['backend']['url'])
        store = QdrantIndex(client, metadata['backend']['collection'], spec, vault_id=args.vault_id)
        store.verify_snapshot(records, vectors)
        info = store.check_configuration()
        server_info = {'version': client.info().version, 'indexed_vectors': info.indexed_vectors_count,
                       'hnsw_config': info.config.hnsw_config.model_dump(mode='json')}
        search = partial(search_qdrant, client, store.collection, spec=spec, vault_id=args.vault_id)
        report = compare_retrieval(records, vectors, queries, cases, spec=spec,
                                   search=search, top_k=args.top_k)
        context_rows, citation_rows = [], []
        if args.context or args.citations:
            from arkb.retrieval.semantic import snapshot_result
            from arkb.evaluation.generation import evaluate_context, evaluate_citation_case
            counter = load_generation_counter(client=ollama, model=args.generation_model,
                                              cache_dir=args.tokenizer_cache, local_files_only=args.offline)
            by_id = {record.chunk_id: record for record in records}
            for case, row in zip(cases, report['results']):
                hits = [snapshot_result(by_id[h['chunk_id']], h['score'], manifest.index_version)
                        for h in row['modes']['qdrant_exact']['hits']]
                if args.context:
                    context_rows.append({'id': case['id'], 'question': case['question'],
                                         'result': evaluate_context(case['question'], hits, case, config=context_config, counter=counter)})
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
            entries = [row['result']['metrics'] for row in context_rows]
            for key in entries[0]:
                values = [entry[key] for entry in entries if entry[key] is not None]
                report['context']['summary'][key] = float(np.mean(values)) if values else None
        package_dir = Path(__file__).resolve().parent.parent
        repo = package_dir.parents[1]
        git = subprocess.run(['git', '-C', str(repo), 'rev-parse', 'HEAD'], capture_output=True, text=True)
        run = {'created_at': datetime.now(timezone.utc).isoformat(), 'manifest': asdict(manifest),
               'build_metadata': metadata, 'cases_sha256': hashlib.sha256(raw_cases).hexdigest(),
               'python': platform.python_version(), 'numpy': np.__version__, 'qdrant_client': version('qdrant-client'),
               'qdrant_server': server_info, 'source_commit': git.stdout.strip() if git.returncode == 0 else None,
               'source_hashes': source_hashes(package_dir),
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
        corpus = corpus_manifest(records)
        (args.output / 'corpus_manifest.json').write_text(json.dumps(corpus, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def main(argv=None) -> int:
    """Select a relevance baseline or exact/ANN snapshot experiment."""
    import sys
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', choices=('baseline', 'ann'))
    selected = parser.parse_args(args[:1])
    return {'baseline': baseline_main, 'ann': ann_main}[selected.experiment](args[1:])


if __name__ == '__main__':
    raise SystemExit(main())
