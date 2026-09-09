"""Relevance and latency comparisons across independently callable retrievers.

This complements evaluation.py's exact/ANN neighbor and evidence diagnostics.
Judgments here are relevance labels, not nearest-neighbor reference rankings.
"""

from collections.abc import Mapping, Sequence
import math


def ranking_metrics(relevance: Mapping[str, int], ranked_ids: Sequence[str], *, k: int) -> dict:
    """Recall over all positive judgments; MRR over the supplied ranking.

    nDCG@K uses gain 2**grade-1 and log2(rank+1) discount. Unjudged IDs have
    grade zero. With no positive judgments all metrics are undefined (None).
    """
    if type(k) is not int or k <= 0:
        raise ValueError('k must be positive.')
    if not isinstance(relevance, Mapping) or any(
        not isinstance(key, str) or not key or type(grade) is not int or not 0 <= grade <= 30
        for key, grade in relevance.items()
    ):
        raise ValueError('Relevance requires nonblank IDs and integer grades in [0, 30].')
    if any(not isinstance(key, str) or not key for key in ranked_ids):
        raise ValueError('Ranked IDs must be nonblank strings.')
    if len(set(ranked_ids)) != len(ranked_ids):
        raise ValueError('Ranked IDs must not contain duplicates.')
    relevant = {key for key, grade in relevance.items() if grade > 0}
    if not relevant:
        return {'recall_at_k': None, 'mrr': None, 'ndcg_at_k': None}
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(ideal, 1))
    dcg = sum((2**relevance.get(key, 0) - 1) / math.log2(rank + 1)
              for rank, key in enumerate(ranked_ids[:k], 1))
    return {'recall_at_k': len(relevant.intersection(ranked_ids[:k])) / len(relevant),
            'mrr': next((1 / rank for rank, key in enumerate(ranked_ids, 1) if key in relevant), 0.),
            'ndcg_at_k': dcg / idcg}


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
    from arkb.retrieval.contracts import SearchResponse, validate_request
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

def main(argv=None) -> int:
    """Run semantic/BM25 baselines against one pinned production snapshot."""
    import argparse
    from contextlib import ExitStack, closing
    from dataclasses import asdict
    import hashlib
    import json
    from pathlib import Path
    import platform
    from arkb.retrieval.bm25 import BM25Retriever
    from arkb.storage import SQLiteStorage

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, default=Path('.obsidian-rag/index.sqlite'))
    parser.add_argument('--vault-id', default='default')
    parser.add_argument('--index-version')
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New JSON file; never overwritten.')
    parser.add_argument('--modes', nargs='+', choices=('semantic', 'bm25', 'hybrid', 'hybrid_reranked'), default=['semantic', 'bm25', 'hybrid'])
    parser.add_argument('--candidate-k', type=int, default=20)
    parser.add_argument('--rerank-candidates', type=int, default=20)
    parser.add_argument('--reranker-cache')
    parser.add_argument('--reranker-model', default='cross-encoder/ms-marco-MiniLM-L6-v2')
    parser.add_argument('--reranker-revision', default='233902d25c440f23af6f7d6e94d2946bac0bee0a')
    parser.add_argument('--reranker-max-length', type=int, default=512)
    parser.add_argument('--rrf-k', type=float, default=60)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--relevance-key', choices=('source', 'source_id', 'chunk_id'), default='source')
    parser.add_argument('--host', default='http://127.0.0.1:11434')
    parser.add_argument('--qdrant-url')
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--tokenizer-cache', type=Path)
    parser.add_argument('--bm25-k1', type=float, default=1.2)
    parser.add_argument('--bm25-b', type=float, default=.75)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('--output already exists')
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout must be positive and finite')
    raw = args.cases.read_bytes()
    cases = [json.loads(line) for line in raw.splitlines() if line.strip()]
    with ExitStack() as resources:
        storage = resources.enter_context(SQLiteStorage(args.db, read_only=True))
        manifest = storage.get_manifest(args.index_version) if args.index_version else storage.active_manifest(args.vault_id)
        if manifest is None or manifest.status != 'ready' or manifest.vault_id != args.vault_id:
            raise ValueError('Evaluation requires a ready snapshot in the requested vault.')
        records = storage.snapshot_records(manifest.index_version)
        known = {getattr(r.chunk, 'source') if args.relevance_key == 'source' else
                 r.document_id if args.relevance_key == 'source_id' else r.chunk_id for r in records}
        for case in cases:
            if set(case['relevance']) - known:
                raise ValueError('Relevance labels refer to evidence outside the pinned snapshot.')
        retrievers = {}
        if set(args.modes) & {'semantic', 'hybrid', 'hybrid_reranked'}:
            from ollama import Client
            from arkb.embeddings import resolve_embedding_spec
            from arkb.retrieval.semantic import SemanticRetriever
            from arkb.retrieval.ollama import OllamaQueryEmbedder
            from arkb.retrieval.qdrant import QdrantSnapshotIndex
            from arkb.storage import connect_qdrant, require_qdrant_backend
            from arkb.tokenization import load_tokenizer
            metadata = storage.build_metadata(manifest.index_version)['backend']
            require_qdrant_backend(metadata)
            qclient = resources.enter_context(closing(connect_qdrant(args.qdrant_url or metadata['url'], args.timeout)))
            client = resources.enter_context(Client(host=args.host, timeout=args.timeout, trust_env=False))
            tokenizer = load_tokenizer(cache_dir=args.tokenizer_cache, local_files_only=args.offline)
            spec = resolve_embedding_spec(client, manifest.embedding_spec.model,
                                          context_length=metadata['input']['max_tokens'])
            index = QdrantSnapshotIndex(storage, qclient, vault_id=args.vault_id,
                                       index_version=manifest.index_version, exact=True)
            embedder = OllamaQueryEmbedder(client=client, spec=spec, tokenizer=tokenizer,
                tokenizer_identity=index.inputs['tokenizer'], max_input_tokens=index.inputs['max_tokens'],
                query_instruction=manifest.query_instruction)
            retrievers['semantic'] = SemanticRetriever(embedder, index)
        if set(args.modes) & {'bm25', 'hybrid', 'hybrid_reranked'}:
            retrievers['bm25'] = BM25Retriever(records, index_id=manifest.index_version,
                                                k1=args.bm25_k1, b=args.bm25_b)
        if set(args.modes) & {'hybrid', 'hybrid_reranked'}:
            from arkb.retrieval.hybrid import HybridRetriever
            retrievers['hybrid'] = HybridRetriever(retrievers['bm25'], retrievers['semantic'],
                                                  candidate_k=args.candidate_k, rrf_k=args.rrf_k)
        if 'hybrid_reranked' in args.modes:
            from arkb.retrieval.cross_encoder import CrossEncoderScorer
            from arkb.retrieval.reranker import Reranker, RerankedRetriever
            if args.rerank_candidates > args.candidate_k:
                raise ValueError('rerank-candidates cannot exceed hybrid candidate-k.')
            reranker = Reranker(CrossEncoderScorer(model=args.reranker_model, revision=args.reranker_revision,
                max_length=args.reranker_max_length, cache_folder=args.reranker_cache, local_files_only=args.offline))
            retrievers['hybrid_reranked'] = RerankedRetriever(retrievers['hybrid'], reranker,
                                                           candidate_k=args.rerank_candidates)
        retrievers = {name: retrievers[name] for name in dict.fromkeys(args.modes)}
        report = evaluate_retrievers(retrievers, cases, top_k=args.top_k, relevance_key=args.relevance_key)
        package = Path(__file__).parent
        report['run'] = {'manifest': asdict(manifest), 'cases': cases,
                         'cases_sha256': hashlib.sha256(raw).hexdigest(),
                         'python': platform.python_version(),
                         'configuration': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                         'source_hashes': {p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in sorted(package.rglob('*.py'))}}
    with args.output.open('x') as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps(report['summary'], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
