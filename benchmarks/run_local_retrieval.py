"""Build an isolated example snapshot and compare real models with Qdrant Local.

Uses the same public retrievers as the server runner. Qdrant Local is exact;
these timings do not measure server/ANN behavior. No existing index is changed.
"""

import argparse
from contextlib import closing
from dataclasses import asdict
from importlib.metadata import version
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter
import warnings

from ollama import Client
from qdrant_client import QdrantClient

from arkb.knowledge.embeddings import resolve_embedding_spec
from arkb.knowledge.models import QdrantConfig
from arkb.knowledge.indexing import build_index
from arkb.knowledge.documents import scan_notes
from arkb.retrieval import BM25Retriever, HybridRetriever, RerankedRetriever, Reranker
from arkb.retrieval.qwen_rerank import QwenRerankerScorer
from arkb.runtime import SnapshotSemanticRetriever
from arkb.evaluation.retrieval import evaluate_retrievers, evaluate_reranker
from arkb.knowledge.sqlite import SQLiteStorage
from arkb.knowledge.embeddings import load_tokenizer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New directory for snapshot and reports.')
    parser.add_argument('--notes-dir', type=Path, default=Path('example_notes'))
    parser.add_argument('--cases', type=Path, default=Path('benchmarks/retrieval-cases.jsonl'))
    parser.add_argument('--modes', nargs='+', choices=('semantic', 'bm25', 'hybrid', 'hybrid_reranked'),
                        default=['semantic', 'bm25', 'hybrid', 'hybrid_reranked'])
    parser.add_argument('--host', default='http://127.0.0.1:11434')
    parser.add_argument('--tokenizer-cache', type=Path, default=Path('.uv-cache/tokenizers'))
    parser.add_argument('--reranker-cache', default='.obsidian-rag/models')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--candidate-k', type=int, default=20)
    args = parser.parse_args(argv)
    if not 0 < args.top_k <= args.candidate_k:
        parser.error('Require 0 < top-k <= candidate-k')
    raw = args.cases.read_bytes()
    cases = [json.loads(line) for line in raw.splitlines() if line.strip()]
    notes = scan_notes(args.notes_dir)
    for case in cases:
        if set(case['relevance']) - {note.source for note in notes}:
            parser.error('Case labels refer to sources outside this corpus')
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = load_tokenizer(cache_dir=args.tokenizer_cache, local_files_only=args.offline)
    with warnings.catch_warnings(), Client(host=args.host, timeout=180, trust_env=False) as client, \
            closing(QdrantClient(path=str(args.output / 'qdrant'))) as qclient, \
            SQLiteStorage(args.output / 'index.sqlite') as storage:
        warnings.filterwarnings('ignore', message='Payload indexes have no effect in the local Qdrant.*')
        warnings.filterwarnings('ignore', message='Local mode performs exact.*')
        spec = resolve_embedding_spec(client, 'qwen3-embedding:0.6b', context_length=8192)
        build = build_index(storage, notes, spec=spec, vault_id='example-benchmark', client=client,
            tokenizer=tokenizer, max_input_tokens=8192, chunking='recursive', chunk_size=512,
            chunk_overlap=64, qdrant_client=qclient, qdrant_config=QdrantConfig())
        semantic = SnapshotSemanticRetriever(storage, vault_id='example-benchmark', spec=spec,
            tokenizer=tokenizer, client=client, exact=True, qdrant_client=qclient)
        bm25 = BM25Retriever.from_snapshot(storage, vault_id='example-benchmark')
        hybrid = HybridRetriever(bm25, semantic, candidate_k=args.candidate_k)
        retrievers = {'semantic': semantic, 'bm25': bm25, 'hybrid': hybrid}
        setup_started = perf_counter()
        reranker = None
        if 'hybrid_reranked' in args.modes:
            reranker = Reranker(QwenRerankerScorer(cache_folder=args.reranker_cache, local_files_only=args.offline))
            retrievers['hybrid_reranked'] = RerankedRetriever(hybrid, reranker, candidate_k=args.candidate_k)
        setup_ms = (perf_counter() - setup_started) * 1000
        report = evaluate_retrievers({name: retrievers[name] for name in dict.fromkeys(args.modes)}, cases,
                                    top_k=args.top_k)
        if reranker:
            report['frozen_reranker'] = [evaluate_reranker(reranker, case['question'],
                hybrid.search(case['question'], top_k=args.candidate_k).results,
                case['relevance'], top_k=args.top_k) for case in cases]
        import arkb
        package = Path(arkb.__file__).parent
        report['run'] = {'build': asdict(build), 'backend': 'Qdrant Local exact', 'cases': cases,
            'cases_sha256': hashlib.sha256(raw).hexdigest(), 'platform': platform.platform(),
            'packages': {name: version(name) for name in ('qdrant-client', 'ollama', 'numpy')},
            'reranker': reranker.scorer.identity if reranker else None, 'reranker_setup_ms': setup_ms,
            'model_packages': {name: version(name) for name in ('transformers', 'torch')} if reranker else {},
            'candidate_k': args.candidate_k, 'rrf_k': 60, 'bm25': {'k1': 1.2, 'b': .75},
            'source_hashes': {p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(package.rglob('*.py'))},
            'corpus': [{'source': r.chunk.source, 'chunk_id': r.chunk_id, 'source_id': r.document_id,
                        'document_revision': r.document_revision} for r in storage.snapshot_records(bm25.index_id)]}
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report['summary'], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
