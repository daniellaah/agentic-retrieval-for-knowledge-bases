"""Answer questions about Markdown notes from the command line."""

import argparse
from collections.abc import Sequence
from dataclasses import asdict
import json
import math
from pathlib import Path
import sqlite3
import sys

from httpx import HTTPError
from ollama import ResponseError

from arkb.config import RuntimeConfig, RetrievalConfig, DEFAULT_EMBEDDING_MODEL, DEFAULT_GENERATION_MODEL
from arkb.runtime import Runtime
from arkb.knowledge.embeddings import resolve_embedding_spec
from arkb.knowledge.models import QdrantConfig
from arkb.generation.models import ContextConfig
from arkb.generation.context import build_context
from arkb.generation.generate import generate_cited_answer, load_generation_counter


def _add_context_arguments(parser):
    parser.add_argument('--context-window', type=int, default=8192,
                        help='Generation window in tokens; also sent as Ollama num_ctx.')
    parser.add_argument('--max-output-tokens', type=int, default=1024,
                        help='Reserved output tokens; also sent as Ollama num_predict.')
    parser.add_argument('--context-safety-margin', type=int, default=128)
    parser.add_argument('--show-context', action='store_true',
                        help='Print final messages, provenance and budget diagnostics without generation.')
    parser.add_argument('--citation-mode', choices=('structured', 'quoted'), default='structured',
                        help='Validated citations (default) or exact quoted citations.')
    parser.add_argument('--answer-json', action='store_true',
                        help='Print structured answer, cited sources, raw response and validation.')

def _context_config(args):
    return ContextConfig(args.context_window, args.max_output_tokens, args.context_safety_margin)

def _validate_context_arguments(parser, args):
    try:
        _context_config(args)
    except ValueError as error:
        parser.error(str(error))
    if sum((args.json, args.show_context, args.answer_json)) > 1:
        parser.error('--json, --show-context and --answer-json are separate output modes')

def _build_cli_context(args, results, client):
    counter = load_generation_counter(client=client, model=args.generation_model,
                                      cache_dir=args.tokenizer_cache, local_files_only=args.offline)
    return build_context(args.question, results, config=_context_config(args), counter=counter,
                          citation_mode=args.citation_mode)


def _answer_output(args, context, client):
    result = generate_cited_answer(context, client=client)
    return json.dumps(result.to_dict(), ensure_ascii=False) if args.answer_json else result.text


def _runtime_config(args):
    return RuntimeConfig(host=args.host, timeout=args.timeout, tokenizer_cache=args.tokenizer_cache,
                         offline=args.offline, embedding_model=args.embedding_model,
                         qdrant_url=args.qdrant_url)


def _parser():
    from arkb.knowledge.embeddings import DEFAULT_QUERY_INSTRUCTION
    parser = argparse.ArgumentParser(prog='arkb')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('index', 'query', 'status'):
        command = commands.add_parser(name)
        command.add_argument('--db', type=Path, default=Path('.obsidian-rag/index.sqlite'))
        command.add_argument('--vault-id', default='default')
        if name == 'status':
            continue
        command.add_argument('--host', default=RuntimeConfig.host)
        command.add_argument('--timeout', type=float, default=RuntimeConfig.timeout)
        command.add_argument('--tokenizer-cache', type=Path)
        command.add_argument('--offline', action='store_true')
        command.add_argument('--embedding-model', default=DEFAULT_EMBEDDING_MODEL if name == 'index' else None)
        if name == 'index':
            command.add_argument('--notes-dir', type=Path, default=Path('example_notes'))
            command.add_argument('--qdrant-url', default=QdrantConfig.url)
            command.add_argument('--hnsw-m', type=int, default=QdrantConfig.hnsw_m)
            command.add_argument('--ef-construct', type=int, default=QdrantConfig.ef_construct)
            command.add_argument('--indexing-threshold', type=int, default=QdrantConfig.indexing_threshold)
            command.add_argument('--full-scan-threshold', type=int, default=QdrantConfig.full_scan_threshold)
            command.add_argument('--index-timeout', type=float, default=QdrantConfig.index_timeout)
            command.add_argument('--require-hnsw', action='store_true', default=QdrantConfig.require_hnsw)
            command.add_argument('--force', action='store_true', help='Rebuild even when unchanged; reuse compatible vectors.')
            command.add_argument('--chunking', choices=('none', 'recursive'), default='recursive')
            command.add_argument('--chunk-size', type=int, default=512)
            command.add_argument('--chunk-overlap', type=int, default=64)
            command.add_argument('--context-length', type=int, default=8192,
                                 help='Per-input budget; also sent as Ollama num_ctx.')
            command.add_argument('--batch-size', type=int, default=32)
            command.add_argument('--max-batch-tokens', type=int)
            command.add_argument('--max-retries', type=int, default=2)
            command.add_argument('--query-instruction', default=DEFAULT_QUERY_INSTRUCTION)
        else:
            command.add_argument('--qdrant-url', help='Override the saved Qdrant endpoint, e.g. after restoring a server.')
            command.add_argument('question')
            command.add_argument('--top-k', type=int, default=2)
            command.add_argument('--source')
            command.add_argument('--mode', choices=('semantic', 'bm25', 'lexical', 'hybrid'), default='semantic')
            command.add_argument('--candidate-k', type=int, default=RetrievalConfig.candidate_k, help='Candidates per retriever for hybrid.')
            command.add_argument('--rrf-k', type=float, default=RetrievalConfig.rrf_k)
            command.add_argument('--rerank', action='store_true')
            command.add_argument('--rerank-candidates', type=int, default=RetrievalConfig.rerank_candidates)
            command.add_argument('--reranker-cache')
            command.add_argument('--reranker-model', default=RetrievalConfig.reranker_model)
            command.add_argument('--reranker-revision', default=RetrievalConfig.reranker_revision)
            command.add_argument('--reranker-max-length', type=int, default=RetrievalConfig.reranker_max_length)
            command.add_argument('--exact', action='store_true')
            command.add_argument('--json', action='store_true', help='Print retrieval results without generation.')
            command.add_argument('--generation-model', default=DEFAULT_GENERATION_MODEL)
            _add_context_arguments(command)
    return parser

def main(argv: Sequence[str] | None = None) -> int:
    from arkb.knowledge.indexing import build_index
    from arkb.knowledge.documents import scan_notes
    from arkb.knowledge.sqlite import SQLiteStorage
    from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

    parser = _parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.vault_id.strip():
        parser.error('--vault-id must not be blank')
    if args.command != 'status' and (not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error('--timeout must be positive and finite')
    if args.command == 'index':
        try:
            qdrant_config = QdrantConfig(url=args.qdrant_url, hnsw_m=args.hnsw_m,
                                         ef_construct=args.ef_construct, indexing_threshold=args.indexing_threshold,
                                         full_scan_threshold=args.full_scan_threshold, index_timeout=args.index_timeout,
                                         require_hnsw=args.require_hnsw)
        except ValueError as error:
            parser.error(str(error))
        if args.context_length <= 0 or args.batch_size <= 0 or not 0 <= args.max_retries <= 8:
            parser.error('context/batch sizes must be positive and retries must be between 0 and 8')
        if args.max_batch_tokens is not None and args.max_batch_tokens <= 0:
            parser.error('--max-batch-tokens must be positive')
        if args.chunking == 'recursive' and (args.chunk_size <= 0 or not 0 <= args.chunk_overlap < args.chunk_size):
            parser.error('chunk size must be positive and overlap must be in [0, chunk size)')
    if args.command == 'query':
        _validate_context_arguments(parser, args)
        if (args.candidate_k <= 0 or args.rerank_candidates <= 0 or args.reranker_max_length <= 0
                or not math.isfinite(args.rrf_k) or args.rrf_k < 0):
            parser.error('candidate/token limits must be positive; RRF k must be finite and nonnegative')
        if args.mode == 'hybrid' and max(args.top_k, args.rerank_candidates if args.rerank else 0) > args.candidate_k:
            parser.error('hybrid requires top-k and rerank-candidates <= candidate-k')
        if args.rerank and args.top_k > args.rerank_candidates:
            parser.error('reranking requires top-k <= rerank-candidates')
    if args.command == 'query' and (not args.question.strip() or args.top_k <= 0):
        parser.error('question must not be blank and --top-k must be positive')
    try:
        if args.command == 'status':
            with SQLiteStorage(args.db, read_only=True) as storage:
                active = storage.active_manifest(args.vault_id)
                print(json.dumps({'vault_id': args.vault_id,
                                  'active_version': active.index_version if active else None,
                                  'builds': [asdict(m) for m in storage.list_builds(args.vault_id)]}, ensure_ascii=False))
            return 0
        if args.command == 'index':
            # Finish the source scan before opening or changing index state.
            notes = scan_notes(args.notes_dir)
            with Runtime(_runtime_config(args)) as runtime:
                tokenizer = runtime.tokenizer()
                client = runtime.model_client()
                spec = resolve_embedding_spec(client, args.embedding_model, context_length=args.context_length)
                with SQLiteStorage(args.db) as storage:
                    qclient = runtime.qdrant_client(args.qdrant_url)
                    report = build_index(storage, notes, spec=spec, vault_id=args.vault_id, client=client,
                                         tokenizer=tokenizer, max_input_tokens=args.context_length,
                                         chunking=args.chunking, chunk_size=args.chunk_size,
                                         chunk_overlap=args.chunk_overlap, batch_size=args.batch_size,
                                         max_batch_tokens=args.max_batch_tokens, max_retries=args.max_retries,
                                         query_instruction=args.query_instruction, force=args.force,
                                         source_scope=str(args.notes_dir.resolve()), qdrant_config=qdrant_config, qdrant_client=qclient)
                    print(json.dumps(asdict(report), ensure_ascii=False))
            return 0
        with Runtime(_runtime_config(args)) as runtime, SQLiteStorage(args.db, read_only=True) as storage:
            manifest = storage.active_manifest(args.vault_id)
            if manifest is None:
                raise ValueError('No published index; run the index command first.')
            settings = RetrievalConfig(candidate_k=args.candidate_k, rrf_k=args.rrf_k,
                rerank_candidates=args.rerank_candidates, reranker_model=args.reranker_model,
                reranker_revision=args.reranker_revision, reranker_max_length=args.reranker_max_length,
                reranker_cache=args.reranker_cache)
            engine = runtime.retrieval_engine(storage, manifest, modes=(args.mode,),
                rerank=args.rerank, settings=settings, exact=args.exact)
            response = engine.search(args.question, mode=args.mode, top_k=args.top_k,
                                     filters={'source': args.source} if args.source is not None else None,
                                     rerank=args.rerank)
            if args.json:
                print(json.dumps({'index_version': manifest.index_version, 'question': args.question,
                                  'results': [asdict(r) for r in response.results],
                                  'method': response.method}, ensure_ascii=False))
            else:
                client = runtime.model_client()
                context = _build_cli_context(args, response.results, client)
                if args.show_context:
                    print(json.dumps(context.to_dict(), ensure_ascii=False))
                else:
                    print(_answer_output(args, context, client))
        return 0
    except (OSError, ValueError, sqlite3.Error, ResponseError, HTTPError,
            ResponseHandlingException, UnexpectedResponse) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
