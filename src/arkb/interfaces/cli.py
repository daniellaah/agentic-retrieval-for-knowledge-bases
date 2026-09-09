"""Parse arguments, call Runtime capabilities, and format their results."""

import argparse
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path
import sqlite3
import subprocess
import sys

from httpx import HTTPError
from ollama import ResponseError
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from arkb.config import (
    DEFAULT_DB, DEFAULT_NOTES_DIR, DEFAULT_EMBEDDING_MODEL, DEFAULT_GENERATION_MODEL,
    DEFAULT_AGENT_THINK, DEFAULT_RETRIEVAL_MODE, RuntimeConfig, RetrievalConfig,
)
from arkb.knowledge.embeddings import DEFAULT_QUERY_INSTRUCTION
from arkb.knowledge.models import QdrantConfig
from arkb.runtime import Runtime


def _nonblank(value):
    if not value.strip():
        raise argparse.ArgumentTypeError('must not be blank')
    return value


def _positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def _positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('must be positive and finite')
    return number


def _add_services(command, *, indexing=False):
    command.add_argument('--host', default=RuntimeConfig.host)
    command.add_argument('--timeout', type=_positive_float, default=RuntimeConfig.timeout)
    command.add_argument('--tokenizer-cache', type=Path)
    command.add_argument('--offline', action='store_true')
    command.add_argument('--embedding-model', type=_nonblank,
                         default=DEFAULT_EMBEDDING_MODEL if indexing else None)
    command.add_argument('--qdrant-url', default=QdrantConfig.url if indexing else None,
                         help='Qdrant endpoint; search/ask otherwise use the saved endpoint.')


def _add_index_arguments(command):
    command.add_argument('--notes-dir', type=Path, default=DEFAULT_NOTES_DIR)
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
    command.add_argument('--context-length', type=_positive_int, default=8192,
                         help='Per-input embedding budget; also sent as Ollama num_ctx.')
    command.add_argument('--batch-size', type=_positive_int, default=32)
    command.add_argument('--max-batch-tokens', type=_positive_int)
    command.add_argument('--max-retries', type=int, choices=range(9), default=2)
    command.add_argument('--query-instruction', default=DEFAULT_QUERY_INSTRUCTION)


def _add_search_arguments(command):
    command.add_argument('--mode', choices=('bm25', 'semantic', 'hybrid'), default=DEFAULT_RETRIEVAL_MODE)
    command.add_argument('--candidate-k', type=_positive_int, default=RetrievalConfig.candidate_k)
    command.add_argument('--rrf-k', type=float, default=RetrievalConfig.rrf_k)
    command.add_argument('--rerank', action='store_true')
    command.add_argument('--rerank-candidates', type=_positive_int, default=RetrievalConfig.rerank_candidates)
    command.add_argument('--reranker-cache')
    command.add_argument('--reranker-model', default=RetrievalConfig.reranker_model)
    command.add_argument('--reranker-revision', default=RetrievalConfig.reranker_revision)
    command.add_argument('--reranker-max-length', type=_positive_int, default=RetrievalConfig.reranker_max_length)
    command.add_argument('--exact', action='store_true', help='Use Qdrant exact vector search.')


def _parser():
    parser = argparse.ArgumentParser(prog='arkb', description='Agentic Retrieval for Knowledge Bases.',
                                     allow_abbrev=False)
    commands = parser.add_subparsers(dest='command', required=True)
    descriptions = {
        'match': 'Find exact lexical occurrences in live notes; no LLM.',
        'search': 'Return ranked results from the index; no answers or agent loop.',
        'ask': 'Let the agent retrieve evidence and respond.',
        'index': 'Build or update the knowledge index.',
        'status': 'Show saved knowledge base and index status.',
    }
    for name, description in descriptions.items():
        command = commands.add_parser(name, help=description, description=description, allow_abbrev=False)
        command.add_argument('--db', type=Path, default=DEFAULT_DB)
        command.add_argument('--vault-id', type=_nonblank, default='default')
        command.add_argument('--json', action='store_true', help='Format the same result as JSON.')
        if name in ('index', 'search', 'ask'):
            _add_services(command, indexing=name == 'index')
        if name in ('match', 'search'):
            command.add_argument('pattern' if name == 'match' else 'query', type=_nonblank)
            command.add_argument('--top-k', type=_positive_int, default=5 if name == 'match' else 2)
            command.add_argument('--source', type=_nonblank, help='Restrict to this exact source filename.')
        if name in ('match', 'ask'):
            command.add_argument('--notes-dir', type=Path,
                                 help='Live notes directory; defaults to the saved scope, then example_notes.')
        if name == 'index':
            _add_index_arguments(command)
        elif name == 'search':
            _add_search_arguments(command)
        elif name == 'ask':
            command.add_argument('query', type=_nonblank)
            command.add_argument('--max-turns', type=_positive_int, default=8,
                                 help='Maximum model turns, including the final response.')
            command.add_argument('--trace', action='store_true', help='Print the available agent trajectory to stderr.')
            command.add_argument('--think', action=argparse.BooleanOptionalAction, default=DEFAULT_AGENT_THINK,
                                 help='Enable model thinking (default); use --no-think to disable it.')
            command.add_argument('--generation-model', type=_nonblank, default=DEFAULT_GENERATION_MODEL)
    return parser


def _validate_arguments(parser, args):
    if args.command == 'index':
        try:
            args.qdrant_config = QdrantConfig(url=args.qdrant_url, hnsw_m=args.hnsw_m,
                ef_construct=args.ef_construct, indexing_threshold=args.indexing_threshold,
                full_scan_threshold=args.full_scan_threshold, index_timeout=args.index_timeout,
                require_hnsw=args.require_hnsw)
        except ValueError as error:
            parser.error(str(error))
        if args.chunking == 'recursive' and (args.chunk_size <= 0 or not 0 <= args.chunk_overlap < args.chunk_size):
            parser.error('chunk size must be positive and overlap must be in [0, chunk size)')
    elif args.command == 'search':
        if not math.isfinite(args.rrf_k) or args.rrf_k < 0:
            parser.error('--rrf-k must be finite and nonnegative')
        if args.mode == 'hybrid' and max(args.top_k, args.rerank_candidates if args.rerank else 0) > args.candidate_k:
            parser.error('hybrid requires top-k and rerank-candidates <= candidate-k')
        if args.rerank and args.top_k > args.rerank_candidates:
            parser.error('reranking requires top-k <= rerank-candidates')


def _runtime_config(args):
    if args.command in ('match', 'status'):
        return RuntimeConfig()
    return RuntimeConfig(host=args.host, timeout=args.timeout, tokenizer_cache=args.tokenizer_cache,
                         offline=args.offline, embedding_model=args.embedding_model,
                         qdrant_url=args.qdrant_url)


def _execute(runtime, args):
    scope = {'db': args.db, 'vault_id': args.vault_id}
    if args.command == 'match':
        return runtime.match(args.pattern, **scope, notes_dir=args.notes_dir,
                             top_k=args.top_k, source=args.source)
    if args.command == 'search':
        settings = RetrievalConfig(candidate_k=args.candidate_k, rrf_k=args.rrf_k,
            rerank_candidates=args.rerank_candidates, reranker_model=args.reranker_model,
            reranker_revision=args.reranker_revision, reranker_max_length=args.reranker_max_length,
            reranker_cache=args.reranker_cache)
        return runtime.search(args.query, **scope, mode=args.mode, top_k=args.top_k, source=args.source,
                              settings=settings, rerank=args.rerank, exact=args.exact)
    if args.command == 'ask':
        try:
            return runtime.ask(args.query, **scope, notes_dir=args.notes_dir,
                               model=args.generation_model, max_turns=args.max_turns, think=args.think)
        except Exception as error:
            partial = getattr(error, 'agent_result', None)
            if args.trace and partial is not None:
                _print_trace(partial)
            raise
    if args.command == 'index':
        return runtime.index(**scope, notes_dir=args.notes_dir, qdrant_config=args.qdrant_config,
            force=args.force, chunking=args.chunking, chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap, context_length=args.context_length, batch_size=args.batch_size,
            max_batch_tokens=args.max_batch_tokens, max_retries=args.max_retries,
            query_instruction=args.query_instruction)
    return runtime.status(**scope)


def _print_retrieval(result):
    print(f'{len(result.results)} result(s) [{result.method}]')
    if result.index_id is not None:
        print(f'Index: {result.index_id}')
    for rank, hit in enumerate(result.results, 1):
        location = f' (body chars {hit.start_char}:{hit.end_char})' if hit.start_char is not None else ''
        score = f' | {hit.score_type}: {hit.score:.6g}' if hit.score is not None else ''
        print(f'\n[{rank}] {hit.source}{location}{score}')
        if hit.metadata.get('title'):
            print(f'Title: {hit.metadata["title"]}')
        print(hit.content)


def _print_trace(result):
    trace = result.trace
    for step, call in enumerate(trace.tool_calls, 1):
        print(f'[{step}] {call.name}', file=sys.stderr)
        for name, value in call.arguments.items():
            print(f'{name}: {json.dumps(value, ensure_ascii=False)}', file=sys.stderr)
        print(file=sys.stderr)
    print(f'[{len(trace.tool_calls) + 1}] {trace.stop_reason}', file=sys.stderr)


def _print_result(args, result):
    if args.command == 'ask' and args.trace:
        _print_trace(result)
    if args.json:
        print(json.dumps(asdict(result) if is_dataclass(result) else result, ensure_ascii=False, allow_nan=False))
    elif args.command in ('match', 'search'):
        _print_retrieval(result)
    elif args.command == 'ask':
        if result.response is not None:
            print(result.response)
    elif args.command == 'index':
        print(f'Index {"reused" if result.reused_index else "published"}: {result.manifest.index_version}')
        print(f'Vault: {result.manifest.vault_id} | Documents: {result.manifest.document_count} | Chunks: {result.manifest.chunk_count}')
        print(f'Embedded: {result.embedded_inputs} | Cached: {result.cached_inputs}')
        print(f'Added: {result.added_documents} | Modified: {result.modified_documents} | Deleted: {result.deleted_documents}')
    else:
        print(f'Vault: {result["vault_id"]}')
        print(f'Notes: {result["notes_dir"] or "unknown"}')
        print(f'Active index: {result["active_version"] or "none"}')
        for name, value in result['backend'].items():
            print(f'{name}: {value}')
        print(f'Builds: {len(result["builds"])}')
        for build in result['builds']:
            print(f'  {build["index_version"]}: {build["status"]}, '
                  f'{build["document_count"]} documents, {build["chunk_count"]} chunks')


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    _validate_arguments(parser, args)
    try:
        with Runtime(_runtime_config(args)) as runtime:
            result = _execute(runtime, args)
        _print_result(args, result)
        if args.command == 'ask' and result.stop_reason == 'max_turns':
            print('Agent reached --max-turns without a final response.', file=sys.stderr)
            return 1
        return 0
    except (OSError, ValueError, LookupError, TypeError, sqlite3.Error, ResponseError, HTTPError,
            subprocess.CalledProcessError, ResponseHandlingException, UnexpectedResponse) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
