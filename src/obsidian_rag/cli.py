"""Answer questions about Markdown notes from the command line."""

import argparse
from collections.abc import Sequence
from contextlib import ExitStack, closing
from dataclasses import asdict
import json
import math
from pathlib import Path
import sqlite3
import sys

from httpx import HTTPError
from ollama import Client, ResponseError

from obsidian_rag.knowledge_base.embeddings import resolve_embedding_spec
from obsidian_rag.context import ContextConfig, build_context, load_generation_counter
from obsidian_rag.generation import generate_answer, generate_cited_answer
from obsidian_rag.knowledge_base.vector_index.qdrant import connect_qdrant
from obsidian_rag.knowledge_base.tokenization import load_tokenizer


def _add_context_arguments(parser):
    parser.add_argument('--context-window', type=int, default=8192,
                        help='Generation window in tokens; also sent as Ollama num_ctx.')
    parser.add_argument('--max-output-tokens', type=int, default=1024,
                        help='Reserved output tokens; also sent as Ollama num_predict.')
    parser.add_argument('--context-safety-margin', type=int, default=128)
    parser.add_argument('--show-context', action='store_true',
                        help='Print final messages, provenance and budget diagnostics without generation.')
    parser.add_argument('--citation-mode', choices=('structured', 'quoted', 'legacy'), default='structured',
                        help='Validated citations (default), exact quoted citations, or legacy filename prompting.')
    parser.add_argument('--answer-json', action='store_true',
                        help='Print structured answer, cited sources, raw response and validation.')

def _context_config(args):
    return ContextConfig(args.context_window, args.max_output_tokens, args.context_safety_margin)

def _validate_context_arguments(parser, args):
    try:
        _context_config(args)
    except ValueError as error:
        parser.error(str(error))
    if sum((getattr(args, 'json', False), args.show_context, args.answer_json)) > 1:
        parser.error('--json, --show-context and --answer-json are separate output modes')
    if args.answer_json and args.citation_mode == 'legacy':
        parser.error('--answer-json requires structured or quoted citations')

def _build_cli_context(args, results, client):
    counter = load_generation_counter(client=client, model=args.generation_model,
                                      cache_dir=args.tokenizer_cache, local_files_only=args.offline)
    return build_context(args.question, results, config=_context_config(args), counter=counter,
                          citation_mode=args.citation_mode)


def _answer_output(args, context, client):
    if args.citation_mode == 'legacy':
        return generate_answer(context, client=client)
    result = generate_cited_answer(context, client=client)
    return json.dumps(result.to_dict(), ensure_ascii=False) if args.answer_json else result.text

def _persistent_parser():
    from obsidian_rag.knowledge_base.embeddings import DEFAULT_QUERY_INSTRUCTION
    parser = argparse.ArgumentParser(prog='obsidian-rag')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('index', 'query', 'status'):
        command = commands.add_parser(name)
        command.add_argument('--db', type=Path, default=Path('.obsidian-rag/index.sqlite'))
        command.add_argument('--vault-id', default='default')
        if name == 'status':
            continue
        command.add_argument('--host', default='http://127.0.0.1:11434')
        command.add_argument('--timeout', type=float, default=180.0)
        command.add_argument('--tokenizer-cache', type=Path)
        command.add_argument('--offline', action='store_true')
        command.add_argument('--embedding-model', default='qwen3-embedding:0.6b' if name == 'index' else None)
        if name == 'index':
            command.add_argument('--notes-dir', type=Path, default=Path('example_notes'))
            command.add_argument('--qdrant-url', default='http://127.0.0.1:6333')
            command.add_argument('--hnsw-m', type=int, default=16)
            command.add_argument('--ef-construct', type=int, default=100)
            command.add_argument('--indexing-threshold', type=int, default=10000)
            command.add_argument('--full-scan-threshold', type=int, default=10000)
            command.add_argument('--index-timeout', type=float, default=30)
            command.add_argument('--require-hnsw', action='store_true')
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
            command.add_argument('--exact', action='store_true')
            command.add_argument('--ef-search', type=int)
            command.add_argument('--index-version', help='Pin a published snapshot instead of the active version.')
            command.add_argument('--json', action='store_true', help='Print retrieval results without generation.')
            command.add_argument('--generation-model', default='qwen3.5:4b')
            _add_context_arguments(command)
    return parser

def _persistent_main(argv: Sequence[str]) -> int:
    from obsidian_rag.knowledge_base.vector_index.indexing import build_index
    from obsidian_rag.knowledge_base.loaders import scan_notes
    from obsidian_rag.retrieval.vector import vector_search, VectorSearchConfig
    from obsidian_rag.knowledge_base.vector_index.storage import SQLiteStorage
    from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

    parser = _persistent_parser()
    args = parser.parse_args(argv)
    if not args.vault_id.strip():
        parser.error('--vault-id must not be blank')
    if args.command != 'status' and (not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error('--timeout must be positive and finite')
    if args.command == 'index':
        if (args.hnsw_m < 2 or args.ef_construct <= 0 or args.indexing_threshold < 0 or args.full_scan_threshold < 10
                or not math.isfinite(args.index_timeout) or args.index_timeout <= 0
                or (args.require_hnsw and args.indexing_threshold == 0)):
            parser.error('Invalid Qdrant HNSW configuration or readiness timeout')
        if args.context_length <= 0 or args.batch_size <= 0 or not 0 <= args.max_retries <= 8:
            parser.error('context/batch sizes must be positive and retries must be between 0 and 8')
        if args.max_batch_tokens is not None and args.max_batch_tokens <= 0:
            parser.error('--max-batch-tokens must be positive')
        if args.chunking == 'recursive' and (args.chunk_size <= 0 or not 0 <= args.chunk_overlap < args.chunk_size):
            parser.error('chunk size must be positive and overlap must be in [0, chunk size)')
    if args.command == 'query':
        _validate_context_arguments(parser, args)
        if args.ef_search is not None and args.ef_search <= 0:
            parser.error('--ef-search must be positive')
    if args.command == 'query' and (not args.question.strip() or args.top_k <= 0):
        parser.error('question must not be blank and --top-k must be positive')
    try:
        if args.command == 'status':
            with ExitStack() as resources, SQLiteStorage(args.db, read_only=True) as storage:
                active = storage.active_manifest(args.vault_id)
                print(json.dumps({'vault_id': args.vault_id,
                                  'active_version': active.index_version if active else None,
                                  'builds': [asdict(m) for m in storage.list_builds(args.vault_id)]}, ensure_ascii=False))
            return 0
        if args.command == 'index':
            # Finish the source scan before opening or changing index state.
            notes = scan_notes(args.notes_dir)
            tokenizer = load_tokenizer(cache_dir=args.tokenizer_cache, local_files_only=args.offline)
            with Client(host=args.host, timeout=args.timeout, trust_env=False) as client:
                spec = resolve_embedding_spec(client, args.embedding_model, context_length=args.context_length)
                with ExitStack() as resources, SQLiteStorage(args.db) as storage:
                    qclient = resources.enter_context(closing(connect_qdrant(args.qdrant_url, args.timeout)))
                    backend = dict(kind='qdrant', url=args.qdrant_url, hnsw_m=args.hnsw_m,
                                   ef_construct=args.ef_construct, indexing_threshold=args.indexing_threshold,
                                   full_scan_threshold=args.full_scan_threshold, index_timeout=args.index_timeout,
                                   require_hnsw=args.require_hnsw)
                    report = build_index(storage, notes, spec=spec, vault_id=args.vault_id, client=client,
                                         tokenizer=tokenizer, max_input_tokens=args.context_length,
                                         chunking=args.chunking, chunk_size=args.chunk_size,
                                         chunk_overlap=args.chunk_overlap, batch_size=args.batch_size,
                                         max_batch_tokens=args.max_batch_tokens, max_retries=args.max_retries,
                                         query_instruction=args.query_instruction, force=args.force,
                                         source_scope=str(args.notes_dir.resolve()), backend=backend, qdrant_client=qclient)
                    print(json.dumps(asdict(report), ensure_ascii=False))
            return 0
        if not args.db.is_file():
            raise ValueError('No published index; run the index command first.')
        with ExitStack() as resources, SQLiteStorage(args.db, read_only=True) as storage:
            manifest = storage.get_manifest(args.index_version) if args.index_version else storage.active_manifest(args.vault_id)
            if manifest is None:
                raise ValueError('No published index; run the index command first.')
            if manifest.status != 'ready' or manifest.vault_id != args.vault_id:
                raise ValueError('Queries require a ready snapshot in the requested vault.')
            metadata = storage.build_metadata(manifest.index_version)['backend']
            if metadata['kind'] != 'qdrant':
                raise ValueError('This index is not Qdrant; run index to rebuild it in Qdrant.')
            qclient = resources.enter_context(closing(connect_qdrant(args.qdrant_url or metadata['url'], args.timeout)))
            tokenizer = load_tokenizer(cache_dir=args.tokenizer_cache, local_files_only=args.offline)
            with Client(host=args.host, timeout=args.timeout, trust_env=False) as client:
                spec = resolve_embedding_spec(client, args.embedding_model or manifest.embedding_spec.model,
                                     context_length=metadata['input']['max_tokens'])
                response = vector_search(storage, args.question, vault_id=args.vault_id, spec=spec,
                                         tokenizer=tokenizer, client=client, qdrant_client=qclient,
                                         index_version=manifest.index_version,
                                         config=VectorSearchConfig(args.top_k, args.source, args.exact, args.ef_search))
                if args.json:
                    print(json.dumps(response.to_dict(), ensure_ascii=False))
                else:
                    context = _build_cli_context(args, response.items, client)
                    if args.show_context:
                        print(json.dumps(context.to_dict(), ensure_ascii=False))
                    else:
                        print(_answer_output(args, context, client))
        return 0
    except (OSError, ValueError, sqlite3.Error, ResponseError, HTTPError,
            ResponseHandlingException, UnexpectedResponse) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1

def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and not arguments[0].startswith('-') and arguments[0] not in ('index', 'query', 'status'):
        arguments.insert(0, 'query')
    return _persistent_main(arguments)


if __name__ == '__main__':
    raise SystemExit(main())
