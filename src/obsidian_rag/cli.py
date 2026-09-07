"""Answer questions about Markdown notes from the command line."""

import argparse
from collections.abc import Sequence
from contextlib import ExitStack, closing
from dataclasses import asdict
from functools import partial
import json
import math
from pathlib import Path
import sqlite3
import sys

from httpx import HTTPError
from ollama import Client, ResponseError

from obsidian_rag.knowledge_base.chunking import chunk_notes, whole_note_chunks
from obsidian_rag.knowledge_base.embeddings import prepare_document, prepare_query, embed_texts, resolve_embedding_spec
from obsidian_rag.context import ContextConfig, build_context, load_generation_counter
from obsidian_rag.generation import generate_answer, generate_cited_answer
from obsidian_rag.knowledge_base.loaders import load_notes
from obsidian_rag.retrieval import SearchResult, retrieve, connect_qdrant
from obsidian_rag.knowledge_base.vector_index.manifest import ChunkRecord, fingerprint_config
from obsidian_rag.knowledge_base.tokenization import count_tokens, load_tokenizer


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

def _legacy_main(argv: Sequence[str] | None = None) -> int:
    """Print an answer and return zero, or report a runtime error and return one.

    Parse the supplied arguments, or the process arguments when argv is None.
    Argument errors exit with status two; help exits with status zero.
    """
    parser = argparse.ArgumentParser(
        prog="obsidian-rag",
        description="Answer a question using local Markdown notes. Persistent commands: index, query, status.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("question", help="Question to answer from the notes.")
    parser.add_argument(
        "--notes-dir",
        type=Path,
        default=Path("example_notes"),
        help="Directory containing Markdown notes, relative to the working directory.",
    )
    parser.add_argument(
        "--top-k", type=int, default=2, help="Maximum number of chunks to retrieve."
    )
    parser.add_argument(
        "--embedding-model",
        default="qwen3-embedding:0.6b",
        help="Ollama embedding model; queries use a Qwen retrieval instruction.",
    )
    parser.add_argument(
        "--generation-model",
        default="qwen3.5:4b",
        help="Ollama model for answer generation.",
    )
    parser.add_argument(
        "--host", default="http://127.0.0.1:11434", help="Ollama server URL."
    )
    parser.add_argument(
        "--timeout", type=float, default=180.0, help="Request timeout in seconds."
    )
    parser.add_argument("--chunking", choices=("none", "recursive"), default="recursive",
                        help="Whole-note baseline or recursive text splitting.")
    parser.add_argument("--chunk-size", type=int, default=512,
                        help="Maximum body tokens per chunk.")
    parser.add_argument("--chunk-overlap", type=int, default=64,
                        help="Target overlap tokens between chunks.")
    parser.add_argument("--tokenizer-cache", type=Path,
                        help="Hugging Face tokenizer cache directory.")
    parser.add_argument("--offline", action="store_true",
                        help="Load the tokenizer from cache without Hub requests.")
    _add_context_arguments(parser)
    args = parser.parse_args(argv)
    _validate_context_arguments(parser, args)

    if not args.question.strip():
        parser.error("question must not be blank")
    if args.top_k <= 0:
        parser.error("--top-k must be a positive integer")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be a positive finite number")

    if args.chunking == "recursive":
        if args.chunk_size <= 0:
            parser.error("--chunk-size must be a positive integer")
        if not 0 <= args.chunk_overlap < args.chunk_size:
            parser.error("--chunk-overlap must be nonnegative and less than --chunk-size")
        if args.embedding_model != "qwen3-embedding:0.6b":
            parser.error("recursive chunking requires qwen3-embedding:0.6b; "
                         "use --chunking none for another embedding model")

    try:
        notes = load_notes(args.notes_dir)
        if not notes:
            raise ValueError(f"No Markdown notes found in {args.notes_dir}.")

        if args.chunking == "none":
            chunks = whole_note_chunks(notes)
        else:
            tokenizer = load_tokenizer(
                cache_dir=args.tokenizer_cache, local_files_only=args.offline,
            )
            chunks = chunk_notes(
                notes, count_tokens=partial(count_tokens, tokenizer=tokenizer),
                chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
            )
        with Client(host=args.host, timeout=args.timeout, trust_env=False) as client:
            chunk_vectors = embed_texts(
                [prepare_document(chunk) for chunk in chunks],
                client=client,
                model=args.embedding_model,
            )
            query = prepare_query(args.question)
            query_vector = embed_texts(
                [query], client=client, model=args.embedding_model
            )[0]
            results = retrieve(chunks, chunk_vectors, query_vector, top_k=args.top_k)
            # Bind in-memory hits to this exact loaded corpus without writing an index.
            by_source = {note.source: note for note in notes}
            version = 'memory:' + fingerprint_config({'notes': [asdict(note) for note in notes]})
            results = [SearchResult(hit.chunk, hit.score,
                       ChunkRecord.from_note(hit.chunk, note=by_source[hit.chunk.source], vault_id='memory'), version)
                       for hit in results]
            context = _build_cli_context(args, results, client)
            if args.show_context:
                print(json.dumps(context.to_dict(), ensure_ascii=False))
                return 0
            answer = _answer_output(args, context, client)
    except (OSError, ValueError, ResponseError, HTTPError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(answer)
    return 0

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
            command.add_argument('--backend', choices=('numpy', 'qdrant'), default='numpy')
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
            command.add_argument('--json', action='store_true', help='Print retrieval results without generation.')
            command.add_argument('--generation-model', default='qwen3.5:4b')
            _add_context_arguments(command)
    return parser

def _persistent_main(argv: Sequence[str]) -> int:
    from obsidian_rag.knowledge_base.vector_index.indexing import build_index
    from obsidian_rag.knowledge_base.loaders import scan_notes
    from obsidian_rag.retrieval import search_index
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
                    qclient = None
                    backend = {'kind': args.backend}
                    if args.backend == 'qdrant':
                        qclient = resources.enter_context(closing(connect_qdrant(args.qdrant_url, args.timeout)))
                        backend.update(url=args.qdrant_url, hnsw_m=args.hnsw_m, ef_construct=args.ef_construct,
                                       indexing_threshold=args.indexing_threshold, full_scan_threshold=args.full_scan_threshold,
                                       index_timeout=args.index_timeout,
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
        with ExitStack() as resources, SQLiteStorage(args.db, read_only=True) as storage:
            manifest = storage.active_manifest(args.vault_id)
            if manifest is None:
                raise ValueError('No published index; run the index command first.')
            metadata = storage.build_metadata(manifest.index_version)['backend']
            qclient = None
            if metadata['kind'] == 'qdrant':
                qclient = resources.enter_context(closing(connect_qdrant(args.qdrant_url or metadata['url'], args.timeout)))
            tokenizer = load_tokenizer(cache_dir=args.tokenizer_cache, local_files_only=args.offline)
            with Client(host=args.host, timeout=args.timeout, trust_env=False) as client:
                spec = resolve_embedding_spec(client, args.embedding_model or manifest.embedding_spec.model,
                                     context_length=metadata['input']['max_tokens'])
                results = search_index(storage, args.question, vault_id=args.vault_id, spec=spec,
                                       tokenizer=tokenizer, client=client, top_k=args.top_k,
                                       source=args.source, exact=args.exact, index_version=manifest.index_version,
                                       qdrant_client=qclient)
                if args.json:
                    print(json.dumps({'index_version': manifest.index_version, 'question': args.question,
                                      'results': [asdict(r) for r in results]}, ensure_ascii=False))
                else:
                    context = _build_cli_context(args, results, client)
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
    if arguments and arguments[0] in ('index', 'query', 'status'):
        return _persistent_main(arguments)
    return _legacy_main(arguments)


if __name__ == '__main__':
    raise SystemExit(main())
