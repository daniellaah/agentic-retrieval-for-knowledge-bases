# ARKB v0.1.0 — Release Notes

ARKB (Agentic Retrieval for Knowledge Bases) provides agent-controlled retrieval over local Markdown notes, with deterministic retrieval capabilities that can also be used directly from Python or the CLI.

## Highlights

- Agent-driven `match`, `search`, and `read` tools, with inspectable traces and a bounded conversation loop. Repeated searches that return no new evidence prompt the model to use collected evidence and explain gaps.
- Exact lexical matching, BM25, semantic search, and hybrid retrieval with Reciprocal Rank Fusion. Optional local Qwen3 reranking refines candidate order.
- Section-aware Markdown chunking, stable document identities, reusable embeddings, SQLite snapshots, and Qdrant vectors. Candidate indexes are validated before atomic publication.
- Independent generation APIs for context budgeting, citation validation, and quoted evidence with source offsets.
- Retrieval baselines, agent behavior evaluation, and model ablation workflows with saved results and provenance.
- Five CLI commands: `match`, `search`, `ask`, `index`, and `status`, each supporting JSON output.

The implementation separates knowledge processing, retrieval, generation, agent control, interfaces, and evaluation. Runtime owns service clients and initializes capabilities as needed.

## Requirements and Quick Start

Use Python **3.13** (`>=3.13,<3.14`), `uv`, and ripgrep (`rg`). For the complete workflow, run Ollama and Qdrant Server; their default endpoints are `http://127.0.0.1:11434` and `http://127.0.0.1:6333`.

From the repository root:

```sh
uv sync --locked
ollama pull qwen3-embedding:0.6b
ollama pull qwen3.5:4b

uv run --locked arkb index --notes-dir example_notes
uv run --locked arkb status
uv run --locked arkb search "agent memory" --mode hybrid --top-k 5
uv run --locked arkb ask "Find notes about agent memory and explain the relevant evidence." --trace
```

The first index build downloads the pinned embedding tokenizer unless it is already cached. `--offline` prevents model-file and tokenizer downloads; configured Ollama and Qdrant requests still occur.

Exact `match` needs neither an index nor a model/vector service. BM25 needs a compatible SQLite snapshot but no running model/vector service.

For optional reranking:

```sh
uv sync --locked --extra rerank
uv run --locked --extra rerank arkb search "agent memory" --mode hybrid --rerank
```

Reranking uses the pinned `Qwen/Qwen3-Reranker-0.6B` model locally through PyTorch and Transformers. Model files must be cached before offline use.

## Compatibility with Earlier Repository Checkouts

Storage and identity schemas are version **2**. Rebuild pre-v2 or retired NumPy indexes into a new database:

```sh
uv run --locked arkb index --notes-dir /path/to/notes --db /path/to/new-index.sqlite
```

Renaming an old database or collection does not migrate identities or embedding-cache keys. Earlier databases and collections remain untouched. Compatible v2 snapshots retain their persisted format through the reviewed simplification.

Python callers using earlier module layouts should update these references:

| Earlier interface | v0.1.0 interface |
| --- | --- |
| `arkb.evaluation.agent_runner` | `arkb.evaluation.runs` |
| `arkb.evaluation.agent` metrics | `arkb.evaluation.agent_metrics` |
| `arkb.retrieval.fusion.rrf` | `from arkb.retrieval import rrf` |
| `arkb.runtime.search_index` | `Runtime.search` or a prepared `SnapshotSemanticRetriever.search` |
| `Note.path` / `Chunk.path` | `.source` |
| `DocumentAccess.read(...).chunk` | A `DocumentSlice` with direct `.source`, `.title`, `.content`, and span fields |
| `BuiltContext.evidence_blocks` / `EvidenceBlock` | `BuiltContext.citation_sources`, using `CitationSource` and `CitationOrigin` |
| `SQLiteStorage.add_chunk(..., ordinal=...)` | `SQLiteStorage.add_chunks(version, records)` with the complete ordered record batch |

Built contexts validate their message/source mapping at construction. Agent tool JSON continues to represent live reads with `chunk_id: null`. Existing retrieval diagnostic JSON, including fusion and reranking provenance, is retained.

## Known Limits

- Ingestion reads UTF-8 `.md` files directly inside one directory; it does not recurse into subdirectories.
- Ranked `search` uses indexed content. `match` and `read` use current files, so locations from an earlier index can become stale after edits. Re-index to update ranked retrieval.
- MCP is not implemented; this version exposes no MCP server or MCP tools.
- `ask` returns the agent model's final text directly. The separate generation package's citation-validation stage is not applied to that command.
- `ask` defaults to eight model requests, including the final response, with thinking enabled and reranking disabled. Exhausting the budget returns no final answer and a nonzero CLI exit status. Model/tool failures preserve available traces and are not automatically retried by the agent loop.
- Agent evaluation measures evidence gathering and tool behavior, not final-answer correctness. Citation reference validation does not establish semantic support for a claim.
- `status` reports saved state; it does not check live freshness or external-service health.

## Validation of the Reviewed Code

Independent review covered the full `7bbf4ce..5cda3ce` range: 16 commits and 48 changed files. No verified blocking or material regression was found.

| Check | Result |
| --- | --- |
| Regular tests | 969 passed; zero failures, errors, or skips |
| Real integration tests | 63 passed; zero failures, errors, or skips |
| Baseline/current deterministic comparisons | 1,768 cases produced byte-identical output, including retrieval diagnostics, tools, budgets, and both citation modes |
| Distribution build | Source archive and wheel built successfully |
| Installed package | All 42 modules imported; six CLI/evaluation help entry points and installed-console live matching passed |
| Optional dependencies | Ordinary imports and BM25 worked without reranker dependencies; reranking provided installation guidance |

Real integration tests used cached models and an isolated Qdrant container with a new 160-chunk example snapshot. The container was removed after validation. Wheel verification reused installed dependencies rather than resolving a fresh environment from the network. Model results establish behavior for the exercised cases, not arbitrary queries.

See the [README](../README.md) for usage, the [evaluation guide](../evaluation/README.md) for experiments, and the [simplification record](simplification-results.md) for implementation and validation details.

License: [Apache License 2.0](../LICENSE).
