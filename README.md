# ARKB

Local retrieval over Markdown notes using Python, Qdrant, and Ollama.

The repository includes a command-line interface, a Markdown note loader, an
Ollama embedding client, ranked retrieval, an agent tool-calling runtime,
fixed-pipeline answer generation, and sample Markdown notes in `example_notes/`.

## Python modules

The package follows six stable capability boundaries. `runtime.py` composes
resources and reusable retrievers; `config.py` supplies explicit application
settings. Indexing prepares knowledge, retrieval returns evidence, and generation
packs that evidence into context and produces validated citations.

| Capability in `src/arkb/` | Modules and responsibility |
| --- | --- |
| `knowledge/` | `models.py`: notes, chunks, manifests and stable identities; `documents.py`: Markdown loading, scans and live document access; `chunking.py`: the single chunker; `embeddings.py`: input preparation, embedding tokenizer and model adapters; `indexing.py`: builds and publication; `sqlite.py` / `qdrant.py`: persistence and raw data access |
| `retrieval/` | `models.py`: source-based contracts; `bm25.py` / `semantic.py`: independent retrieval; `hybrid.py` / `fusion.py`: fixed composition and RRF; `rerank.py`: reranking contracts; `qwen_rerank.py`: the fixed Qwen3 scorer; `engine.py`: explicit mode selection; `exact.py`: literal/regex matching over live source text |
| `generation/` | `models.py`: context/citation contracts; `context.py`: evidence packing and budgets; `citations.py`: parsing, validation and rendering; `generate.py`: answer generation and generation token counting |
| `agent/` | `tools.py`: thin `match`, `search`, `read` adapters and provider-independent tool definitions; `state.py`: conversation and turn count; `loop.py`: bounded model/tool orchestration |
| `interfaces/` | `cli.py`: argument parsing, Runtime calls and output formatting; `mcp.py`: protocol placeholder |
| `evaluation/` | `datasets.py`: experiment inputs and fingerprints; `models.py`: evaluation contracts; `agent.py`: deterministic agent metrics; `baselines.py`: fixed engine execution and source-level reporting; `agent_runner.py`: Agent/baseline runs and artifacts; `metrics.py`: ranking, coverage and citation metrics; `retrieval.py`: relevance, ANN and frozen-candidate experiments; `generation.py`: context and citation experiments |
| `runtime.py` | Lazy client/tokenizer creation, resource reuse and closure, snapshot-bound retrieval composition and `match`/`search`/`ask`/`index`/`status` entry points |
| `config.py` | Explicit runtime/retrieval settings and application defaults; no I/O |

Use `arkb.knowledge.indexing.build_index`,
`arkb.retrieval.SemanticRetriever.search`, and
`arkb.generation.context.build_context` as the main stage APIs. `arkb.retrieval`
exports the shared contracts, independent retrieval primitives and `RetrievalEngine`.
`arkb.runtime.SnapshotSemanticRetriever` composes the current saved-snapshot
adapters; `arkb.runtime.search_index` remains the one-shot entry point. Direct
vector reads and writes use `arkb.knowledge.qdrant.search_qdrant` and
`arkb.knowledge.qdrant.QdrantIndex`, with `QdrantConfig` from
`arkb.knowledge.models`. Knowledge does not import Retrieval or Generation;
production capabilities do not import Evaluation.

`with Runtime(RuntimeConfig(...)) as runtime:` owns only the model/vector
clients it opens. Its tokenizer and clients are loaded on demand and reused
until the context exits, including exceptional exits. Explicitly supplied SQLite
storage and clients remain caller-owned. Keep a prepared retriever or engine to
retain its pinned snapshot across a query session. BM25-only setup opens neither
model nor vector clients; importing the retrieval package never loads a model.
Agent tools expose retrieval and live document access. `Runtime.ask` composes
them and calls `Runtime.run_agent` for a bounded conversation; the lower-level
entry point still accepts prepared tools and an injected client. One-shot entry
points close their own SQLite connections. MCP remains unimplemented and can
reuse these same Runtime entry points.

Python imports now use these capability paths; the old root `chunking`,
`embeddings`, `tokenization`, `storage`, `schema`, `cli`, `indexing`, `context`
and `retrieval_evaluation` paths have been removed. This import migration does
not change persisted identities, index formats or product CLI behavior. See the
[directory plan](docs/directory-refactor-plan.md) and
[completed migration](docs/refactor-results.md) for the mapping and validation.

The package and command are named `arkb`. Run `uv sync --locked` after updating.
Only `match`, `search`, `ask`, `index`, and `status` are supported; the old bare-question command,
`obsidian-rag` alias, `retrieve`, and `search_numpy` APIs have been removed.
Qdrant is the semantic search backend; BM25 uses saved SQLite text and an
in-memory lexical index. NumPy remains a dependency for embedding
validation, vector serialization, snapshot verification, and evaluation statistics.

The default `.obsidian-rag/index.sqlite` path, persisted identity namespaces,
Qdrant collection names/ownership, and `OBSIDIAN_RAG_*` test variables retain
their existing names. Existing Qdrant indexes remain compatible. Queries against
old NumPy indexes explicitly require `arkb index` to rebuild into Qdrant. Keep the
same `--db`, `--vault-id`, note directory, and embedding settings to reuse compatible
cached vectors. The old active snapshot is replaced only after successful publication;
rebuilding does not delete historical snapshots or the embedding cache.

Tests roughly mirror the six capabilities under `tests/`. Shared fixtures live
in `tests/conftest.py`; runtime ownership checks live in `tests/test_runtime.py`.
Real-service checks live in each capability's `integration/` subdirectory and
carry the `integration` marker as well as their existing environment gates.
Run regular tests with `uv run --locked python -m pytest -q -m "not integration"`; they
need no network or running services. Exact lexical matching tests require `rg`
(ripgrep) on `PATH`.

## Requirements

- Python 3.13
- uv
- ripgrep (`rg` on `PATH`) for the `match` tool
- Ollama, running locally for model operations
- Qdrant Server, running for indexing and semantic/hybrid queries
- Optional `uv sync --locked --extra rerank` for local Qwen3 reranking

## Set up the Python environment

Run the following command from the repository root:

```sh
uv sync --locked
```

This installs the project, its runtime dependencies, and the development
dependencies into `.venv/`. The default development dependency is pytest.
Dependencies are declared in `pyproject.toml`; exact resolved versions are stored
in `uv.lock`. The uv cache is kept in `.uv-cache/`.

Use `uv run` to execute commands in the project environment without activating it
manually:

```sh
uv run --locked python --version
uv run --locked python -c "import arkb, numpy, ollama; print('Imports OK')"
uv run --locked python -m pytest --version
```

## CLI: match, search, ask, index, status

```sh
uv run --locked arkb index --notes-dir example_notes --qdrant-url http://127.0.0.1:6333
uv run --locked arkb match "RAG"
uv run --locked arkb search "Agent Memory"
uv run --locked arkb search "Agent Memory" --mode bm25
uv run --locked arkb search "如何管理智能体的长期记忆" --mode hybrid --top-k 5 --json
uv run --locked arkb ask "我想写一篇 Agent Memory 的文章，帮我找相关素材" --trace
uv run --locked arkb status
```

| Command | Meaning | Main options |
| --- | --- | --- |
| `match <pattern>` | Case-sensitive literal occurrences in live note bodies; no model or index required | `--source`, `--top-k` (5), `--notes-dir` |
| `search <query>` | Deterministic ranked Retrieval Engine results from the saved snapshot; no generation or agent loop | `--mode bm25\|semantic\|hybrid` (semantic), `--top-k` (2), `--source` |
| `ask <query>` | Existing Agent Runtime chooses match/search/read and search modes, processes observations, and returns its final response | `--think` / `--no-think` (on), `--max-turns` (8), `--trace`, `--notes-dir`, `--generation-model` (`qwen3.5:4b`) |
| `index` | Existing scanning, chunking, embedding, cache reuse and snapshot publication | `--notes-dir` (`example_notes`), `--force`, chunking/embedding/Qdrant options |
| `status` | Saved manifests, active version, note directory and backend configuration; no service probes | `--db`, `--vault-id` |

Every command supports `--db` (default `.obsidian-rag/index.sqlite`), `--vault-id`
(default `default`), and `--json`. Default output is human-readable. **`--json`
changes only serialization:** match/search return a `SearchResponse` with `query`,
`method`, `index_id` and `results`; ask returns the existing `AgentResult` with
`response`, `stop_reason`, and `state` (messages and model-turn count). Index JSON
retains its build report, and status JSON retains `vault_id`, `active_version`,
and `builds`, adding `notes_dir` and saved `backend` information.

Normal ask text contains only the final response. `--trace` formats a summary of
the available trajectory to stderr; final text or JSON stays on stdout, including when both
`--trace` and `--json` are selected. Trace does not change the agent run:

```text
[1] search
query: "agent memory"
mode: "bm25"

[2] read
source: "34_agent_memory_lifecycle.md"

[3] final
```

`--max-turns` counts model requests, including the final response. If the limit
is exhausted, there is no fabricated answer: JSON contains `response: null` and
`stop_reason: "max_turns"`, a diagnostic goes to stderr, and the exit code is 1.
Other exit codes remain 0 for success, 1 for runtime failure, and 2 for invalid
arguments. On model/tool failures, `--trace` also renders the partial call history
with an `error` stop reason. Errors still propagate from the Agent Runtime;
no final response or success JSON is fabricated. This CLI summary lists requested
calls, including any without a recorded result; full observations are available
through the Python `AgentTrace` API below.

The CLI calls only `Runtime.match`, `Runtime.search`, `Runtime.ask`,
`Runtime.index`, and `Runtime.status`. Runtime composes `ExactRetriever`,
`RetrievalEngine`, the existing `run_agent`/`AgentTools`, `build_index`, and
`SQLiteStorage`, respectively. It contains no query-text routing. For ask, BM25
and semantic adapters are prepared only when the agent chooses to search; the
same snapshot is retained across turns and modes. **Ask has no `--mode` option.**

Agent thinking is enabled by default for the current `qwen3.5:4b` model.
`--think` enables it explicitly; `--no-think` disables it for comparison or a
model that does not support thinking. This changes the model request, independently
of `--json` or `--trace`. Normal text and trajectory output omit provider thinking
text; JSON retains the full `AgentResult.state`, including provider message fields.
There is no model-name heuristic or automatic fallback if the provider rejects
the setting. Thinking can improve multi-step completion and can also increase
repeated searches and latency; use `--max-turns` to bound model turns.

```sh
arkb ask "帮我核对关联笔记中的事实" --think --trace
arkb ask "帮我核对关联笔记中的事实" --no-think --json
```

Search retains explicit advanced controls: `--candidate-k` (20), `--rrf-k` (60),
`--rerank`, `--rerank-candidates` (20),
`--reranker-cache`, `--reranker-max-length` (512), and `--exact` (Qdrant exact vector
search). Algorithms, fusion and reranking remain in Retrieval. BM25 needs no model
or vector service. Semantic/hybrid may call the embedding model; neither calls a
chat/generation model. ANN and embedding-provider behavior retain their existing
repeatability limits; there is no agent planning or hidden fallback in search.

Index keeps its existing options and behavior: `--chunking none|recursive`,
`--chunk-size` (512), `--chunk-overlap` (64), `--context-length` (8192),
`--batch-size` (32), `--max-batch-tokens`, `--max-retries` (2),
`--query-instruction`, `--force`, `--hnsw-m`, `--ef-construct`,
`--indexing-threshold`, `--full-scan-threshold`, `--index-timeout`, and
`--require-hnsw`. `--force` rebuilds while reusing compatible vectors. An empty
directory publishes an empty index; a failed scan leaves the active index intact.
Index/search/ask also accept `--host`, `--timeout`, `--embedding-model`,
`--qdrant-url`, `--tokenizer-cache`, and `--offline`. Offline prevents model-file
and tokenizer downloads; configured Ollama/Qdrant service calls still occur.

Repeat index after editing notes. Search reads indexed content, while match/read
read current files. Status reports saved state; it does not currently compute
live index freshness or probe backend health. For match/ask, the note directory
comes from the saved index scope, falling back to `example_notes` when no scope
exists. `--notes-dir` can select an unindexed directory; when an index has a saved
scope, an explicit directory must match it. This prevents mixing indexed
knowledge with unrelated live documents. Relative paths resolve from the current
working directory. The existing flat Markdown scope is unchanged.

**Migration:** `query` has been removed, with no alias. Use `search` for old
retrieval-only usage (`query ... --json`) and `ask` for agentic responses. The old
CLI `--show-context`, `--answer-json`, context-budget and citation switches are
removed; fixed RAG remains a Python API and evaluation baseline:

```python
from arkb.runtime import Runtime
from arkb.generation.context import build_context
from arkb.generation.generate import load_generation_counter, generate_cited_answer
from arkb.generation.models import ContextConfig

with Runtime() as runtime:
    question = "Why combine lexical and vector retrieval?"
    retrieved = runtime.search(question, mode="hybrid", top_k=5)
    client = runtime.model_client()
    counter = load_generation_counter(client=client)
    context = build_context(question, retrieved.results, config=ContextConfig(), counter=counter)
    answer = generate_cited_answer(context, client=client)
    print(answer.text)
```

Use `arkb --help` or any command's `--help` for all parameters. The console entry
point remains `arkb.interfaces.cli:main`; `python -m arkb.interfaces.cli` exposes
the same five commands. CLI contract tests use a fake Runtime; workflow tests use
Qdrant Local and fake Ollama/tokenizer boundaries, with no real LLM calls.

## Read notes

`arkb.knowledge.documents.load_notes` accepts a directory as a `pathlib.Path` and returns
notes with `title`, `content`, and `source` fields. It reads UTF-8 `.md` files
directly inside that directory in filename order.

The first line starting with `# ` becomes the title and is removed from the body.
If there is no such line, the filename without its extension becomes the title.
Other Markdown remains in the body, with leading and trailing whitespace removed.
Source references contain filenames rather than absolute paths. An empty
directory returns an empty list; filesystem and decoding errors are reported to
the caller.

Run the tests with:

```sh
uv run --locked python -m pytest -q
```

## Agent tools

`arkb.agent.AgentTools` exposes three primitives. `TOOL_DEFINITIONS` contains
plain JSON input schemas and descriptions of when to use each tool, with no
provider SDK or dispatch logic in the tool definitions. `tools.tool_definitions()`
returns a copy whose search-mode choices reflect the composed engine; the agent
loop uses this copy so a BM25-only engine advertises only BM25.

```python
match(query, *, target="content", regex=False, case_sensitive=True,
      source=None, limit=5)
search(query, *, source=None, limit=5, mode=None)
read(document_id=None, *, source=None, section_id=None, start_char=None, end_char=None)
```

- `match`: use for a known word, phrase, symbol, filename, or text pattern.
  Literal substring matching is case-sensitive by default, with no tokenization
  or implicit word boundaries. Set `regex=True` for a pattern (for example,
  `r"\bword\b"`); unsupported or malformed expressions raise `ValueError`.
  `target="content"` searches normalized bodies and returns each exact occurrence
  with character coordinates. `target="source"` searches filenames/relative
  paths and returns each matching document once, with its body and null offsets.
  Ordering is filename, then position; `limit` caps the total result count.
- `search`: use for a question, topic, or concept when the wording is unknown.
  Delegates directly to `RetrievalEngine.search`, preserving its result order.
  The agent can choose `mode="bm25"`, `"semantic"`, or `"hybrid"` per call.
  Omitting mode uses the application's configured default. Fusion and reranking
  stay inside Retrieval; scores and internal metadata remain absent from the
  agent-facing evidence results.
- `read`: supply a returned `document_id` or known `source` filename
  (for example, `read(source="rag.md")`) to read current source text or expand
  context. If both are provided they must identify the same document; conflicting
  selectors raise `LookupError`. With no section/range selector, read the full
  body. A range uses zero-based Python
  character positions in `Note.content`, with an exclusive end; an omitted
  endpoint means the corresponding document boundary. Out-of-bounds ranges
  fail instead of silently clipping. `section_id` and range parameters are
  mutually exclusive. A section includes its heading and direct body, ending
  at the next heading, using the existing Markdown parser and IDs.

`source` is an exact knowledge-relative path filter, applied before `limit`.
The current scope matches the loader/indexer: UTF-8 `.md` files directly in the
configured directory, without recursive traversal. Live document access excludes
symlinks that resolve outside that root. Content follows the existing loader's
title removal and whitespace handling; these are not raw-file line/byte offsets.

`match` and `search` return `{"query": ..., "results": [...]}`; `read` returns
`{"result": ...}`. Every evidence object has exactly these JSON-friendly fields:

| Field | Meaning |
| --- | --- |
| `document_id` | Existing vault/path document identity (`ChunkRecord.document_id`) |
| `source` | Knowledge-relative source path |
| `title` | Document title, or null when unavailable |
| `content` | Verbatim body text or excerpt; an empty body is valid |
| `document_revision` | Existing title/body content digest, or null when unavailable |
| `chunk_id` | Indexed chunk ID for search hits; null for live match/read results |
| `section_id` | Search hit's section or explicitly read section; otherwise null |
| `start_char`, `end_char` | Body character range, or null when unavailable |

The adapter maps the retrieval model's existing `source_id` field to the public
name `document_id`; it creates no new identity scheme. Edits preserve document
identity within a vault/path, while renames change it. Unknown document or section
IDs raise `LookupError`; malformed IDs/parameters raise `ValueError`. No matches
produce an empty `results` list. Filesystem, decoding, and backend errors propagate
instead of being reported as empty successful results.

Compose once with the existing runtime and the same directory/vault used for
indexing. A manually prepared engine must support the modes used by its tools;
`Runtime.ask` supplies all three supported modes lazily:

```python
from pathlib import Path
from arkb.knowledge.sqlite import SQLiteStorage
from arkb.runtime import Runtime

with Runtime() as runtime, SQLiteStorage(Path(".obsidian-rag/index.sqlite"), read_only=True) as storage:
    manifest = storage.active_manifest("default")
    if manifest is None:
        raise ValueError("Build an index first.")
    engine = runtime.retrieval_engine(storage, manifest, modes=("semantic",))
    tools = runtime.agent_tools(engine=engine, directory=Path("example_notes"),
                                vault_id=manifest.vault_id)
    hits = tools.search("How can retrieval improve answers?")["results"]
    if hits:
        document = tools.read(hits[0]["document_id"])["result"]
```

Runtime injects `DocumentAccess`, `ExactRetriever`, and the prepared engine; tools
do not import Runtime or own clients. Exact matching runs `rg` with fixed argv and
normalized text on stdin inside Retrieval, without a shell. `rg` is a replaceable
backend detail; its executable must be available for matching. `match` and `read`
do not call embedding, vector, or generation services.

**Eventual consistency:** `match` and `read` read current files on every call;
`search` reads the built index captured by its prepared engine. Until reindexing,
search can return old content, ranges, sections, or deleted document IDs. A live
read returns current content (or an error), without checking the indexed revision.
After rebuilding, prepare an engine from the latest active manifest to see that
index. No new snapshots, persistence, or version management are introduced.
For whole-note indexes, use `read(document_id)` to expand the entire document:
their root section ID can also identify just the current Markdown preamble.
`read(section_id=...)` always interprets IDs as current Markdown sections.

Run focused tests with:

```sh
uv run --locked python -m pytest -q tests/agent tests/retrieval/test_exact.py tests/knowledge/test_documents.py tests/test_runtime.py
```

## Agent runtime

`arkb.agent.run_agent(query, *, client, tools, model, max_turns=8, think=True)` runs a minimal
ReAct-style conversation. `Runtime.run_agent` supplies a reused Ollama client and
defaults to `qwen3.5:4b`; an explicitly supplied `client` remains caller-owned.
Both `runtime.ask(..., think=False)` and `runtime.run_agent(..., think=False)`
forward the explicit boolean unchanged on every model turn. The default is
`DEFAULT_AGENT_THINK=True`; non-booleans are rejected. This setting belongs only
to Agent execution. The independent fixed RAG generation API keeps its existing
non-thinking token-counting profile.
The model must support tool calling. The loop follows the existing
[Ollama tool-calling protocol](https://docs.ollama.com/capabilities/tool-calling).

To run the complete slice with local Markdown and an in-memory lexical index:

```python
from pathlib import Path
from arkb.knowledge.documents import DocumentAccess
from arkb.retrieval import BM25Retriever, RetrievalEngine
from arkb.runtime import Runtime

directory = Path("example_notes")
documents = DocumentAccess(directory, vault_id="default")
engine = RetrievalEngine(bm25=BM25Retriever(list(documents.records()), index_id="local"))

with Runtime() as runtime:
    tools = runtime.agent_tools(engine=engine, directory=directory,
                                vault_id="default", mode="bm25")
    result = runtime.run_agent("帮我找一些写 Agent Memory 的素材", tools=tools, max_turns=8)
    print(result.stop_reason)
    if result.response is not None:
        print(result.response)
    print([call["function"]["name"] for call in result.state.tool_calls])
```

For semantic or hybrid retrieval, use `runtime.retrieval_engine` with an existing
snapshot as in the composition example above, then pass its tools to
`runtime.run_agent`. The application prepares supported capabilities; the agent
selects `match`, `search`, or `read`, and the mode for each search. Knowledge and Retrieval never import
Agent, and the loop contains no retrieval algorithms or query routing rules.

Each run starts with a short system instruction and the user query. On every
turn the model sees the full conversation and all three function schemas. The
Ollama SDK parses structured `message.tool_calls`; the loop preserves the
assistant message, executes its calls sequentially, and appends one message per
result: `{"role": "tool", "tool_name": name, "content": "<JSON result>"}`. The next
model request includes these observations. This supports multiple calls in a
single response (including repeated names) and repeated calls across turns.
Ollama associates these observations by tool name and order; this is not an
OpenAI `tool_call_id` adapter.

An assistant message without calls is final if it contains nonblank text. Its
text is returned unchanged in `AgentResult.response`. Inputs that need no
knowledge can finish on the first turn. `AgentState` stores only `messages` and
`turn`; `tool_calls` derives ordered history from the messages, without duplicating
results in an evidence store. A new query always starts a fresh conversation.
`stop_reason="final"` describes protocol termination; it does not certify that
the answer satisfies every requested fact. Thinking text alone is not a final answer.

`result.trace` derives an `AgentTrace` snapshot on demand from these messages and
the `AgentResult`. It records `query`, `turns` (attempted model requests), ordered
`tool_calls`, `final_response`, and `stop_reason`. Each `AgentToolTrace` contains
its one-based `turn`, tool `name`, `arguments`, and decoded JSON `result`.
Repeated tool names are paired with observations by turn and execution order.
A `result` of `None` means no observation was recorded: the call failed or had
not executed when the run stopped. No second trajectory is maintained in the loop.

```python
from dataclasses import asdict

trace = result.trace
print(asdict(trace))  # JSON-serializable snapshot, detached from AgentState.
```

Model/protocol/tool exceptions retain their original type and identity. When the
exception accepts attributes, `error.agent_result` holds the partial `AgentResult`
with `response=None` and `stop_reason="error"`; use `error.agent_result.trace`
to inspect it. Completed observations survive errors, including a failure midway
through a batch. The failed model request counts toward `turns`. Invalid run
options and failures before the loop has initialized its state have no trajectory.
`AgentResult` serialization still contains only `response`, `stop_reason`, and
`state`; serialize `asdict(result.trace)` explicitly for the structured trace.

`max_turns` bounds **model requests**, including the final response. Every tool
call in the last allowed response is executed and recorded; if that response
contains calls, the run stops with `stop_reason="max_turns"` and `response=None`.
The caller can inspect the collected trajectory. There is no extra finalization
request or synthetic answer. The bound does not limit calls per response, total
tokens, or tool execution time; Runtime's timeout applies to model requests.

Errors propagate immediately: unknown tools, invalid arguments, missing documents,
backend/model failures, malformed model output, truncated output, and empty final
messages are not converted into successful responses. No retry or automatic error
recovery is implemented. Tool methods enforce their arguments even when the SDK
or model does not enforce JSON Schema constraints. In particular, installed
Ollama SDK 0.6.2 filters keywords such as `anyOf` and `additionalProperties` when
serializing tool schemas; descriptions explain selectors and Python validates them.

Deterministic tests use scripted models to check A/B/C tool dispatch, D's
`search → read → search → final` with real tools/retrieval, E's direct response,
and F's turn limit. They also cover batched calls, observation content/order,
empty results, independent runs, error propagation, and resource ownership.
An offline `httpx.MockTransport` test exercises the real Ollama SDK wire contract.
Scripted choices verify orchestration, not a real model's routing accuracy.

Real-model smoke tests are separate and opt-in; they assert tool trajectories
and successful termination without requiring fixed natural-language responses:

```sh
OBSIDIAN_RAG_RUN_MODEL_TESTS=1 uv run --locked python -m pytest -q tests/agent/integration/test_qwen_agent.py
```

They use temporary Markdown notes, host-configured BM25, and local Ollama, without
Qdrant or embedding requests. Optionally set `OBSIDIAN_RAG_AGENT_MODEL` to another
installed model that supports tool calling. Multi-turn capability is enforced by
the scripted test; real-model checks allow it to stop when its evidence is enough.

For the complete persisted runtime path, start local Ollama and Qdrant, cache the
embedding tokenizer, and run:

```sh
OBSIDIAN_RAG_RUN_MODEL_TESTS=1 \
OBSIDIAN_RAG_QDRANT_URL=http://127.0.0.1:6333 \
uv run --locked python -m pytest -q -s tests/agent/integration/test_runtime_qdrant.py
```

This test builds five temporary Markdown documents through the real indexing CLI,
embeds their chunks with `qwen3-embedding:0.6b`, publishes SQLite/Qdrant state, and
reopens the snapshot with a fresh Runtime for each query. The agent uses real
hybrid retrieval and Ollama tool calling. Cases cover match, search, direct read,
cross-document reading with a random verification code, direct conversation, and
the turn limit. No model, retrieval, storage, or tool client is mocked. The small
collection verifies query readiness; it does not benchmark HNSW/ANN performance.
The cross-document case fails if the model only names a reference instead of
retrieving the requested facts; a final message alone does not establish success.
Paired runs also compare `think=False` and `think=True` twice each on the same
snapshot, checking the actual HTTP option on every turn and recording elapsed
time, search/read counts, and new source/snippet counts. The non-thinking group
is a diagnostic control: its protocol checks may pass while `facts_complete` is
false. The thinking group must return the random code and retention period.
Search gains count distinct returned sources and exact snippets; they do not
measure semantic novelty or prove that the final answer is relevant.

See the [thinking configuration validation](docs/agent-thinking-validation.md)
for the earlier diagnosis, current results, and remaining model-quality limits.

Set `OBSIDIAN_RAG_AGENT_REPORT_DIR` to a **new directory** to retain the generated
notes, SQLite database, index report, and per-case JSON conversations; otherwise
pytest's temporary directory is used. Tests never create the default application
index. Test-owned Qdrant collections are deleted afterward, so retained SQLite
files are diagnostic artifacts and need reindexing before further vector queries.

The Agent Runtime is exposed through the Python API and `arkb ask`. MCP,
streaming, agent context-budget optimization, agent citation validation,
and conversation persistence are not implemented. [Agent Evaluation v1](evaluation/README.md)
provides a curated 40-case dataset, deterministic behavior metrics, and a runner
through the existing Runtime. Its [deterministic retrieval baselines](evaluation/deterministic-retrieval-baselines.md)
run BM25, Semantic, Hybrid, and Hybrid + Rerank on that same dataset and a pinned snapshot,
using one original-query engine request per applicable case. Fixed-pipeline
generation retains its existing budgeting and citation validation APIs. Live `match`/`read` versus indexed `search` retain the eventual
consistency described above; callers must align the tools' directory/vault with
their prepared engine.

## Count tokens

`arkb.knowledge.embeddings` provides local token counts for Ollama's
`qwen3-embedding:0.6b`. The recursive CLI path loads this tokenizer once per
invocation and reuses it for all chunk counts.

```python
from arkb.knowledge.embeddings import count_tokens, load_tokenizer

tokenizer = load_tokenizer()  # Download the tokenizer if needed, then reuse it.
title = "Permanent Notes"
body = "Develop one idea per note."

body_tokens = count_tokens(body, tokenizer=tokenizer)
input_tokens = count_tokens(
    f"{title}\n\n{body}", tokenizer=tokenizer, add_special_tokens=True
)
print(body_tokens, input_tokens)
```

The loader fetches only `tokenizer.json` (about 11.4 MB) from the public
[Qwen3-Embedding-0.6B repository](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
at revision `c54f2e6e80b2d7b7de06f51cec4959f6b3e03418`. It uses the Hugging Face
Hub cache, or an explicit `cache_dir=Path(...)`. It does not download model
weights or require PyTorch, Transformers, or an Ollama connection. After caching
that snapshot, use `load_tokenizer(local_files_only=True)` to prevent all Hub
network requests. A missing offline snapshot or download failure is reported to
the caller; there is no fallback to character counting.

`count_tokens` excludes automatically added special tokens by default, including
when measuring chunk size or overlap. Empty text has a count of zero. For a full
nonempty embedding request, measure the assembled title and body together with
`add_special_tokens=True`; Ollama appends an end marker even when the text itself
ends with that marker. Concatenated text must be measured as a whole, since its
token count need not equal the sum of the individual counts. The helper does not
split text or enforce a context limit. Keep the returned tokenizer's padding and
truncation disabled.

The tokenizer library is pinned to `0.23.2`. The loader disables the official
file's NFC normalization to match the combining-character behavior verified
with Ollama `0.33.2`. This pairing is validated for `qwen3-embedding:0.6b`; do not
assume it measures arbitrary embedding models correctly. Recheck compatibility
when changing the model or tokenizer configuration.

Regular tokenization tests run offline using a tiny tokenizer in a temporary Hub
cache. They cover loading, counts, Unicode handling, end markers, and failures.
The optional integration tests use the actual cached snapshot and the local
Ollama model, including Chinese, English, Markdown, code, emoji, and token-length
boundaries. Prepare the tokenizer with `load_tokenizer()` and make sure Ollama is
running with `qwen3-embedding:0.6b`, then run:

```sh
OBSIDIAN_RAG_RUN_MODEL_TESTS=1 .venv/bin/python -B -m pytest \
  -p no:cacheprovider -q tests/knowledge/integration/test_qwen_tokenizer.py
```

## Split notes into chunks

`arkb.knowledge.chunking` exposes `chunk_notes` and `whole_note_chunks`. It depends only on
the standard library and shared records in `arkb.knowledge.models`; it performs no I/O,
retrieval, embedding, indexing, LLM calls, or Agent reasoning. All callers use
this single canonical module.

Both functions return immutable `Chunk` objects. Existing fields remain:
`content`, `title`, `source`, `chunk_index`, `start_char`, and `end_char`. New
provenance includes `note_id`, `chunk_id`, `path` (an alias for `source`),
`heading_path`, `section_id`, `parent_id`, `section_start_char`, and
`section_end_char`. `parent_id` points to the section; `note_id` and `path` link
every section back to its note. Indexed records provide the vault-scoped
`document_id` accepted by `AgentTools.read` for parent-note expansion.

All ranges are Python character offsets into loaded `Note.content`, with
exclusive ends. They are not byte offsets or raw-file line numbers: the existing
loader removes the title line and outer whitespace. A section covers its heading
and direct body up to the next heading; `heading_path` includes ancestor titles.
Text before the first heading uses the root section and an empty heading path.
Whole-note chunks use a root section spanning the complete body.

```python
from functools import partial
from pathlib import Path

from arkb.knowledge.chunking import chunk_notes, whole_note_chunks
from arkb.knowledge.documents import load_notes
from arkb.knowledge.embeddings import count_tokens, load_tokenizer

notes = load_notes(Path("example_notes"))
tokenizer = load_tokenizer()
chunks = chunk_notes(notes, count_tokens=partial(count_tokens, tokenizer=tokenizer))
whole_chunks = whole_note_chunks(notes)
```

Splitting defaults to a 512-token body budget and up to 64 overlapping tokens.
The counter is supplied by the caller; use `count_tokens=len` for character
budgets without a tokenizer. ATX (`## Heading`) and Setext headings create hard
section boundaries, even in short notes. Paragraphs, lists (including task and
nested lists), and backtick/tilde fenced code blocks stay whole when they fit.
Code contents and leading YAML frontmatter do not create headings. Obsidian
links, embeds, tags, and callouts remain verbatim text; this is a small block
recognizer, not a complete CommonMark parser or a semantic chunker.

Oversized lists split at items first, code at lines, and prose at paragraphs,
lines, Chinese/English sentence punctuation, spaces, then characters. Split code
remains exact source text without synthesized opening or closing fences. Merged
slices are recounted because token counts are not additive. Overlap retains
whole trailing units up to its budget, may be zero, and never crosses a section
boundary. Titles and downstream formatting are outside the body budget.

Notes are processed independently, indices restart at zero, and all source text
and whitespace remain in order. Empty bodies retain their title and provenance
in one chunk. Invalid budgets and a fallback character that cannot fit raise
`ValueError`. All chunks remain in memory. The CLI retains the `recursive` mode
name for Markdown splitting; use `--chunking none` for whole-note chunks.

IDs use deterministic SHA-256 hashes. `note_id` is scoped to a source collection
and derives from its canonical relative path; renaming the note changes it.
Section identities derive from heading ancestry and same-heading sibling
occurrences. Chunk identities derive from their section, exact content, and
same-content occurrence within that section. They exclude absolute positions,
note-wide revision, and unrelated content. Moving an unchanged section or editing
another section preserves its IDs. Inserting indistinguishable duplicate
headings or chunks before existing ones may renumber those occurrences; no edit
history is tracked. Changed chunk boundaries/content produce new IDs.

`ChunkRecord.chunk_id` adds vault scope to the new chunk identity, while
`document_revision` still records the complete note revision for citations.
Legacy records without section metadata keep their original IDs and remain
readable. The next index build uses new chunking fingerprints (`markdown-v1` or
`whole-note-v2`) to publish a fresh snapshot and can reuse unchanged embedding
inputs from the cache. Historical snapshots remain readable without migration.

The standard chunking tests are offline. To check the real Qwen tokenizer's
512/64 budgets and lossless reconstruction on long multilingual examples, cache
the tokenizer first and run (no Ollama service is required for this test file):

```sh
OBSIDIAN_RAG_RUN_MODEL_TESTS=1 .venv/bin/python -B -m pytest \
  -p no:cacheprovider -q tests/knowledge/integration/test_qwen_chunking.py
```

## Prepare embedding inputs

`arkb.knowledge.embeddings` owns the text formats used by the CLI:

- `prepare_document(chunk)` returns `title + "\n\n" + content` using the versioned
  `DOCUMENT_TEMPLATE = "title-body-v1"`. Pass `EmbeddingSpec.document_template`
  explicitly when preparing indexed documents; unsupported templates raise
  `ValueError` rather than silently changing cache meaning.
- `prepare_query(question)` uses the existing Qwen retrieval instruction and
  preserves the original question after `Query:`. Supply `instruction=` to use
  an index's `IndexManifest.query_instruction`; an empty string returns the raw
  question. The default instruction is exported as `DEFAULT_QUERY_INSTRUCTION`.
- `validate_input_tokens(text, tokenizer=..., max_tokens=..., source=...)`
  returns the complete input token count or raises `ValueError` if it exceeds
  the supplied limit. Equality with the limit is allowed. The error reports
  the source and counts without including the input text.

Formatting preserves whitespace, Unicode, Markdown, and literal special markers.
Neither source paths nor chunk coordinates enter the document input. A title-only
note is valid; an entirely blank document or question is not. A whitespace-only
instruction is rejected. Use the exact prepared document text for both
`EmbeddingSpec.embedding_key(text)` and `embed_texts([text], ...)`.

```python
from arkb.knowledge.chunking import whole_note_chunks
from arkb.knowledge.embeddings import (
    prepare_document, prepare_query, validate_input_tokens,
)
from arkb.knowledge.models import Note
from arkb.knowledge.embeddings import load_tokenizer

note = Note(title="Permanent Notes", content="Develop one idea.", source="idea.md")
chunk = whole_note_chunks([note])[0]
text = prepare_document(chunk)
tokenizer = load_tokenizer(local_files_only=True)  # Requires the cached snapshot.
input_tokens = validate_input_tokens(
    text, tokenizer=tokenizer, max_tokens=8192,
    source=f"{chunk.source}, chunk {chunk.chunk_index}",
)
query = prepare_query("How should I write permanent notes?")
query_tokens = validate_input_tokens(
    query, tokenizer=tokenizer, max_tokens=8192, source="query",
)
```

The `8192` limit above is an example; use the embedding server's actual configured
per-input context limit. It is not `chunk_size`, which budgets only the body.
Validation counts the assembled text once with special tokens enabled, because
BPE counts are not additive across titles, separators, and bodies. The tokenizer
must match the model/runtime and have padding and truncation disabled. The helper
does not load tokenizers, discover model limits, truncate text, or contact Ollama.

The CLI now uses the formatting helpers with the same default inputs as before.
Local budget validation is an explicit API for callers that know the context
limit; the CLI continues to rely on Ollama's `truncate=False` for limit enforcement.
The Qwen tokenizer pairing remains restricted to the validated model described
above. Keep the original question for answer generation, without its instruction.

Regular tests use a small offline BPE tokenizer to check merged title/body
boundaries, special markers, exact limits, and invalid tokenizer settings. The
optional `tests/knowledge/integration/test_qwen_embeddings.py` compares prepared
document and query counts with Ollama's actual `prompt_eval_count`; enable it
with `OBSIDIAN_RAG_RUN_MODEL_TESTS=1` and the cached tokenizer/local model.

## Generate embeddings

`arkb.knowledge.embeddings.embed_texts` converts a list of texts into a NumPy matrix
with one vector per input, preserving order. Supply an Ollama client to select the
server and timeout. The default model is `qwen3-embedding:0.6b`.

```python
from ollama import Client
from arkb.knowledge.embeddings import embed_texts

client = Client(host="http://127.0.0.1:11434", timeout=60, trust_env=False)
vectors = embed_texts(
    ["Capture a passing thought.", "Connect related ideas."],
    client=client,
)
print(vectors.shape)
```

Use the same embedding model for documents and questions. An empty input list
returns a `(0, 0)` matrix without contacting Ollama. Blank texts, inconsistent
vector shapes, non-finite values, and zero vectors raise `ValueError`. Automatic
truncation is disabled; oversized inputs and Ollama service errors are reported
to the caller.

The automated embedding tests mock the Ollama client and require no running
model. The example above makes a real request to the local Ollama service.

`embed_texts` accepts `batch_size` (default 32), optional complete-input
`token_counts` with `max_batch_tokens`, and an expected `dimensions`, `dtype`,
and `normalization`. These are validated before requests; vector dimensions
must remain consistent across batches. `normalization="l2"` verifies unit vectors
without silently transforming them. Defaults preserve the original float64 values.
`max_retries` defaults to zero; explicit retries cover only transient transport
errors and HTTP 429/500/502/503/504 with capped exponential backoff.
Use `iter_embedding_batches` to save each successful batch before requesting the
next; failed batches never appear as successful results. Both APIs disable
truncation. Batch token limits supplement per-input context validation.

## Build context and generate an answer

`context/builder.py` owns the full context pipeline. `build_context` preserves source
identity, drops blank and duplicate hits, merges overlapping spans within the
same snapshot/document revision, and tries whole candidate chunks in retrieval
priority order. Each trial merges overlap before counting the complete messages.
If an addition does not fit, already selected evidence is retained and later
candidates are still tried. It never expands to a live note or truncates text.
This greedy policy is deterministic; it does not claim globally optimal evidence
selection or automatically infer which facts a question requires.

The shared `SearchResult` does not require a chunk or score. This context builder
currently consumes snapshot evidence with declared finite scores, chunk IDs, source spans,
and `title`, `chunk_index`, `vault_id`, `document_revision`, `index_version` metadata.
It verifies those identities before merging or citing; unsupported evidence raises
an error. These are context-consumer constraints. Citation origins retain complete
source identities. Conflicting overlap within one declared revision raises an error.

```python
from ollama import Client
from arkb.generation.models import ContextConfig
from arkb.generation.context import build_context
from arkb.generation.generate import generate_cited_answer, load_generation_counter

with Client(host="http://127.0.0.1:11434", timeout=180, trust_env=False) as client:
    counter = load_generation_counter(client=client)  # tokenizer cached after first use
    context = build_context(question, results, config=ContextConfig(), counter=counter,
                            citation_mode='structured')
    result = generate_cited_answer(context, client=client)
    answer = result.text
    structured_answer = result.to_dict()
    diagnostics = context.to_dict()
```

`BuiltContext` contains immutable message strings, evidence blocks, original hit
identities/scores, processing decisions and token accounting. `messages` returns
a fresh API payload. Python callers use `evidence_blocks` for evidence processing;
`to_dict()` exposes one evidence registry, `citation_sources`, containing text,
source positions and every contributing origin. It omits the redundant
`evidence_blocks` and `citation_map` JSON fields.
Decision ranks address the original candidate list. A `merged` event describes
consolidation, while the representative's later `selected` or `budget` event
describes the block's outcome. Sources are numbered `S1`, `S2`, etc. in final
evidence order. A `context_id` fingerprint binds the messages and source identities.
IDs are local to one context. Every budget trial includes
the citation protocol and IDs. Only evidence actually sent to the model may be
cited; merged blocks retain all contributing chunk identities.

`context/citation.py` contains immutable answer/source contracts and pure parsing,
validation and rendering functions. Generation uses Ollama JSON Schema output:
`status`, `claims` (`text` and `source_ids`), and `missing_information`. Unknown IDs,
missing references, conflicting answer states, duplicate fields and model-written
citation links fail explicitly. Truncated or malformed output is retained on
`CitedGenerationError.raw_response`; it is never silently repaired or retried.
The renderer numbers sources by first use, includes only cited sources and shows
snapshot excerpts. It does not manufacture filesystem URLs. Display prose and
metadata are escaped; raw text remains available in JSON.

`references_valid` checks source membership and protocol rules, while
`support_status` remains `not_checked`: this implementation does not judge whether
the evidence entails each claim. Character ranges are Python character positions
in loaded `Note.content`, with exclusive ends. The loader removes the title line
and trims whitespace, so these are not raw-file offsets or Markdown line numbers.
Opening a changed live file does not change the evidence from a saved snapshot.

The enforced relationship is:

```text
complete rendered input + max_output_tokens + safety_margin <= context_window
```

The built-in generation counter uses `Qwen/Qwen3.5-4B` tokenizer revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, verifies its file hash and the supported
Ollama model artifact, and reproduces the text-only system/user template with
`think=False`. This profile was checked against Ollama 0.33.2. It includes the
system prompt, JSON, titles, source names, special tokens and assistant prefix;
it does not reuse the embedding tokenizer. Only `tokenizer.json` is downloaded.
`load_generation_counter(local_files_only=True)` uses the local Hub cache.

The supported default `qwen3.5:4b` artifact has digest
`2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd`.
Unknown model artifacts or custom templates require an explicit
`GenerationCounter(model, identity, count_messages, ...)` adapter through Python.
Set `is_estimate=True` if an adapter estimates rather than counts tokens; that
limitation is retained in diagnostics. Estimates cannot guarantee a hard token
bound. Revalidate adapters when model or serving templates change.

Generation sends the built messages unchanged and applies matching `num_ctx`
and `num_predict`. When Ollama supplies its actual prompt count, generation
rejects a budget overflow or a mismatch with an exact counter instead of returning
an answer from potentially truncated input. `generate_cited_answer(context,
client=...)` is the only generation entry point. It requires a `BuiltContext`;
model selection and token budgeting happen when building that context. Use
`result.text` for Markdown and `result.to_dict()` for JSON. Validation is reused
across these output forms. `build_context` defaults to structured citations.

Without a budget, `build_context` returns prepared evidence for inspection.
Generation requires a budget whenever evidence is present.

An empty evidence set returns an insufficient-information message without
calling the generation model. If the fixed question/system prompt cannot fit,
or all evidence is excluded by the budget, generation raises
`ContextBudgetError`. `context.to_dict()` exposes `budget_exhausted` as a diagnostic
status. Answer factuality, completeness and citation support still require evaluation.

Fixed-pipeline context/generation is a Python API (see the CLI migration example
above) and an evaluation baseline. Use `ContextConfig(context_window=8192,
max_output_tokens=1024, safety_margin=128)` to set its budget. Inspect
`context.to_dict()` before generation or `answer.to_dict()` afterward. Search
never loads a generation tokenizer. Changing context settings does not require
rebuilding the index or recomputing document embeddings.

For exact supporting excerpts, build a Python context with
`citation_mode='quoted'`. Each claim then includes `quotes`,
objects containing `source_id` and verbatim `text`, with at least one quote per
cited source. Python computes the offsets; model-provided offsets are rejected.
Quotes must occur exactly once inside their cited evidence block. Missing,
paraphrased, normalized or ambiguous matches fail validation, including repeated
overlapping text. They cannot span separate evidence blocks. JSON diagnostics
include `resolved_quotes` with claim indices and Note.content character ranges;
the display places these excerpts directly below the corresponding claim.
Exact occurrence is a provenance check, not a semantic entailment check.
The default `structured` protocol and schema do not request quotes.

```sh
OBSIDIAN_RAG_RUN_MODEL_TESTS=1 .venv/bin/python -B -m pytest \
  -p no:cacheprovider -q tests/generation/integration/test_generation_counter.py
```

## Local models

Ollama serves its local API at `http://127.0.0.1:11434` by default. Start Ollama if
it is not already running, then inspect the available models:

```sh
ollama list
```

| Role | Model |
| --- | --- |
| Text embeddings | `qwen3-embedding:0.6b` |
| Answer generation | `qwen3.5:4b` |

Model files are managed separately by Ollama; syncing the Python environment
does not download them.

## Repository layout

```text
example_notes/       Markdown documents
src/arkb/            Python package
tests/               Automated tests
benchmarks/          Indexing measurement script and recorded results
pyproject.toml       Project metadata and dependencies
uv.lock              Resolved dependency versions
```

The sample documents can be shared with the repository. Virtual environments,
caches, local `config.toml`, and local `.env` files are excluded from Git. Keep
machine-specific paths and credentials in ignored local files.

## Persistent snapshots

`SQLiteStorage(Path(...))` manages versioned chunk snapshots, immutable vector
cache entries, build manifests and an active version per vault. Use its context
manager to close connections, or `read_only=True` for existing query databases.
Vectors retain the declared float32/float64 representation, are checksummed, and
are validated on write and read. Publishing verifies counts, source identities,
and every vector before atomically marking a build ready and changing its active
pointer. Failed candidates cannot replace active data. The storage schema is
versioned independently of record fingerprints; unknown versions are rejected.
SQLite files belong in a local runtime directory, never the retrieval corpus.

## Retrieval strategies

Choose modes explicitly; each primitive remains independently callable:

```python
from arkb.retrieval import BM25Retriever, RetrievalEngine, Reranker
from arkb.retrieval.qwen_rerank import QwenRerankerScorer

bm25 = BM25Retriever.from_snapshot(storage, vault_id="default", index_version=index.index_id)
engine = RetrievalEngine(semantic=semantic, bm25=bm25, candidate_k=20,
                         reranker=Reranker(QwenRerankerScorer(local_files_only=True)))
response = engine.search("rare identifier", mode="bm25", top_k=5)
response = engine.search("related meaning", mode="hybrid", top_k=5, rerank=True)
```

Here `semantic` and `index` refer to the adapters in the semantic example below.
Omit the `reranker` argument to use no reranking dependency. The engine imports
no provider SDK or model and makes no automatic mode choices or fallbacks.
`BM25Retriever`, `rrf`, `HybridRetriever`, `Reranker`, and `RerankedRetriever`
remain public for direct use and ablations. Reranking works with any retriever.

```sh
uv run --locked arkb search "ERR_CONNECTION_RESET" --mode bm25 --json
uv run --locked arkb search "How do rankings combine?" --mode hybrid --top-k 5 --json
uv run --locked --extra rerank arkb search "How do rankings combine?" \
  --mode hybrid --rerank --top-k 5 --json
```

BM25 indexes title+body with NFC/casefold Unicode word tokens. It retains
underscores, splits punctuation, and performs no stemming or CJK segmentation.
A lexical query only needs the published SQLite snapshot. Hybrid queries use
that same snapshot for both primitives and fuse ranks with RRF, never raw scores.
The fixed depths must satisfy `top_k <= candidate_k` for hybrid and
`top_k <= rerank_candidates <= candidate_k` for hybrid with reranking.

Each hit declares `score_type`: cosine similarity, BM25, RRF, or the Qwen
yes/no logit difference (`yes_no_logit_difference`). Higher is better for these implementations, but scores from different
methods/configurations are not comparable. `metadata.fusion.contributions`
retains input ranks, scores, semantics and provenance; `metadata.rerank` retains
the input rank/score and scorer identity. Ties in the new stages use stable
identity. Context and citation output preserve each origin's method and score
semantics without recalibrating them.

The only supported reranker is `Qwen/Qwen3-Reranker-0.6B`, pinned to commit
`e61197ed45024b0ed8a2d74b80b4d909f1255473`. It runs on CPU in float32, with
16 candidates per batch and a default 512-token input budget. The official
instruction and yes/no scoring suffix are reserved before query/title+body
truncation; returned evidence stays verbatim. `--reranker-max-length` adjusts
that budget, and `--reranker-cache` selects the Hugging Face cache directory.
`--offline` prevents downloads. Locally retained weights are in `.obsidian-rag/models`;
pass `--reranker-cache .obsidian-rag/models --offline` to reuse them.
See the [reranker integration results](evaluation/reranker-integration.md).
See [retrieval benchmarks](benchmarks/retrieval.md) for the shared dataset,
commands, frozen reranker evaluation, exact formulas and measured limitations.

## Semantic retrieval

`SemanticRetriever` is an independently callable primitive:

```python
from arkb.retrieval import SemanticRetriever

semantic = SemanticRetriever(embedder, index)
response = semantic.search(question, top_k=5, filters={"source": "notes/a.md"})
for result in response.results:
    print(result.source_id, result.content, result.score, result.score_type)
```

Its entire flow is query → query embedding → vector lookup → `SearchResponse`.
`Embedder.embed_query(query)` returns a vector and declares an `EmbeddingSpec`.
`VectorIndex.search(vector, top_k=..., filters=...)` returns ranked, backend-neutral
`SearchResult` objects and declares a matching spec and pinned `index_id`. The
adapter converts database hits and restores source evidence before returning.
These structural protocols require no inheritance or registry. Incompatible
embedding spaces fail before embedding; a different provider/backend can implement
the same capabilities without changing callers.

The core imports no provider SDK, storage, indexing, context builder or generation
module. It neither rewrites queries nor selects strategies, chunks documents,
builds indexes, writes embedding caches or generates answers. BM25, fusion,
hybrid and reranking are separate primitives sharing the same contract.
Tools, agents, routing and query rewriting remain outside this subsystem.

### Result contract

| Field | Meaning |
| --- | --- |
| `source_id` | Stable document identity, independent of chunks and revisions; current snapshots use `ChunkRecord.document_id` (vault + source path) |
| `source` | Source address; currently a vault-relative Markdown path |
| `content` | Source content or a verbatim snippet |
| `method` | Retrieval stage: `semantic`, `bm25`, `rrf`, `hybrid`, or `reranked` |
| `metadata` | JSON provenance; snapshots retain title, vault, document revision, index version, chunk index and Markdown section information |
| `chunk_id` | Optional chunk identity |
| `identity` | Computed deduplication key: document + chunk, then document + span, then document |
| `start_char`, `end_char` | Optional end-exclusive source span; current coordinates address `Note.content` |
| `score`, `score_type` | Optional score and its declared semantics; neither is fabricated for unscored methods |

`SearchResponse` contains the original `query`, `method`, ranked `results` tuple,
and optional `index_id`. Rank is its one-based tuple position. No matches produce an empty tuple; errors propagate
instead of becoming empty success responses. Dataclasses serialize with `asdict`.
For example, an unscored source needs only:

```python
from arkb.retrieval import SearchResult

result = SearchResult(source_id="document-id", source="notes/a.md",
                      content="An excerpt", method="grep")  # data only; no grep implementation
assert result.score is None and result.chunk_id is None
```

Today `filters` supports exact `source` equality. Blank queries, nonpositive or
noninteger `top_k`, malformed filters and unsupported filter keys fail before
embedding or search. The original query text, including whitespace, is preserved.
Scores are not normalized or compared across methods by the semantic primitive.

### Current adapters

`retrieval.qdrant.search_index(...)` keeps the existing one-shot input arguments
and now returns `SearchResponse`. It composes `OllamaQueryEmbedder` and
`QdrantSnapshotIndex`. Supply the actual runtime model spec and tokenizer;
mismatches fail before model calls. The saved query instruction and input limit
are reused. An empty snapshot validates the query without calling either backend.

To reuse a captured snapshot across calls, compose the adapters explicitly:

```python
from arkb.retrieval import SemanticRetriever
from arkb.knowledge.embeddings import OllamaQueryEmbedder
from arkb.retrieval.semantic import QdrantSnapshotIndex

# storage is an existing SQLiteStorage, preferably opened with read_only=True;
# qdrant_client, ollama_client, runtime_spec and tokenizer are caller-owned.
index = QdrantSnapshotIndex(storage, qdrant_client, vault_id="default",
                            exact=True, ef_search=None)
embedder = OllamaQueryEmbedder(
    client=ollama_client, spec=runtime_spec, tokenizer=tokenizer,
    tokenizer_identity=index.inputs["tokenizer"],
    max_input_tokens=index.inputs["max_tokens"],
    query_instruction=index.manifest.query_instruction,
)
semantic = SemanticRetriever(embedder, index)
response = semantic.search(question, top_k=5)
```

The adapter captures one READY snapshot when opened; publishing a new active
version does not change it. An explicit `index_version` can select an older
snapshot. Content comes from SQLite hit records, including Markdown section
provenance, and never from edited live files or a full vector-cache reload.
The caller owns connection lifetimes. No query path performs index lifecycle writes.

`retrieval.qdrant.search_qdrant` remains the low-level vector benchmark API,
returning `schema.VectorHit` (chunk ID and cosine score). Direct callers validate
collections using `arkb.knowledge.qdrant.check_qdrant_collection`; `QdrantSnapshotIndex`
does that itself. Source filters apply before top-k. Invalid identities, scores,
duplicate hits and missing snapshot records fail instead of returning partial data.

`score_type="cosine_similarity"` means higher is better, in [-1, 1]. Existing
float32 tolerance is preserved: Qdrant scores within 1e-5 beyond the bounds are
clipped to the bound; larger excursions fail. Other scores are unchanged.
Returned ties are ordered by chunk ID. `exact` (default false) and `ef_search`
are explicit adapter settings. Exact lookup on a fixed snapshot is the reproducible
baseline; ANN, embedding-provider behavior, and ties at the top-k cutoff can still
affect repeatability. The primitive does not add randomness or hidden decisions.

Python callers now use `response.results` and direct result fields such as
`result.content` and `result.source_id`; snapshot revision/version are in
`result.metadata`. The match/search CLI JSON serializes `SearchResponse` directly (`query`,
`method`, `index_id`, `results`), including each result using this contract. `build_context` consumes
`response.results`; answer generation remains an application choice.

## Build an index snapshot

`indexing.build_index` accepts a complete list of loaded notes, a resolved
`EmbeddingSpec`, vault ID, matching tokenizer, active context limit, and Ollama
client, plus an explicit `qdrant_client` and `qdrant_config=QdrantConfig(...)`
from `arkb.knowledge.indexing`. It validates inputs, chunks notes, caches
successful embedding batches, verifies the SQLite and Qdrant candidates, then publishes.
`QdrantConfig` owns the endpoint, HNSW settings, readiness timeout, defaults and
validation. The CLI uses the same configuration. For direct collection operations,
pass it as `QdrantIndex(..., config=config)`; `wait_ready(expected_count=...)` uses
that configuration. The builder writes `kind='qdrant'` itself. Existing saved
metadata with omitted default settings still reuses its published snapshot.

The report includes the published manifest and counts of unique embedded/cached
inputs. Duplicate text occurrences share vectors but retain separate chunk IDs.
A later batch failure leaves previous published snapshots available and earlier
successful batches reusable. Model artifact discovery belongs to the caller;
changing the declared model revision changes cache identity.

## Incremental updates and recovery

Repeat `index` to scan for additions, edits, renames, and deletions. Unchanged
corpus/configuration reuses the published version and makes no document embedding
requests. `--force` creates a fresh snapshot while reusing compatible cached
vectors. Reports count added/modified/deleted documents; renames are delete/add
operations and can reuse vectors when the final input text is unchanged.
Incremental embedding currently assembles a complete candidate snapshot; it does
not claim to mutate only changed vector-database records.

A process-level advisory lock serializes builds for a SQLite database. Interrupted
building snapshots are marked failed by the next writer, and completed embedding
batches remain reusable. Locks are released by the OS when a process exits.
Runtime verifies the flat Markdown scope is stable while reading and binds each
vault to its source directory, so a different or failed scan cannot silently
replace its corpus. An intentionally emptied directory publishes an empty index.
Historical snapshots are retained. The storage API does not expose snapshot deletion.

## Qdrant backend

`indexing.QdrantIndex` creates, writes, verifies, and removes collections using
external vectors, with one collection per candidate. Queries live in
`retrieval.qdrant.search_qdrant`. Collection metadata binds the embedding spec
and vault; opening incompatible collections fails without changing their data.
Payload indexes for vault, embedding spec and source are created before ingestion.
Chunk SHA-256 IDs map deterministically to UUID point IDs; full identities stay in
payload and are verified on retrieval. Qdrant stores cosine vectors as float32,
so snapshot verification allows small rounding differences from cached float64 vectors.
`create=True` never recreates an existing collection. Point writes wait for
completion. Failed-candidate cleanup removes only owned collections. Text remains
in SQLite, and this collection handler never runs an embedding model.

Regular backend tests use Qdrant Local for API behavior only. Real server tests
are enabled with `OBSIDIAN_RAG_QDRANT_URL=http://127.0.0.1:6333`; they create and
remove uniquely named test collections. Qdrant Server/client 1.19 are the validated
pair; ANN and payload-index performance are not inferred from Local Mode tests.

Run Qdrant Server separately, for example with the validated
`qdrant/qdrant:v1.19.0` Docker image.

The endpoint and collection are saved with the snapshot; queries open that exact
collection and fetch only the returned source records from SQLite. Set
`QDRANT_API_KEY` when needed; credentials are never saved in manifests. Search and ask
accept `--qdrant-url` for an explicitly restored/moved server.

`--hnsw-m`, `--ef-construct`, `--indexing-threshold` (Qdrant's KB threshold), and
`--index-timeout` configure construction. `--full-scan-threshold` controls the
query planner in KB and must be at least 10. Small collections can be query-ready
without HNSW. `--require-hnsw` waits until every vector is indexed and fails on
timeout; use a suitable positive threshold when explicitly testing ANN.
Before publication the builder verifies every point ID, payload and vector,
then checks optimizer health/readiness. SQLite's active-version transaction is
the publication authority; no cross-database atomic transaction is assumed.

A failed build leaves the previous active collection unchanged. The next writer
cleans only owned FAILED collections on the same configured server and reuses
cached embeddings. Historical READY collections remain available to pinned
queries and rollback workflows; they are not automatically deleted. Record
metadata includes observed point/index counts. Changing only index parameters
rebuilds the search projection without recomputing compatible embeddings.

## Evaluate retrieval and evidence

`evaluation.retrieval.compare_retrieval` compares Qdrant exact and ANN modes on the same
query vectors, using a supplied Qdrant search callable. Exact mode is the reference. It reports neighbor Recall@k separately from
source-group recall and union coverage of labeled sections. Section anchors must
use `body_start_char`/`body_end_char` in loaded `Note.content`; raw-file offsets
are rejected. Empty reference sets produce an undefined recall, not a perfect
score. Exact ties and float32 rounding can change rank order, so raw IDs/scores
are retained. No generated-answer accuracy is inferred from these metrics.

```sh
uv run --locked python -m arkb.evaluation.retrieval ann \
  --db .obsidian-rag/index.sqlite --cases path/to/cases.jsonl \
  --output path/to/new-evaluation --offline --top-k 2
```

The runner captures an active snapshot, verifies the actual embedding model and
Qdrant data, embeds each question once, and compares Qdrant exact and ANN. It writes frozen cases, raw results,
metrics, source/configuration hashes and runtime metadata into a new directory;
existing evaluation directories are never overwritten. Source hashes cover all Python
modules recursively, keyed by their paths relative to the package root. Search timing includes
backend I/O but excludes embedding and snapshot load. It reports stored vector
bytes and SQLite file sizes, not total process or server memory. Build reports
include `build_seconds` measured inside the writer lock (source scan excluded).

For a controlled HNSW experiment on a small corpus, rebuild with
`--indexing-threshold 1 --full-scan-threshold 10 --require-hnsw`. The full-scan
threshold affects the query planner; simply requesting ANN does not prove the
server used a graph for a small collection. No large-scale latency claim should
be made from the bundled small corpus. Keep evaluation outputs outside note
folders. Real end-to-end lifecycle tests require both `OBSIDIAN_RAG_RUN_MODEL_TESTS=1` and
`OBSIDIAN_RAG_QDRANT_URL`; they check separate-process queries and incremental
edits/deletes against real services while cleaning their test collections.


Use `python -m arkb.evaluation.generation` to evaluate context by default, or
add `--context` to the ANN runner. Both use each frozen Qdrant exact candidate list. `evaluation.generation.evaluate_context` reports the packed evidence
under the configured generation budget. Coverage retention compares the packed
evidence with the original hits, without rendering historical prompt variants.
`context_results.jsonl` retains final messages and provenance; `metrics.json`
includes input tokens, block counts, duplicate source-span fractions, section
coverage/retention and build time. These are context metrics, not answer scores.

```sh
uv run --locked python -m arkb.evaluation.generation \
  --db .obsidian-rag/index.sqlite --cases path/to/cases.jsonl \
  --output path/to/new-context-evaluation --offline --top-k 2 --context \
  --context-window 8192 --max-output-tokens 1024
```

Add `--citations` to generate structured answers once on the same frozen Qdrant exact
hits. `citation_results.jsonl` includes final messages, response schemas, source
registries, raw model output, validation failures, token usage and generation
time. Existing output directories are rejected. This can be combined with
`--context`; it does not modify the index or document vectors.

`evaluation.metrics.citation_statistics` measures ID validity and the fraction of emitted
claims having a known reference. These metrics do not identify omitted facts or
prove support. Optional review data supplies an identified `reviewer`, one
`claim_support` label per claim (`supported`, `partial`, `contradicted`,
`insufficient`, or null), and optional boolean `answer_correct` / `answer_complete`.
Support is judged against the cited sources jointly. The supported-claim rate
uses reviewed claims only and reports review coverage separately. Zero
denominators and absent semantic reviews remain null, never perfect scores.
Summaries include failed cases and each metric's number of defined cases.

For indexing costs, run the following with a new output path:

```sh
uv run --locked python -B benchmarks/profile_indexing.py --output /tmp/indexing-overhead-new.json
```

The [measurement report](benchmarks/indexing-overhead.md) records the baseline,
method and limits; its timings exclude real model inference and Qdrant Server.
