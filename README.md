# ARKB

Local retrieval over Markdown notes using Python, Qdrant, and Ollama.

The repository includes a command-line interface, a Markdown note loader, an
Ollama embedding client function, cosine similarity search, answer generation,
and sample Markdown notes in `example_notes/`.

## Python modules

The package separates knowledge preparation, semantic retrieval, context
construction, and answer generation. The CLI composes these stages; generation
is an independent capability. Embeddings, embedding tokenization, storage, and
data records are shared by the stages that use them.

| File in `src/arkb/` | Responsibility |
| --- | --- |
| `__init__.py` | Package marker |
| `indexing/loaders.py` | Markdown loading and consistent source-directory scans |
| `chunking.py` | Shared Markdown block splitting, section provenance, and size/overlap policy |
| `indexing/chunking.py` | Compatibility imports for the shared chunker |
| `indexing/index.py` | Complete/incremental builds, Qdrant writes, HNSW readiness, verification, and failed-candidate cleanup |
| `retrieval/semantic.py` | Qdrant exact/ANN search, query embedding, and snapshot evidence lookup |
| `context/builder.py` | Evidence provenance, deduplication, overlap merging, budgets, and message rendering |
| `context/citation.py` | Citation contracts, strict parsing, reference/quote validation, and rendering |
| `embeddings.py` | Document/query input preparation, token budgets, model identity, batching, retries, and vector validation |
| `tokenization.py` | Pinned embedding tokenizer loading, token counting, and tokenizer fingerprints |
| `schema.py` | Shared notes, chunks, search results, embedding specs, manifests, and stable identity rules |
| `storage.py` | SQLite cache/snapshots, build states, locks, atomic publication, and shared Qdrant connection/configuration checks |
| `generation.py` | Grounded answer generation and the Qwen generation-tokenizer adapter |
| `evaluation.py` | Qdrant exact/ANN comparisons, context/citation evaluation and reproducible run artifacts |
| `cli.py` | Command arguments, resource setup, workflow calls, and output |

Use `arkb.indexing.build_index`, `arkb.retrieval.search_index`, and
`arkb.context.build_context` as the main stage APIs. Direct vector search remains
available through `arkb.retrieval.search_qdrant`;
collection writes and lifecycle operations use `arkb.indexing.QdrantIndex`, with
settings supplied by `arkb.indexing.QdrantConfig`.
`Note`, `Chunk`, and `SearchResult` are defined in `arkb.schema`; the loader,
chunker, and retrieval package also expose their respective record types.

The package and command are named `arkb`. Run `uv sync --locked` after updating.
Only `index`, `query`, and `status` are supported; the old bare-question command,
`obsidian-rag` alias, `retrieve`, and `search_numpy` APIs have been removed.
Qdrant is the only search backend. NumPy remains a dependency for embedding
validation, vector serialization, snapshot verification, and evaluation statistics.

The default `.obsidian-rag/index.sqlite` path, persisted identity namespaces,
Qdrant collection names/ownership, and `OBSIDIAN_RAG_*` test variables retain
their existing names. Existing Qdrant indexes remain compatible. Queries against
old NumPy indexes explicitly require `arkb index` to rebuild into Qdrant. Keep the
same `--db`, `--vault-id`, note directory, and embedding settings to reuse compatible
cached vectors. The old active snapshot is replaced only after successful publication;
rebuilding does not delete historical snapshots or the embedding cache.

Functional tests remain in `tests/test_<module>.py`. Real-service tests live in
`tests/integration/`; generation counting checks also remain in
`tests/test_context.py`. Regular tests need no network or running services.

## Requirements

- Python 3.13
- uv
- Ollama, running locally for model operations
- Qdrant Server, running for indexing and queries

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

## Index notes and ask a question

Start Ollama with the models below and Qdrant Server, then build an index:

```sh
uv run --locked arkb index --notes-dir example_notes --qdrant-url http://127.0.0.1:6333
uv run --locked arkb query "Why combine lexical and vector retrieval?"
uv run --locked arkb query "How does Reciprocal Rank Fusion combine rankings?" --top-k 2 --json
uv run --locked arkb status
```

`index` reads Markdown files directly in the selected directory, splits long notes,
embeds uncached inputs, writes a Qdrant collection, and publishes a snapshot.
`query` embeds only the question and reads evidence from that saved snapshot.
Its default output is a structured-citation answer rendered as text; `--json`
returns retrieved evidence without calling the generation model.

Repeat `index` after editing notes. Relative paths resolve from the current
working directory; subdirectories are not scanned. The default database is
`.obsidian-rag/index.sqlite`; use `--db` and `--vault-id` consistently across commands.
`index` prints a JSON build report. `status` reads saved manifests without loading
a tokenizer or contacting Ollama. A missing index or mismatched model/tokenizer
produces an error instead of rebuilding during a query.

| Command | Option | Default | Purpose |
| --- | --- | --- | --- |
| index | `--notes-dir` | `example_notes` | Directory containing Markdown notes |
| index | `--qdrant-url` | `http://127.0.0.1:6333` | Qdrant endpoint saved with the snapshot |
| index | `--embedding-model` | `qwen3-embedding:0.6b` | Embedding model |
| index | `--chunking` | `recursive` | Markdown sections with recursive overflow splitting, or whole-note `none` |
| index | `--chunk-size` / `--chunk-overlap` | `512` / `64` | Body token limit and target overlap |
| index | `--context-length` | `8192` | Full embedding-input limit, including title and special tokens |
| query | `--top-k` | `2` | Maximum candidates before context processing |
| query | `--source` | unset | Filter by an exact saved source path |
| query | `--exact` | off | Request Qdrant exact search |
| query | `--generation-model` | `qwen3.5:4b` | Answer model |
| query | `--context-window` | `8192` | Generation window, also passed as `num_ctx` |
| query | `--max-output-tokens` | `1024` | Output reserve, also passed as `num_predict` |
| query | `--context-safety-margin` | `128` | Additional reserved space |
| query | `--show-context` | off | Inspect final messages, evidence and budget without generation |
| query | `--citation-mode` | `structured` | Validated source IDs; `quoted` also requires exact excerpts |
| query | `--answer-json` | off | Answer, used sources, raw response and validation |
| index/query | `--host` | `http://127.0.0.1:11434` | Ollama endpoint |
| index/query | `--timeout` | `180` | Request timeout in seconds |
| index/query | `--tokenizer-cache` | Hub default | Tokenizer cache directory |
| index/query | `--offline` | off | Prevent tokenizer downloads; local model/server calls still occur |

The CLI validates the supported Qwen 0.6b embedding tokenizer/model pairing.
The installed model supplies its digest, dimensions and maximum context length.
`--context-length` cannot exceed that limit and is sent as `num_ctx` for document
and query embeddings. `--batch-size`, `--max-batch-tokens`, and `--max-retries`
control embedding work; `--query-instruction ''` saves a raw-query configuration.
Both chunking modes count the complete embedding input and keep `truncate=False`.
In recursive mode, require `chunk_size > 0` and `0 <= chunk_overlap < chunk_size`.
Use `--offline` after caching tokenizers; a missing tokenizer is an error.

`--json`, `--show-context`, and `--answer-json` are mutually exclusive.
An empty source directory publishes an empty index. Source scan errors leave the
previous active index unchanged. Exit codes are `0` for success, `1` for runtime
failure, and `2` for invalid arguments; errors go to standard error.

Use `arkb index --help`, `arkb query --help`, and `arkb status --help` for all options.
The same commands are available through `python -m arkb.cli`.
Regular CLI tests use Qdrant Local and replace Ollama/tokenizer boundaries, so
no running services or network are required.

## Read notes

`arkb.indexing.loaders.load_notes` accepts a directory as a `pathlib.Path` and returns
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

## Count tokens

`arkb.tokenization` provides local token counts for Ollama's
`qwen3-embedding:0.6b`. The recursive CLI path loads this tokenizer once per
invocation and reuses it for all chunk counts.

```python
from arkb.tokenization import count_tokens, load_tokenizer

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
  -p no:cacheprovider -q tests/integration/test_qwen_tokenizer.py
```

## Split notes into chunks

`arkb.chunking` exposes `chunk_notes` and `whole_note_chunks`. It depends only on
the standard library and shared records in `arkb.schema`; it performs no I/O,
retrieval, embedding, indexing, LLM calls, or Agent reasoning. The previous
`arkb.indexing.chunking` import path remains available for compatibility.

Both functions return immutable `Chunk` objects. Existing fields remain:
`content`, `title`, `source`, `chunk_index`, `start_char`, and `end_char`. New
provenance includes `note_id`, `chunk_id`, `path` (an alias for `source`),
`heading_path`, `section_id`, `parent_id`, `section_start_char`, and
`section_end_char`. `parent_id` points to the section; `note_id` and `path` link
every section back to its note for future `read_note` or parent-note expansion.

All ranges are Python character offsets into loaded `Note.content`, with
exclusive ends. They are not byte offsets or raw-file line numbers: the existing
loader removes the title line and outer whitespace. A section covers its heading
and direct body up to the next heading; `heading_path` includes ancestor titles.
Text before the first heading uses the root section and an empty heading path.
Whole-note chunks use a root section spanning the complete body.

```python
from functools import partial
from pathlib import Path

from arkb.chunking import chunk_notes, whole_note_chunks
from arkb.indexing.loaders import load_notes
from arkb.tokenization import count_tokens, load_tokenizer

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
  -p no:cacheprovider -q tests/integration/test_qwen_chunking.py
```

## Prepare embedding inputs

`arkb.embeddings` owns the text formats used by the CLI:

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
from arkb.indexing.chunking import whole_note_chunks
from arkb.embeddings import (
    prepare_document, prepare_query, validate_input_tokens,
)
from arkb.indexing.loaders import Note
from arkb.tokenization import load_tokenizer

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
optional `tests/integration/test_qwen_embeddings.py` compares prepared
document and query counts with Ollama's actual `prompt_eval_count`; enable it
with `OBSIDIAN_RAG_RUN_MODEL_TESTS=1` and the cached tokenizer/local model.

## Generate embeddings

`arkb.embeddings.embed_texts` converts a list of texts into a NumPy matrix
with one vector per input, preserving order. Supply an Ollama client to select the
server and timeout. The default model is `qwen3-embedding:0.6b`.

```python
from ollama import Client
from arkb.embeddings import embed_texts

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

`SearchResult` requires a matching `ChunkRecord` and a nonblank `index_version`.
Citation origins also require complete source identities.
Conflicting overlap within one declared revision raises an error.

```python
from ollama import Client
from arkb.context import ContextConfig, build_context
from arkb.generation import generate_cited_answer, load_generation_counter

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
`ContextBudgetError`. `--show-context` exposes `budget_exhausted` as a diagnostic
status. The query command defaults to structured citations.
Answer factuality, completeness and citation support still require evaluation.

The query command accepts the context options listed above:

```sh
uv run --locked arkb query "What does chunking preserve?" --offline --show-context
uv run --locked arkb query "What does chunking preserve?" --offline --answer-json
uv run --locked arkb query "What does chunking preserve?" --offline \
  --context-window 8192 --max-output-tokens 1024 --context-safety-margin 128
```

`query --json` remains retrieval-only and does not load a generation tokenizer.
`--json`, `--show-context` and `--answer-json` are mutually exclusive;
`--answer-json` requires structured or quoted citations. Changing context settings does not
require rebuilding the index or recomputing document embeddings.

For exact supporting excerpts, select `--citation-mode quoted`, or build a
Python context with `citation_mode='quoted'`. Each claim then includes `quotes`,
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
  -p no:cacheprovider -q tests/test_context.py
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

## Vector search

`retrieval.search_qdrant` returns `schema.VectorHit`: a stable chunk ID and a
cosine score (larger is better). Source filtering happens before top-k selection.
Qdrant supports exact and ANN queries; callers open and validate the collection
with `arkb.storage.check_qdrant_collection` before direct searches.
`retrieval.search_index` handles validation and resolves hits to snapshot evidence
for application queries. No SQLite/NumPy search fallback is available.

`retrieval.search_index` queries a captured READY snapshot and embeds only the
question. Supply the actual model spec and tokenizer; mismatches fail before
model calls. The saved query instruction and context limit are reused. Explicit
`index_version` pins a request across concurrent publication. Returned chunks
come from the snapshot, not potentially edited source files; unknown vector hits
and invalid scores are rejected. Supply the Qdrant client explicitly.

## Build an index snapshot

`indexing.build_index` accepts a complete list of loaded notes, a resolved
`EmbeddingSpec`, vault ID, matching tokenizer, active context limit, and Ollama
client, plus an explicit `qdrant_client` and `qdrant_config=QdrantConfig(...)`
from `arkb.indexing`. It validates inputs, chunks notes, caches
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
The CLI verifies the flat Markdown scope is stable while reading and binds each
vault to its source directory, so a different or failed scan cannot silently
replace its corpus. An intentionally emptied directory publishes an empty index.
Historical snapshots are retained. The storage API does not expose snapshot deletion.

## Qdrant backend

`indexing.QdrantIndex` creates, writes, verifies, and removes collections using
external vectors, with one collection per candidate. Queries live in
`retrieval.search_qdrant`. Collection metadata binds the embedding spec
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
`QDRANT_API_KEY` when needed; credentials are never saved in manifests. The query
command accepts `--qdrant-url` for an explicitly restored/moved server.

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

`evaluation.compare_retrieval` compares Qdrant exact and ANN modes on the same
query vectors, using a supplied Qdrant search callable. Exact mode is the reference. It reports neighbor Recall@k separately from
source-group recall and union coverage of labeled sections. Section anchors must
use `body_start_char`/`body_end_char` in loaded `Note.content`; raw-file offsets
are rejected. Empty reference sets produce an undefined recall, not a perfect
score. Exact ties and float32 rounding can change rank order, so raw IDs/scores
are retained. No generated-answer accuracy is inferred from these metrics.

```sh
uv run --locked python -m arkb.evaluation \
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


Add `--context` to evaluate the production context builder on each frozen Qdrant
exact candidate list. `evaluation.evaluate_context` reports the packed evidence
under the configured generation budget. Coverage retention compares the packed
evidence with the original hits, without rendering historical prompt variants.
`context_results.jsonl` retains final messages and provenance; `metrics.json`
includes input tokens, block counts, duplicate source-span fractions, section
coverage/retention and build time. These are context metrics, not answer scores.

```sh
uv run --locked python -m arkb.evaluation \
  --db .obsidian-rag/index.sqlite --cases path/to/cases.jsonl \
  --output path/to/new-context-evaluation --offline --top-k 2 --context \
  --context-window 8192 --max-output-tokens 1024
```

Add `--citations` to generate structured answers once on the same frozen Qdrant exact
hits. `citation_results.jsonl` includes final messages, response schemas, source
registries, raw model output, validation failures, token usage and generation
time. Existing output directories are rejected. This can be combined with
`--context`; it does not modify the index or document vectors.

`evaluation.citation_statistics` measures ID validity and the fraction of emitted
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
