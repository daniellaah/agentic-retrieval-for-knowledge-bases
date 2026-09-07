# Obsidian RAG

Local retrieval over Markdown notes using Python, NumPy, and Ollama.

The repository includes a command-line interface, a Markdown note loader, an
Ollama embedding client function, cosine similarity search, answer generation,
and sample Markdown notes in `example_notes/`.

## Python modules

The package stays flat: one module per stage, with both retrieval backends in
`retrieval.py` and all embedding preparation and model calls in `embeddings.py`.

| File in `src/obsidian_rag/` | Responsibility |
| --- | --- |
| `__init__.py` | Package marker |
| `loaders.py` | Note records, Markdown loading, and consistent source-directory scans |
| `chunking.py` | Whole-note and recursive splitting with source positions and section metadata |
| `tokenization.py` | Pinned tokenizer loading, token counting, and tokenizer fingerprints |
| `embeddings.py` | Document/query input preparation, token budgets, model identity, batching, retries, and vector validation |
| `schema.py` | Shared embedding specs, chunk records, manifests, search hits, and stable identity rules |
| `storage.py` | SQLite embedding cache, immutable snapshots, build states, writer locks, and atomic publication |
| `indexing.py` | Complete/incremental builds, Qdrant writes, HNSW readiness, verification, and failed-candidate cleanup |
| `retrieval.py` | NumPy exact search, Qdrant exact/ANN search, query embedding, and snapshot evidence lookup |
| `context.py` | Evidence provenance, deduplication, overlap merging, generation token counting, budgets, and message rendering |
| `generation.py` | Grounded answer generation from the final budgeted messages |
| `evaluation.py` | Fixed-case backend comparisons, evidence metrics, and reproducible run artifacts |
| `browsecomp.py` | Pinned BrowseComp-Plus downloads, local records, and reversible document identities |
| `benchmark.py` | Frozen benchmark preparation, index/run/judge/report/export workflows and CLI |
| `judging.py` | Versioned answer-judge prompt, verdict parsing, and separate judge failure records |
| `cli.py` | Command arguments, resource setup, workflow calls, and output |

Each functional module has a corresponding `tests/test_<module>.py` file.
Real-service tests live in `tests/integration/`: `test_qwen_tokenizer.py`,
`test_qwen_chunking.py`, `test_qwen_embeddings.py`, `test_qdrant_retrieval.py`,
`test_qdrant_indexing.py`, and `test_indexing_lifecycle.py`. The context-specific
real-model counting checks remain in `tests/test_context.py` alongside its unit tests.

Python imports changed with this refactor: `notes` became `loaders`,
`index_schema` became `schema`, and `embedding_inputs` merged into `embeddings`.
The former vector-store adapters were removed. Use `retrieval.search_numpy` or
`retrieval.search_qdrant` for search, and `indexing.QdrantIndex` for collection
writes and lifecycle operations. `build_index` and `search_index` select their
backend from snapshot settings; their former `vector_store` injection argument
was removed. Evaluation backends now supply `(search_callable, exact_flag)` pairs.

CLI commands, input templates, chunk/cache identity rules, SQLite schema, and
Qdrant collection metadata remain compatible. Existing indexes can be reopened
without migrating data or embedding the corpus again.

## Requirements

- Python 3.13
- uv
- Ollama, running locally for model operations

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
uv run --locked python -c "import obsidian_rag, numpy, ollama; print('Imports OK')"
uv run --locked pytest --version
```

## Ask a question

After syncing the environment and starting Ollama with the models below
available, run this command from the repository root:

```sh
uv run --locked obsidian-rag "When should temporary notes be processed and deleted?"
```

The command reads `example_notes/`, splits long notes into chunks, embeds the
chunks and question, retrieves the two most similar chunks, and prints the answer. The embedding query
uses a Qwen retrieval instruction; answer generation receives the original
question. Each invocation reads, splits, and embeds the notes again, keeping
chunks and vectors in memory for that invocation.

Specify another note directory or result count with:

```sh
uv run --locked obsidian-rag "How do literature and permanent notes differ?" \
  --notes-dir example_notes --top-k 2
```

Relative note paths are resolved from the current working directory. The loader
reads Markdown files directly in that directory without visiting subdirectories.

| Option | Default | Purpose |
| --- | --- | --- |
| `--notes-dir` | `example_notes` | Directory containing Markdown notes |
| `--top-k` | `2` | Maximum number of retrieved candidates before context processing |
| `--embedding-model` | `qwen3-embedding:0.6b` | Qwen embedding model |
| `--generation-model` | `qwen3.5:4b` | Model used to generate the answer |
| `--host` | `http://127.0.0.1:11434` | Ollama server URL |
| `--timeout` | `180` | Ollama request timeout in seconds |
| `--chunking` | `recursive` | `recursive` for B1, `none` for whole-note B0 |
| `--chunk-size` | `512` | Maximum body tokens per chunk |
| `--chunk-overlap` | `64` | Target overlap in body tokens |
| `--tokenizer-cache` | Hub default | Optional tokenizer cache directory |
| `--offline` | off | Prevent embedding/generation tokenizer Hub requests; Ollama is still used |
| `--context-window` | `8192` | Generation window, also passed as `num_ctx` |
| `--max-output-tokens` | `1024` | Output reserve, also passed as `num_predict` |
| `--context-safety-margin` | `128` | Extra space reserved outside the measured input |
| `--show-context` | off | Print final messages, evidence identities and budget diagnostics without generating |
| `--citation-mode` | `structured` | Validate citations; `quoted` adds exact excerpts, `legacy` uses filename prompting |
| `--answer-json` | off | Print answer, used sources, raw response and citation diagnostics |

Recursive mode currently supports the validated `qwen3-embedding:0.6b` tokenizer
pairing. For another embedding model, use `--chunking none`. In recursive mode,
`--chunk-size` must be positive and `0 <= --chunk-overlap < --chunk-size`.
Use `--offline` after the tokenizer is cached; missing cache files are reported.
Whole-note mode skips the embedding tokenizer and ignores chunk budget options.
Answer generation and `--show-context` still load the generation tokenizer.
Embedding requests keep `truncate=False`, so Ollama rejects full inputs exceeding
its active context limit rather than silently truncating titles or text.

`--top-k` must be a positive integer and `--timeout` a positive finite number.
Answers go to standard output; errors go to standard error. Exit codes are `0`
for success, `1` for a runtime failure, and `2` for invalid arguments. An empty
note directory reports an error without contacting Ollama.

Show all options with `uv run --locked obsidian-rag --help`. The same interface
is available through `uv run --locked python -m obsidian_rag.cli`.

CLI tests exercise loading, tokenization, chunking, embedding conversion,
retrieval, context building, and generation together. External tokenizer loading,
generation counting and Ollama are replaced at their boundaries in CLI unit tests;
context tests independently validate the real counter. Regular tests need no network.

## Read notes

`obsidian_rag.loaders.load_notes` accepts a directory as a `pathlib.Path` and returns
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
uv run --locked pytest -q
```

## Count tokens

`obsidian_rag.tokenization` provides local token counts for Ollama's
`qwen3-embedding:0.6b`. The recursive CLI path loads this tokenizer once per
invocation and reuses it for all chunk counts.

```python
from obsidian_rag.tokenization import count_tokens, load_tokenizer

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

`obsidian_rag.chunking` exposes `chunk_notes` and `whole_note_chunks`. Both return
immutable `Chunk` objects with `content`, `title`, `source`, `chunk_index`,
`start_char`, and `end_char`. Positions are Python character offsets into the
loaded `Note.content`, with an exclusive end; the loader has already removed the
title line and outer whitespace, so these are not raw-file line numbers.

```python
from functools import partial
from pathlib import Path

from obsidian_rag.chunking import chunk_notes, whole_note_chunks
from obsidian_rag.loaders import load_notes
from obsidian_rag.tokenization import count_tokens, load_tokenizer

notes = load_notes(Path("example_notes"))
tokenizer = load_tokenizer()
chunks = chunk_notes(notes, count_tokens=partial(count_tokens, tokenizer=tokenizer))
whole_chunks = whole_note_chunks(notes)  # B0: one whole note per chunk.
```

Recursive splitting defaults to a 512-token body budget and up to 64 overlapping
tokens. It prefers paragraphs, lines, sentence punctuation in Chinese/English,
then spaces, falling back to character boundaries for oversized units. Smaller
units are merged by recounting the combined text. Overlap retains whole trailing
units, so it may be below the target or zero. Titles and embedding end markers
are outside the body budget and need to be counted with the final input.

Short notes remain whole. Notes are processed independently, indices restart at
zero, and repeated passages keep their distinct positions. Text, whitespace, and
Markdown markers are preserved verbatim; this baseline does not interpret code
fences or table structure. Empty bodies retain their title and source in one
chunk. Invalid budgets and a fallback character that cannot fit raise `ValueError`.
All chunks remain in memory. The CLI uses recursive chunks by default; select
`--chunking none` to use whole-note chunks.

The standard chunking tests are offline. To check the real Qwen tokenizer's
512/64 budgets and lossless reconstruction on long multilingual examples, cache
the tokenizer first and run (no Ollama service is required for this test file):

```sh
OBSIDIAN_RAG_RUN_MODEL_TESTS=1 .venv/bin/python -B -m pytest \
  -p no:cacheprovider -q tests/integration/test_qwen_chunking.py
```

## Prepare embedding inputs

`obsidian_rag.embeddings` owns the text formats used by the CLI:

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
from obsidian_rag.chunking import whole_note_chunks
from obsidian_rag.embeddings import (
    prepare_document, prepare_query, validate_input_tokens,
)
from obsidian_rag.loaders import Note
from obsidian_rag.tokenization import load_tokenizer

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

`obsidian_rag.embeddings.embed_texts` converts a list of texts into a NumPy matrix
with one vector per input, preserving order. Supply an Ollama client to select the
server and timeout. The default model is `qwen3-embedding:0.6b`.

```python
from ollama import Client
from obsidian_rag.embeddings import embed_texts

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

## Retrieve chunks

`obsidian_rag.retrieval.retrieve` ranks chunks by cosine similarity and returns
`SearchResult` objects containing the original `chunk` and a numeric `score`.
Each row of the chunk matrix must correspond to the chunk at the same index. Pass
a single query vector with the same dimension, using the same embedding model
for chunks and queries.

```python
from pathlib import Path

from ollama import Client

from functools import partial

from obsidian_rag.chunking import chunk_notes
from obsidian_rag.embeddings import prepare_document, prepare_query
from obsidian_rag.tokenization import count_tokens, load_tokenizer
from obsidian_rag.embeddings import embed_texts
from obsidian_rag.loaders import load_notes
from obsidian_rag.retrieval import retrieve

client = Client(host="http://127.0.0.1:11434", timeout=60, trust_env=False)
notes = load_notes(Path("example_notes"))
tokenizer = load_tokenizer()
chunks = chunk_notes(notes, count_tokens=partial(count_tokens, tokenizer=tokenizer))
chunk_vectors = embed_texts(
    [prepare_document(chunk) for chunk in chunks],
    client=client,
)
question = "When should temporary notes be processed and deleted?"
query = prepare_query(question)
query_vector = embed_texts([query], client=client)[0]

for result in retrieve(chunks, chunk_vectors, query_vector, top_k=2):
    print(f"{result.score:.4f} {result.chunk.source}: {result.chunk.title}")
```

The query includes a task instruction in the format recommended by the
[Qwen3 Embedding model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).
Note text is embedded without that instruction. `embed_texts` passes its inputs
to the model unchanged, so the caller prepares the query text.

`top_k` defaults to 2 and must be a positive integer. Results are sorted by
score from highest to lowest; ties preserve input order. If fewer chunks are
available, all are returned. An empty collection with a zero-row matrix returns
an empty list. Incompatible shapes, non-finite values, zero vectors, and
non-finite vector norms raise `ValueError`. Input vectors are left unchanged.

Retrieval tests use small, fixed vectors and require no model or network access.
The example above calls Ollama. A similarity score ranks relevance; it is not a
probability or proof that a note contains an answer. Answer generation must still
check whether the retrieved text supports a response.

## Build context and generate an answer

`context.py` owns the full context pipeline. `build_context` preserves source
identity, drops blank and duplicate hits, merges overlapping spans within the
same snapshot/document revision, and tries whole candidate chunks in retrieval
priority order. Each trial merges overlap before counting the complete messages.
If an addition does not fit, already selected evidence is retained and later
candidates are still tried. It never expands to a live note or truncates text.
This greedy policy is deterministic; it does not claim globally optimal evidence
selection or automatically infer which facts a question requires.

Persistent `SearchResult` objects retain their `ChunkRecord` and `index_version`.
The legacy CLI binds its in-memory hits to the exact loaded corpus. Direct
`retrieve` callers without records retain explicitly unknown provenance: exact
identical chunks can be deduplicated, but unversioned overlaps are not merged.
Conflicting overlap within one declared revision raises an error.

```python
from ollama import Client
from obsidian_rag.context import ContextConfig, build_context, load_generation_counter
from obsidian_rag.generation import generate_cited_answer

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
identities/scores, a source-filename citation map, processing decisions and token
accounting. `messages` returns a fresh API payload. Citation-map values in
`to_dict()` are zero-based indices into `evidence_blocks`; decision ranks address
the original candidate list. A `merged` event describes consolidation, while the
representative's later `selected` or `budget` event describes the block's outcome.
Structured contexts additionally expose `citation_sources`, numbered `S1`, `S2`,
etc. in final evidence order, and a `context_id` fingerprint binding the messages
and source identities. IDs are local to one context. Every budget trial includes
the citation protocol and IDs. Only evidence actually sent to the model may be
cited; merged blocks retain all contributing chunk identities.

`citation.py` contains immutable answer/source contracts and pure parsing,
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
an answer from potentially truncated input. The legacy
`generate_answer(question, results, client=...)` API builds the same budgeted
context automatically; it also accepts `config` and `counter`. For backward
compatibility, this string-input API and `build_context` default to the legacy
protocol. Use `generate_cited_answer` for structured output, or pass a structured
context to `generate_answer` to obtain its rendered string.

An empty evidence set returns an insufficient-information message without
calling the generation model. If the fixed question/system prompt cannot fit,
or all evidence is excluded by the budget, generation raises
`ContextBudgetError`. `--show-context` exposes `budget_exhausted` as a diagnostic
status. Both CLI question modes now default to structured citations. Select
`--citation-mode legacy` only when comparing the historical filename prompt.
Answer factuality, completeness and citation support still require evaluation.

Both CLI question modes accept the context options listed above:

```sh
uv run --locked obsidian-rag query "What does chunking preserve?" --offline --show-context
uv run --locked obsidian-rag query "What does chunking preserve?" --offline --answer-json
uv run --locked obsidian-rag query "What does chunking preserve?" --offline \
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
src/obsidian_rag/    Python package
tests/              Automated tests
pyproject.toml      Project metadata and dependencies
uv.lock             Resolved dependency versions
```

The sample documents can be shared with the repository. Virtual environments,
caches, local `config.toml`, and local `.env` files are excluded from Git. Keep
machine-specific paths and credentials in ignored local files.

## Bounded embedding execution

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

`retrieval.search_numpy` and `retrieval.search_qdrant` return `schema.VectorHit`:
a stable chunk ID and a cosine score (larger is better). Both filter by source
before top-k selection. NumPy searches the validated SQLite snapshot directly,
with exact cosine ranking and input-order ties. It does not maintain a second
mutable copy of the snapshot. Qdrant supports exact and ANN queries; callers open
and validate its collection with `retrieval.check_qdrant_collection` before use.
`retrieval.search_index` handles this validation for application queries.

## Build an index snapshot

`indexing.build_index` accepts a complete list of loaded notes, a resolved
`EmbeddingSpec`, vault ID, matching tokenizer, active context limit, and Ollama
client. It validates inputs, chunks notes, caches successful embedding batches,
validates the candidate snapshot, prepares Qdrant when selected, then publishes.
The report includes the published manifest and counts of unique embedded/cached
inputs. Duplicate text occurrences share vectors but retain separate chunk IDs.
A later batch failure leaves previous published snapshots available and earlier
successful batches reusable. Model artifact discovery belongs to the caller;
changing the declared model revision changes cache identity.

`retrieval.search_index` queries a captured READY snapshot and embeds only the
question. Supply the actual model spec and tokenizer; mismatches fail before
model calls. The saved query instruction and context limit are reused. Explicit
`index_version` pins a request across concurrent publication. Returned chunks
come from the snapshot, not potentially edited source files; unknown vector hits
and invalid scores are rejected. The original `retrieve` remains the exact
in-memory reference function.

## Persistent CLI commands

```sh
uv run --locked obsidian-rag index --notes-dir example_notes --offline
uv run --locked obsidian-rag query "How should I write permanent notes?" --offline
uv run --locked obsidian-rag query "How should I write permanent notes?" --offline --json
uv run --locked obsidian-rag status
```

The default database is `.obsidian-rag/index.sqlite` (Git-ignored); use `--db`
and `--vault-id` consistently across commands. `index` prints a JSON build report.
`query --json` prints retrieved chunks without generation; `--source` filters by
exact source path. `status` reads saved manifests without loading a tokenizer or
contacting Ollama. Queries use a pinned snapshot even if the source notes change.
Each command opens/closes its own database connection.

Persistent commands currently validate the Qwen 0.6b tokenizer/model pairing.
The CLI reads the installed model digest and dimensions rather than inventing a
revision. `index --context-length` defaults to 8192, cannot exceed the model's
advertised limit, and is explicitly sent as `num_ctx` for document and query
embedding requests. `--batch-size`, `--max-batch-tokens`, and `--max-retries` control
embedding work. `--query-instruction ''` saves a raw-query configuration.
A missing index or mismatched model/tokenizer produces an error instead of
silently rebuilding on a query. The original `obsidian-rag "question"` invocation
remains an ephemeral baseline; use `query` to reuse a persistent index.

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
Historical snapshots are retained; deletion of the active snapshot is rejected.

## Qdrant backend

`indexing.QdrantIndex` creates, writes, verifies, and removes collections using
external vectors, with one collection per candidate. Queries live in
`retrieval.search_qdrant`. Collection metadata binds the embedding spec
and vault; opening incompatible collections fails without changing their data.
Payload indexes for vault, embedding spec and source are created before ingestion.
Chunk SHA-256 IDs map deterministically to UUID point IDs; full identities stay in
payload and are verified on retrieval. Qdrant stores cosine vectors as float32,
so allow small score-rounding differences from the NumPy float64 reference.
`create=True` never recreates an existing collection. Writes/deletes wait for
completion. Text remains in SQLite, and this collection handler never runs an embedding model.

Regular backend tests use Qdrant Local for API behavior only. Real server tests
are enabled with `OBSIDIAN_RAG_QDRANT_URL=http://127.0.0.1:6333`; they create and
remove uniquely named test collections. Qdrant Server/client 1.19 are the validated
pair; ANN and payload-index performance are not inferred from Local Mode tests.

## Publish Qdrant indexes

Run a local server separately (for example the validated `qdrant/qdrant:v1.19.0`
Docker image), then select the backend explicitly:

```sh
uv run --locked obsidian-rag index --backend qdrant --qdrant-url http://127.0.0.1:6333 --offline
uv run --locked obsidian-rag query "What does chunking preserve?" --offline --json
```

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

## Evaluate retrieval backends

`evaluation.compare_retrieval` runs the same query vectors against a NumPy exact
reference and supplied backends. It reports neighbor Recall@k separately from
source-group recall and union coverage of labeled sections. Section anchors must
use `body_start_char`/`body_end_char` in loaded `Note.content`; raw-file offsets
are rejected. Empty reference sets produce an undefined recall, not a perfect
score. Exact ties and float32 rounding can change rank order, so raw IDs/scores
are retained. No generated-answer accuracy is inferred from these metrics.

```sh
uv run --locked python -m obsidian_rag.evaluation \
  --db .obsidian-rag/index.sqlite --cases path/to/cases.jsonl \
  --output path/to/new-evaluation --offline --top-k 2
```

The runner captures an active snapshot, verifies the actual embedding model and
Qdrant data, embeds each question once, and compares NumPy exact, Qdrant exact,
and Qdrant ANN when the snapshot uses Qdrant. It writes frozen cases, raw results,
metrics, source/configuration hashes and runtime metadata into a new directory;
existing evaluation directories are never overwritten. Search timing includes
backend I/O but excludes embedding and snapshot load. It reports stored vector
bytes and SQLite file sizes, not total process or server memory. Build reports
include `build_seconds` measured inside the writer lock (source scan excluded).

For a controlled HNSW experiment on a small corpus, rebuild with
`--indexing-threshold 1 --full-scan-threshold 10 --require-hnsw`. The full-scan
threshold affects the query planner; simply requesting ANN does not prove the
server used a graph for a small collection. No large-scale latency claim should
be made from the bundled small corpus. Keep evaluation outputs outside note
folders. Real end-to-end tests use `OBSIDIAN_RAG_RUN_MODEL_TESTS=1` and optionally
`OBSIDIAN_RAG_QDRANT_URL`; they check separate-process queries and incremental
edits/deletes against real services while cleaning their test collections.


Add `--context` to the evaluation command to compare three policies on each
identical NumPy candidate list: historical unbounded `raw`, `raw_budgeted`, and
processed `built`. The latter two use the same configurable generation budget.
`context_results.jsonl` retains final messages and provenance; `metrics.json`
includes input tokens, block counts, duplicate source-span fractions, section
coverage/retention and build time. These are context metrics, not answer scores.

```sh
uv run --locked python -m obsidian_rag.evaluation \
  --db .obsidian-rag/index.sqlite --cases path/to/cases.jsonl \
  --output path/to/new-context-evaluation --offline --top-k 2 --context \
  --context-window 8192 --max-output-tokens 1024
```

Add `--citations` to generate structured answers once on the same frozen NumPy
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

## Evaluate with BrowseComp-Plus

The benchmark adapter reuses the existing indexing, retrieval, context,
generation and citation modules. Questions and scoring labels are stored
separately; reference answers and relevance labels are never supplied to the
retriever or answer generator. Public tests use synthetic data.

Install the optional data-download dependencies:

```sh
uv sync --locked --extra benchmark
```

`obsidian-rag-benchmark` and `python -m obsidian_rag.benchmark` expose the same
commands. Use `--help` on a command to see its options. Paths below are examples;
keep downloaded data and evaluation artifacts outside the tracked source tree.
Keep disposable indexes under the ignored `.obsidian-rag/` directory.

```sh
# Download is explicit and requires immutable Hugging Face commit SHAs.
# Replace the two revision placeholders with the actual dataset revisions.
.venv/bin/obsidian-rag-benchmark download \
  --query-revision FULL_QUERY_COMMIT_SHA --corpus-revision FULL_CORPUS_COMMIT_SHA \
  --output /path/to/browsecomp-data

# Local input also accepts the official decrypted JSONL case format.
# Omit --query-ids to select all questions; supplying a subset preserves order.
.venv/bin/obsidian-rag-benchmark prepare \
  --corpus /path/to/browsecomp-data/corpus.jsonl \
  --cases /path/to/browsecomp-data/cases.jsonl \
  --dataset-revision my-frozen-dataset \
  --output /path/to/evaluations/prepared

.venv/bin/obsidian-rag-benchmark index \
  --prepared /path/to/evaluations/prepared \
  --db .obsidian-rag/browsecomp.sqlite --offline

.venv/bin/obsidian-rag-benchmark run \
  --prepared /path/to/evaluations/prepared \
  --db .obsidian-rag/browsecomp.sqlite --output /path/to/evaluations/run-1 \
  --offline --chunk-top-k 2 --doc-ks 1 5 10 --generate

.venv/bin/obsidian-rag-benchmark judge \
  --run /path/to/evaluations/run-1 --output /path/to/evaluations/judge-1 \
  --judge-model qwen3:32b

.venv/bin/obsidian-rag-benchmark report \
  --run /path/to/evaluations/run-1 --judgments /path/to/evaluations/judge-1 \
  --output /path/to/evaluations/report-1

.venv/bin/obsidian-rag-benchmark export \
  --run /path/to/evaluations/run-1 --output /path/to/evaluations/export-1
```

Downloads stream the official data at pinned revisions, retain corpus text
once, and compact case records to questions/answers/document IDs. A completion
manifest records revisions, counts and file hashes; interrupted downloads are
not accepted as complete. `prepare` preserves every supplied corpus document,
including negatives. It records selected question IDs, source URLs, text hashes
and reversible `source`/`docid` identities without creating Markdown files.
Original whitespace and Markdown titles remain in `Note.content`. Local custom
corpora are marked as custom; selecting fewer questions does not shrink the corpus.

The current index builder holds notes, chunks and vectors in memory. Streaming
downloads do not remove that indexing constraint. Start with a fixed small
corpus containing positive and negative documents to verify the workflow and
measure resource needs. Such a reduced corpus is a custom experiment, not an
official full-corpus benchmark score. Data preparation reports total characters
and largest document size. Qdrant builds are also available with
`index --backend qdrant --qdrant-url http://127.0.0.1:6333`; this does not remove
the builder's in-memory preparation. Runs default to exact search; use `--ann`
explicitly for Qdrant ANN and retain that setting in comparisons.

`run` without `--generate` evaluates retrieval only. `--chunk-top-k` controls
retrieved candidate chunks; `--context-top-k` optionally selects a smaller
prefix for generation. `--doc-ks` only changes document-level metric cutoffs,
never candidate depth. Documents are ranked by their best retrieved chunk
score, with ascending `docid` for ties. Multiple chunks count once. A short
document ranking is scored as returned; Recall@k divides by all labeled
relevant documents. Evidence and gold qrels are scored separately. These
metrics do not establish that an answer-bearing passage reached the model.

Every run freezes a snapshot and records code/package/model identities and CLI
server metadata before queries. `results.jsonl` retains candidates, document
ranks, final context, citations, raw output, tokens and stage timings. Retrieval,
context and generation failures remain rows in the selected query set. Files
are flushed per question; interrupted runs lack a completion summary. Existing
outputs are rejected. Report and export verify hashes and never modify the run.
`--offline` prevents tokenizer downloads; local Ollama is still contacted.

Judging reads only generated claims and missing-information statements, not the
renderer's source-text appendix. The official prompt is bundled at revision
`046949032b0328319cc9a02663a759ec601d9402`, with its SHA-256 and MIT notice in
`browsecomp_judge.json`. The default judge is the installed `qwen3:32b`; weights
are never pulled automatically. Other installed judges can be selected, with
their actual digest and settings recorded. Ollama/quantized execution is a local
adaptation, not a claim of identical execution to the official vLLM evaluator.
A conservative UTF-8 byte input budget is used for Qwen tokenizers; it is not
an exact token count and oversized inputs are rejected rather than trimmed.
Judge output truncation, malformed verdicts and service failures remain
unresolved grading errors. Reports distinguish accuracy on graded answers,
pipeline failures and judge errors; overall accuracy is null while judge errors
remain, accompanied by lower/upper bounds. Changing the judge needs a new output
directory and never regenerates answers. Citation semantic support remains null.

Export writes `run.trec` and, for generation runs, per-question JSON under
`runs/` matching the upstream evaluator. TREC scores are descending rank
ordinals to preserve tie order; original cosine scores remain in the raw run.
`retrieved_docids` includes all retrieved documents, not only those cited.
`answered`, `partial` and `insufficient_evidence` all map to `completed` when
execution succeeds; answer correctness is scored separately. To use the
official evaluator, point it at the exported `runs/` subdirectory.
