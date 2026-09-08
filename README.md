# Obsidian RAG — Agentic Knowledge Retrieval Foundations

Independent retrieval capabilities over a personal Markdown knowledge base.
The knowledge layer owns source text, chunks, identities and indexes. Retrieval
methods return a shared `SearchResponse`; Context and Citation consume verified
source excerpts. The current CLI provides a deterministic vector retrieval
and answer workflow.

Implemented methods: `grep_search`, `metadata_search`, `bm25_search`, and
`vector_search`. Grep, metadata and BM25 are Python APIs; vector search also has
persistent CLI commands. Agent orchestration, hybrid fusion and reranking are
not implemented.

## Package layout

```text
src/obsidian_rag/
├── knowledge_base/
│   ├── models.py           # Note, Chunk, ChunkRecord
│   ├── identity.py         # Stable source/content/configuration identities
│   ├── sources.py          # SourceRef, SourceExcerpt, KnowledgeSnapshot
│   ├── loaders.py          # Markdown loading and consistent directory scan
│   ├── chunking.py         # Shared whole-note / recursive chunking
│   ├── metadata.py         # Frontmatter inspection
│   ├── lexical_index.py    # Immutable note/chunk term statistics
│   ├── embeddings.py       # Embedding preparation, batching and validation
│   ├── tokenization.py     # Verified embedding tokenizer
│   └── vector_index/
│       ├── manifest.py     # EmbeddingSpec, IndexManifest, vector identities
│       ├── storage.py      # SQLite snapshots and embedding cache
│       ├── indexing.py     # Build, reuse, publish and recovery
│       └── qdrant.py       # Qdrant collection lifecycle and low-level queries
├── retrieval/
│   ├── models.py          # Shared result/response/score/target contracts
│   ├── pagination.py      # Deterministic scoped pages
│   ├── grep.py
│   ├── metadata.py
│   ├── bm25.py
│   └── vector.py
├── context.py
├── citation.py
├── generation.py
├── evaluation.py
└── cli.py
```

Filenames and functions use `snake_case`; classes use `CapWords`. Public methods
are `grep_search`, `metadata_search`, `bm25_search`, and `vector_search`. Types
stay with their owning domain or behavior; there is no `*Model` naming suffix
and no file per dataclass. `vector.py` owns retrieval policy, while collection
writes, source parsing, chunking and caching have their own responsibilities.

## Setup and vector CLI

Use Python 3.13 and uv from the repository root:

```sh
uv sync --locked
ollama pull qwen3-embedding:0.6b
ollama pull qwen3.5:4b
```

Ollama must be running for vector embedding and generation. Start a Qdrant
server separately; the integration suite has been verified with 1.19.0:

```sh
docker run --rm --name obsidian-qdrant \
  -p 127.0.0.1:6333:6333 -v obsidian-qdrant:/qdrant/storage \
  qdrant/qdrant:v1.19.0
```

Build once, then query the published snapshot:

```sh
uv run --locked obsidian-rag index --notes-dir example_notes \
  --qdrant-url http://127.0.0.1:6333
uv run --locked obsidian-rag status
uv run --locked obsidian-rag query "What do my notes say about agent memory?" --json
uv run --locked obsidian-rag query "What do my notes say about agent memory?" --show-context
uv run --locked obsidian-rag query "What do my notes say about agent memory?" --answer-json
```

A question without the `query` command is shorthand for the same indexed query.
It never rescans or embeds the documents. If no index exists, run `index` first.
`--json`, `--show-context` and `--answer-json` are mutually exclusive output modes.
Without an output flag, the CLI renders a cited answer.

| Command options | Behavior |
| --- | --- |
| `--db`, `--vault-id` | Select SQLite state (default `.obsidian-rag/index.sqlite`) and vault (`default`) |
| `index --chunking recursive --chunk-size 512 --chunk-overlap 64` | Default body-token chunk budgets |
| `index --chunking none` | Whole-note chunks |
| `index --context-length 8192` | Complete embedding input limit, also sent as Ollama `num_ctx` |
| `index --batch-size`, `--max-batch-tokens`, `--max-retries` | Bounded embedding execution; retries checkpoint successful batches |
| `index --force` | Rebuild the collection using compatible cached document vectors |
| `query --top-k 2 --source file.md` | Limit candidates; source filtering happens before top-k |
| `query --exact` | Qdrant exact search; default requests ANN |
| `query --ef-search N` | Qdrant HNSW search breadth |
| `query --index-version VERSION` | Pin a published snapshot in the selected vault |
| `query --qdrant-url URL` | Explicitly override a restored/moved server endpoint |
| `--offline --tokenizer-cache PATH` | Use only cached tokenizer assets |
| `--host`, `--timeout` | Ollama endpoint and request timeout |

Run `index --help` or `query --help` for all options. Qdrant credentials use
`QDRANT_API_KEY`; endpoint URLs must not embed credentials. HNSW build settings
include `--hnsw-m`, `--ef-construct`, `--indexing-threshold`,
`--full-scan-threshold`, `--index-timeout` and `--require-hnsw`. Small collections
can use flat search even for ANN requests. Use Server, not QdrantLocal, to test
actual HNSW construction and approximate search.

## Shared knowledge and source coordinates

```python
from pathlib import Path
from obsidian_rag.knowledge_base.loaders import load_notes
from obsidian_rag.knowledge_base.sources import KnowledgeSnapshot

snapshot = KnowledgeSnapshot.from_notes(load_notes(Path('example_notes')), vault_id='personal')
source = snapshot.note_refs()[0]
note = snapshot.read_note(source)
excerpt = snapshot.read_span(source, 0, min(100, len(note.content)))
```

The loader retains the existing flat-directory behavior: UTF-8 `.md` files,
filename order, no recursive traversal. The first `# ` line becomes the title
and is removed from the body; surrounding body whitespace is stripped. All
coordinates are Python character positions in this loaded `Note.content`, with
an exclusive end. They are not raw Markdown byte offsets. Frontmatter extraction
never rewrites this body. The current metadata API reads YAML tags/aliases;
inline hashtags remain searchable body text.

A `KnowledgeSnapshot` owns immutable loaded notes. Document IDs depend on vault
and relative path; document revisions depend on title and full loaded body.
Chunk IDs additionally bind source span/content. They do not depend on the
embedding model. Overlapping chunks retain their exact source positions.

To inspect or search the same snapshot that produced vector hits:

```python
from pathlib import Path
from obsidian_rag.knowledge_base.vector_index.storage import SQLiteStorage

with SQLiteStorage(Path('.obsidian-rag/index.sqlite'), read_only=True) as storage:
    manifest = storage.active_manifest('default')
    if manifest is None:
        raise ValueError('Build an index first.')
    snapshot = storage.knowledge_snapshot(manifest.index_version)
    records = storage.snapshot_records(manifest.index_version)
```

This source view reconstructs complete notes from saved spans and verifies
coverage, overlap, document revisions and corpus fingerprint. It reads no
embedding BLOBs and calls no model or Qdrant service. Its references preserve the
published snapshot ID, so vector hits can be passed to `snapshot.read_note`.
Freshly loaded live notes are a different snapshot even when filenames match.

## Independent retrieval APIs

```python
from obsidian_rag.retrieval.grep import grep_search
from obsidian_rag.retrieval.metadata import MetadataQuery, metadata_search
from obsidian_rag.knowledge_base.lexical_index import LexicalIndex
from obsidian_rag.retrieval.bm25 import bm25_search

literal = grep_search(snapshot, 'RAG', case_sensitive=False, limit=20)
by_title = metadata_search(snapshot, MetadataQuery(title_contains='memory'))
by_tag = metadata_search(snapshot, MetadataQuery(tags=('AI',), path_prefix='notes/'))
lexical = LexicalIndex.build(snapshot)
ranked_notes = bm25_search(lexical, 'retrieval evaluation', limit=10)
```

| Method | Match and result semantics |
| --- | --- |
| grep | Literal body match; matching lines or multiline ranges; no score |
| metadata | AND title/path/frontmatter tag/alias filters; note targets without excerpts or scores |
| BM25 notes | Ranked title+body terms; note targets to inspect before generation |
| BM25 chunks | Ranked title+chunk terms; chunk targets and verbatim evidence |
| vector | Query embedding → Qdrant → source records; chunk targets with cosine score |

For chunk-level BM25, reuse shared chunks, rather than a BM25-specific splitter:

```python
# `records` above belong to the same published `snapshot`.
lexical_chunks = LexicalIndex.build(snapshot, chunks=[r.chunk for r in records])
ranked_chunks = bm25_search(lexical_chunks, 'retrieval evaluation', limit=10)
```

Grep, metadata and BM25 accept `paths=(...)`, `limit` and `cursor`. Follow
`next_cursor` until `has_more` is false. Cursors bind the query, scope, snapshot
and relevant method settings; ranks restart at one per page. Vector top-k is
not an exhaustive enumeration: `has_more=None` means unknown.

The BM25 analyzer uses case folding and NFC for search terms, plus Han character
unigrams/bigrams for Chinese text without downloaded segmentation models. Source
text stays unchanged. It uses `k1=1.2`, `b=.75`, positive Robertson IDF and tf
saturation; query terms count once. Defaults and IDF follow the
[Lucene BM25 reference](https://lucene.apache.org/core/10_2_2/core/org/apache/lucene/search/similarities/BM25Similarity.html).
Scores include the conventional `(k1 + 1)` numerator factor; do not assume numeric
parity with Lucene. Source filtering retains whole-index document statistics.
The initial lexical index is immutable and in memory; rebuild it for changed
content or analyzer settings. Persistence and language-specific segmentation
are not implemented.

Vector API callers supply explicit model/tokenizer/client dependencies:

```python
from obsidian_rag.retrieval.vector import vector_search, VectorSearchConfig

# `storage`, `spec`, `tokenizer`, `ollama_client`, and `qdrant_client` are opened
# by the caller. Resolve `spec` from the runtime model, not an invented digest.
response = vector_search(
    storage, 'What is agent memory?', vault_id='default', spec=spec,
    tokenizer=tokenizer, client=ollama_client, qdrant_client=qdrant_client,
    index_version=manifest.index_version,
    config=VectorSearchConfig(top_k=2, exact=True),
)
```

## Results, evidence and citations

`SearchResponse` contains `query`, `method`, `scope`, `items`, `limit`,
`has_more` and `next_cursor`. Use `response.to_dict()` for JSON. Each
`SearchResult` carries:

- `source`: vault, document ID/revision, path/title and snapshot ID.
- `target`: `NoteTarget`, `SpanTarget` or `ChunkTarget`.
- `method` and page-local `rank`.
- `excerpts`: immutable original source spans; may be empty.
- `score`: optional `SearchScore(value, metric, higher_is_better)`.

Scores are method-specific. A BM25 score can exceed one; it is not comparable to
cosine similarity or a probability. Unscored results are valid. A note target
without excerpts is a candidate to inspect, not evidence of a claim.

```python
from obsidian_rag.context import build_context
from obsidian_rag.retrieval.models import SearchResult, NoteTarget

hit = ranked_notes.items[0] if ranked_notes.items else None
if hit is not None:
    note = snapshot.read_note(hit.source)
    evidence = SearchResult(hit.source, NoteTarget(), 'read_note', 1,
                           (snapshot.read_span(hit.source, 0, len(note.content)),))
    context = build_context('What do these notes establish?', [evidence], citation_mode='structured')
```

Production generation passes `ContextConfig` and a verified `GenerationCounter`
to `build_context`. Defaults reserve 1,024 output tokens and 128 safety tokens
inside an 8,192-token window. Whole evidence candidates are selected within this
budget. Known overlapping spans merge only within the same vault/document
revision/snapshot and must agree on text. Different retrieval methods retain
separate provenance. Empty-excerpt candidates receive `needs_inspection`.

The default structured protocol assigns `S1`, `S2`, … to final evidence blocks,
validates references, and renders citations in application code. `--citation-mode
quoted` additionally validates exact unique source quotations and computes their
positions. Invalid references and truncated generation fail explicitly.
`--citation-mode legacy` retains filename prompting. Valid citation IDs and
verbatim quotes do not prove that a claim is semantically supported.

Generation token counting is bound to the verified `qwen3.5:4b` artifact and chat
template. The embedding tokenizer is separately pinned to Qwen3-Embedding-0.6B.
Unknown generation artifacts need an explicit Python `GenerationCounter` adapter.
There is no silent character-count fallback or automatic input truncation.
Embedding input remains `title + "\n\n" + body`, and the query instruction and
original whitespace are preserved. Cache identities and model digests are checked
before reuse. Use the module docstrings for exact tokenizer revisions and limits.

## Migration and lifecycle

The NumPy vector search backend, `retrieve`, `search_numpy`, `search_index`, and
in-memory CLI embedding workflow have been removed. NumPy remains a numeric
library for vector validation, cached arrays and evaluation statistics.

- Existing Qdrant SQLite v1 state, cache keys, chunk IDs and collection metadata
  remain readable. No full document re-embedding is required by this refactor.
- Old NumPy index metadata is readable for compatible cache reuse. Queries reject
  it explicitly; run `index` to publish a Qdrant replacement using that database.
- `index --backend ...` is removed: Qdrant is the only vector backend.
- Imports moved to the package paths above; old flat module paths are removed.
- `query --json` now returns the shared response contract instead of the old
  `index_version/question/results` wrapper. Citation diagnostics additionally
  carry score metric and retrieval method.
- Unchanged builds reuse verified active snapshots; changed/renamed/deleted notes
  publish a new candidate. Failed or interrupted candidates never replace the
  active version. Successful embedding batches remain reusable.
- Historical READY snapshots stay available for pinned readers. Failed owned
  Qdrant collections are cleaned up during recovery. Source scope cannot silently
  switch directories under one vault ID.

## Tests and reproducible evaluation

The default suite runs without network, using real temporary SQLite and
QdrantLocal stores plus boundary substitutes for external models:

```sh
.venv/bin/python -B -m pytest -p no:cacheprovider -q
```

With cached tokenizers, the supported models and a dedicated test Qdrant Server:

```sh
OBSIDIAN_RAG_RUN_MODEL_TESTS=1 \
OBSIDIAN_RAG_QDRANT_URL=http://127.0.0.1:6333 \
.venv/bin/python -B -m pytest -p no:cacheprovider -q
```

The opt-in suite exercises real token counts, generation, HNSW publication,
recovery and subprocess CLI persistence.

Evaluation uses Qdrant exact as the neighbor reference and compares ANN on the
same snapshot and frozen query vectors:

```sh
uv run --locked python -m obsidian_rag.evaluation \
  --db .obsidian-rag/index.sqlite --vault-id default \
  --cases cases.jsonl --output evaluation-new --offline --context --citations
```

The output folder must not exist. It retains cases, hits/scores/timings, metrics,
corpus identities, model/server/configuration metadata, and optional context and
citation records. Cases use unique `id`, nonblank `question`, optional
`required_source_groups`, and evidence anchors with `body_start_char` and
`body_end_char`. Neighbor recall, source recall, section coverage and citation
validity are separate measures; semantic support requires an explicit review.
Old evaluation artifacts remain historical NumPy/Qdrant baselines and are never
rewritten to appear comparable to the new reference.
