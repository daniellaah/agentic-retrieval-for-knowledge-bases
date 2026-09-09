# Retrieval baselines

Build `example_notes` with the normal `arkb index` command. The small authored
`retrieval-cases.jsonl` fixture labels documents by source path, with relevance
grades 0–3. It is a regression/example dataset, not an independently judged
quality benchmark. Unlisted documents are treated as irrelevant; expand and
review the labels before making quality claims.

```sh
uv run --locked python -m arkb.evaluation.retrieval baseline \
  --cases benchmarks/retrieval-cases.jsonl --output /tmp/retrieval-baselines.json \
  --modes semantic bm25 hybrid --top-k 5 --offline
```

Semantic uses Qdrant exact search and the saved Ollama embedding configuration.
BM25 alone (`--modes bm25`) reads a READY SQLite snapshot without connecting to
Ollama or Qdrant. Both use identical persisted chunks. `--index-version` pins a
historical build. Runners never overwrite an output file.

The JSON report preserves questions, labels, source hashes, configuration,
snapshot manifest, raw ranked responses, and per-query metrics and latency.
Recall@K divides by **all** positive judgments, nDCG@K uses exponential graded
gain and logarithmic rank discount, and MRR is limited to the retrieved depth.
Document labels collapse repeated source hits in first-occurrence order after
retrieving K chunks; use `--relevance-key chunk_id` with chunk judgments to
measure chunk-level quality. No positive judgments produces null metrics and
an explicit defined-case count. Timing includes query embedding/search but
excludes model loading and building BM25 postings. Mode execution order rotates.

The same comparison is callable as `evaluate_retrievers({name: retriever}, cases)`;
each retriever exposes `search(query, top_k=..., filters=...)`. These baselines do
not require a unified engine.

BM25 indexes title plus body as one field, using NFC, casefold and Unicode `\w+`
tokens; underscores remain part of identifiers, punctuation splits tokens.
There is no stemming, stopword removal or CJK word segmentation. Distinct query
terms each contribute once; punctuation-only queries return no matches.
The positive [Lucene IDF](https://lucene.apache.org/core/7_6_0/core/org/apache/lucene/search/similarities/BM25Similarity.html)
is combined with standard `(k1+1)*tf / (tf+k1*(1-b+b*dl/avgdl))` saturation.
Defaults are k1=1.2 and b=0.75. Source filtering precedes top-K while corpus
statistics stay fixed. Scores are higher-is-better and meaningful only within
the same corpus/tokenizer/settings; they are not cosine similarities.

`arkb.retrieval.rrf` accepts named ranked lists (or a sequence of lists) and
returns a tuple of shared results. It implements the [RRF paper's rank sum](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf),
`sum(1/(k+rank))`, with one-based positions and default k=60. Within-list
duplicates vote only once at their first position; later entries retain their
original ranks. Equal scores sort by `SearchResult.identity`. Identity scopes
chunks to documents, falls back to located spans, then whole sources. Conflicting
text, spans, or known snapshot/revision metadata for one identity are errors.
`metadata.fusion.contributions` records each list, rank, method, raw score/type
and input metadata. Fused scores are tagged `rrf`; input scores are never summed.

`HybridRetriever(bm25, semantic, candidate_k=20, rrf_k=60)` calls each primitive
at the fixed candidate depth, then fuses with RRF and returns the requested
`top_k` (which must not exceed `candidate_k`). Both retrievers must pin the same
snapshot. Input queries and filters are passed through; failures propagate.
The runner exposes `--candidate-k` and `--rrf-k` and the independent `hybrid` mode.

`Reranker(scorer).rerank(query, candidates, top_k=...)` is independent of candidate
retrieval. `CandidateScorer` declares its identity/score semantics and returns
one finite higher-is-better score per input candidate. Duplicate candidates and
malformed scores fail explicitly. Ties sort by identity, source evidence stays
unchanged, and `metadata.rerank` retains the input rank, method, and raw score.
`evaluate_reranker` measures before/after metrics, rank movement and latency on
one frozen list; it does not perform candidate retrieval.

The optional [Sentence Transformers cross-encoder](https://www.sbert.net/docs/package_reference/cross_encoder/model.html)
adapter runs `cross-encoder/ms-marco-MiniLM-L6-v2` on CPU at pinned revision
`233902d25c440f23af6f7d6e94d2946bac0bee0a`. Other single-logit models can be supplied
with an explicit commit revision. It scores query/title+body pairs, explicitly
requests raw logits, and declares `cross_encoder_logit` semantics. The tokenizer
truncates pairs to 512 tokens by default; source evidence remains unmodified.
There is no remote code execution, generation, query rewriting or strategy choice.
Loading is explicit; importing the retrieval package never imports PyTorch.

```sh
uv sync --locked --extra rerank
ARKB_RUN_RERANKER_TESTS=1 ARKB_RERANKER_OFFLINE=0 \
  ARKB_RERANKER_CACHE=.uv-cache/reranker-models \
  uv run --locked --extra rerank python -m pytest -q tests/retrieval/integration/test_cross_encoder_model.py
```

After caching, use `ARKB_RERANKER_OFFLINE=1` to verify entirely offline. CPU
inference is repeatable within a fixed environment; bitwise agreement across
hardware or library versions is not guaranteed. The tiny integration fixture
checks relevance ordering, not broad model quality.

`RerankedRetriever(retriever, reranker, candidate_k=20)` composes any primitive
with optional reranking, requesting the full pool before applying final top-K.
For hybrid, require `top_k <= rerank_candidates <= candidate_k` (the per-source
hybrid depth). Retrieval cannot recover candidates excluded from that pool.

```sh
uv run --locked --extra rerank python -m arkb.evaluation.retrieval baseline \
  --cases benchmarks/retrieval-cases.jsonl --output /tmp/retrieval-four-way.json \
  --modes semantic bm25 hybrid hybrid_reranked --top-k 5 \
  --candidate-k 20 --rerank-candidates 20 \
  --reranker-cache .uv-cache/reranker-models --offline
```

`--reranker-model`, `--reranker-revision`, and `--reranker-max-length` select an
explicit reranker configuration. Relevance and latency use the same questions
and snapshot for every mode. Full input ranks in reranked hit metadata show
which candidates moved; isolated `evaluate_reranker` diagnostics measure just
reranking latency. Whole-pipeline timings include all retrieval and scoring calls.

The engine and CLI choose a supplied mode explicitly:

```python
from arkb.retrieval import RetrievalEngine
engine = RetrievalEngine(semantic=semantic, bm25=bm25, reranker=reranker)
engine.search("query", mode="hybrid", rerank=True, top_k=5)
```

`arkb search --mode semantic|bm25|hybrid --rerank --json` exposes the same choices;
`lexical` aliases `bm25`. Semantic remains the default. Without `--rerank`, no
reranker model is loaded. The independent benchmark runner continues to call the
primitives directly so engine composition does not hide baseline behavior.

For an isolated build with real Ollama embeddings and a cross-encoder, without
Qdrant Server, use the example runner:

```sh
uv run --locked --extra rerank python -B benchmarks/run_local_retrieval.py \
  --output /tmp/arkb-local-comparison
```

The output directory must be new. It contains the SQLite snapshot, Qdrant Local
index, and `report.json` with complete responses, source hashes, corpus identity,
model configuration, and frozen-candidate reranker ablations. It indexes the same
Markdown chunks as production. `--offline` requires both tokenizers and reranker
artifacts to be cached. Ollama still runs locally. `--modes` can select individual
baselines. Qdrant Local runs exact search; its timings cannot establish server or
ANN performance. The checked-in [example measurement](retrieval-example.md)
records a real run and its limits.
