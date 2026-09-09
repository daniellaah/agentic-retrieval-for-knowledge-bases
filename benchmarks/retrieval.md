# Retrieval baselines

Build `example_notes` with the normal `arkb index` command. The small authored
`retrieval-cases.jsonl` fixture labels documents by source path, with relevance
grades 0–3. It is a regression/example dataset, not an independently judged
quality benchmark. Unlisted documents are treated as irrelevant; expand and
review the labels before making quality claims.

```sh
uv run --locked python -m arkb.retrieval_evaluation \
  --cases benchmarks/retrieval-cases.jsonl --output /tmp/retrieval-baselines.json \
  --modes semantic bm25 --top-k 5 --offline
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
