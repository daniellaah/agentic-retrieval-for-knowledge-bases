# Combining Lexical and Vector Rankings

## Core idea

Lexical retrieval matches words, while dense vector retrieval can match related meaning expressed with different wording. Hybrid search combines their candidate results. Reciprocal Rank Fusion uses positions in the ranked lists, which avoids treating the raw lexical and vector scores as if they were directly comparable.

## Practical implication

Evaluate hybrid retrieval on both exact terminology and paraphrases. A fused result has a new ranking score whose interpretation differs from a cosine similarity value.

## Source

[Elastic — Elasticsearch hybrid search: Overview and queries](https://www.elastic.co/search-labs/blog/hybrid-search-elasticsearch)
