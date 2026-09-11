# Semantic Search with Sparse Term Weights

## Core idea

A learned sparse encoder represents text using a sparse set of weighted vocabulary terms. Elastic describes text expansion that adds related terms beyond the literal wording of a query. This supplies a semantic matching signal while retaining a representation based on terms rather than a dense array of abstract coordinates.

## Practical implication

Distinguish learned expansion from a manually maintained synonym list. Evaluate it on vocabulary mismatch, where a relevant document uses different words for the same concept.

## Source

[Elastic — Introducing Elastic Learned Sparse Encoder](https://www.elastic.co/search-labs/blog/introducing-elastic-learned-sparse-encoder-elser/)
