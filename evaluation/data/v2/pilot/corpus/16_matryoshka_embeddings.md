# Embeddings with Nested Dimensions

## Core idea

Matryoshka embedding models are trained so useful information is retained in shorter prefixes of their output vectors. A system can use fewer dimensions for a cheaper initial search and retain longer vectors for a more detailed comparison. This behavior depends on the training method; arbitrary embedding dimensions are not automatically interchangeable.

## Practical implication

Measure quality at the chosen dimension. After truncating a normalized vector, normalize the shorter vector again when the similarity calculation requires unit length.

## Source

[Hugging Face — Introduction to Matryoshka Embedding Models](https://huggingface.co/blog/matryoshka)
