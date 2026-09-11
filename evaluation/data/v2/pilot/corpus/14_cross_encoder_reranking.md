# Joint Scoring with a Cross-Encoder

## Core idea

A cross-encoder processes a question and a candidate passage together to produce a relevance score. An embedding model instead represents each text separately so document vectors can be reused. Joint scoring is more expensive across a large collection, so a cross-encoder is commonly applied to a shortlist returned by an initial retriever.

## Practical implication

Measure initial candidate coverage before evaluating reranking. Reordering a shortlist cannot recover a useful passage that never entered that shortlist.

## Source

[Hugging Face — Training and Finetuning Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-reranker)
