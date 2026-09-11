# Choosing Meaningful Chunk Boundaries

## Core idea

Chunking determines the unit stored and retrieved from a document. Fixed-size splitting controls length, while content-aware splitting follows boundaries such as paragraphs or sections. A fragment should retain enough context to be useful when retrieved independently. Very small fragments can lose meaning, while large fragments can include unrelated content.

## Practical implication

Test chunking against actual questions and document structure. Preserve nearby context when a definition or condition would otherwise be separated from the claim it qualifies.

## Source

[Pinecone — Chunking Strategies for LLM Applications](https://www.pinecone.io/learn/chunking-strategies/)
