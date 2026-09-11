# External Evidence Versus Parameter Updates

## Core idea

Retrieval-augmented generation places selected external information into the prompt at question time. Fine-tuning changes model parameters using training examples. The two approaches act at different points: one supplies evidence during inference, while the other adapts learned behavior. A system can use both approaches when its requirements call for both.

## Practical implication

Diagnose whether a failure comes from unavailable evidence or inadequate task behavior. Updating a knowledge collection and training a model are different maintenance operations.

## Source

[NVIDIA — RAG 101: Retrieval-Augmented Generation Questions Answered](https://developer.nvidia.com/blog/rag-101-retrieval-augmented-generation-questions-answered/)
