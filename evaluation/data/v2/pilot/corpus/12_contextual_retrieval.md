# Adding Document Context to Chunks

## Core idea

A retrieved passage may contain a useful fact but omit the company, subject, or period needed to interpret it. Contextual Retrieval prepends a short explanation specific to that passage before building its embedding and lexical index entry. The explanation situates the passage within the larger document.

## Practical implication

Use context that resolves the particular passage, rather than attaching the same generic summary everywhere. Check the added text against the source before treating it as evidence.

## Source

[Anthropic — Introducing Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval)
