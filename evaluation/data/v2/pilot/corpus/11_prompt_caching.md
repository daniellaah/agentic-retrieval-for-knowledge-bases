# Reusing Repeated Prompt Context

## Core idea

Prompt caching reuses computation associated with prompt material that appears repeatedly across requests. Long instructions, document collections, or examples can form a reusable context portion. This can reduce processing cost and latency when later requests share that material.

## Practical implication

Look for stable, repeated context when considering caching. Caching concerns reuse of prompt processing; it does not summarize the material or select which documents answer a question.

## Source

[Anthropic — Prompt caching with Claude](https://claude.com/blog/prompt-caching)
