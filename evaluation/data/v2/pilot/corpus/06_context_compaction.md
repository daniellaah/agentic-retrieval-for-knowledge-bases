# Compaction and Persistent Agent Notes

## Core idea

A long-running conversation can exceed the context available for the next model call. Compaction summarizes earlier history so work can continue with a smaller active context. Persistent notes keep selected facts outside that context for later retrieval. Useful summaries preserve decisions, unresolved problems, and dependencies while removing redundant material.

## Practical implication

Check what survives a context reset. An aggressively shortened summary can omit a constraint that becomes important later in the task.

## Source

[Anthropic — Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
