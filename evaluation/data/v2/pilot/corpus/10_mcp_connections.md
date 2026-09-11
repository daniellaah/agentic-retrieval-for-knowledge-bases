# MCP Clients and Servers

## Core idea

The Model Context Protocol defines a common interface for connecting AI applications to external capabilities and data. A server exposes an integration, while a client inside an AI application connects to that server. Standardizing this boundary reduces the need to build a different connector for every application and data-source pairing.

## Practical implication

Use the client-server distinction to reason about integration responsibilities. A connection protocol supplies access to capabilities; the application still decides how those capabilities serve a task.

## Source

[Anthropic — Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol)
