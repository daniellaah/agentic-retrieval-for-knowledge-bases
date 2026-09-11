# Tool Interfaces for Language Models

## Core idea

A useful agent tool has a distinct purpose, an informative name, clear parameter descriptions, and outputs that help the agent decide what to do next. Overlapping operations and ambiguous identifiers make tool selection harder. Returning a focused result can be more useful than exposing every field from an underlying service.

## Practical implication

Test tool interfaces on realistic tasks. Inspect incorrect calls and unnecessary output before adding more tools to the same agent.

## Source

[Anthropic — Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents)
