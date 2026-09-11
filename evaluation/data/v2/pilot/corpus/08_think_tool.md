# Reasoning Between Tool Calls

## Core idea

The think-tool pattern gives an agent a designated step to review newly received information during a tool-use sequence. It is especially relevant when the next action depends on earlier results or detailed policies. The tool provides a place for reasoning; it does not itself fetch new evidence or modify the external environment.

## Practical implication

Evaluate this pattern on sequential decisions that require interpreting tool output. Its role differs from reasoning performed before the model starts its response.

## Source

[Anthropic — The think tool: Enabling Claude to stop and think](https://www.anthropic.com/engineering/claude-think-tool)
