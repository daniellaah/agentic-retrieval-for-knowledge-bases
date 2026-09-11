# Processing Tool Results in Code

## Core idea

Passing every intermediate tool result through a language model can fill its context with data that only needs filtering or aggregation. An execution environment can call connected services, process their results with ordinary code, and return a small summary. Loops, joins, and conditionals can run without a separate model decision for every operation.

## Practical implication

For a large spreadsheet, calculate totals in the execution environment and return the totals and necessary evidence rather than every row.

## Source

[Anthropic — Code execution with MCP: Building more efficient agents](https://www.anthropic.com/engineering/code-execution-with-mcp)
