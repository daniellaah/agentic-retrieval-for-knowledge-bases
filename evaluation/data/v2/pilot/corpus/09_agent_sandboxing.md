# Filesystem and Network Boundaries

## Core idea

An execution sandbox restricts what an agent and its subprocesses can reach. Filesystem boundaries limit readable or writable locations, while network boundaries limit reachable services. These controls address different paths by which a mistaken or manipulated action can affect resources outside the task.

## Practical implication

Review both dimensions of access. Restricting file writes alone does not control outbound traffic, and restricting the network alone does not protect local files.

## Source

[Anthropic — Beyond permission prompts: making Claude Code more secure and autonomous](https://www.anthropic.com/engineering/claude-code-sandboxing)
