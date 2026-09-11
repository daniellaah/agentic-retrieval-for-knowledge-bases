# Progressive Disclosure in Agent Skills

## Core idea

An Agent Skill packages task instructions with optional references and executable resources. The agent first sees the skill name and description, then reads the detailed instructions when the task calls for them. Additional reference files can remain outside the context until needed. This layered loading keeps specialized knowledge available without reading every manual at startup.

## Practical implication

Keep the entry instructions focused and link to detailed material. Reusable scripts can perform deterministic operations without placing their entire implementation into the model context.

## Source

[Anthropic — Equipping agents for the real world with Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)
