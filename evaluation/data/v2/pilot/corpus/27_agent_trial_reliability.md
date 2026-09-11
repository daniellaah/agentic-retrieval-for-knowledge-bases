# Success Once Versus Consistent Success

## Core idea

An agent can behave differently across repeated attempts at the same task. The pass-at-k view asks whether at least one of several attempts succeeds. The all-trials-success view asks whether every attempt succeeds. These measure different product needs: finding one working solution and delivering dependable behavior on repeated use.

## Practical implication

Report how many attempts were allowed and retain failures. A successful retry should not erase the earlier failure when measuring reliability.

## Source

[Anthropic — Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)
