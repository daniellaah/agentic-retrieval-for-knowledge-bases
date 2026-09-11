# Agent Evaluation: Harnesses, Infrastructure, Traces, and Conversation Tests

An agent's final sentence is only one observation about its behavior. Evaluation also depends on the tools, environment, execution history, and actual state produced. This note connects four perspectives on measuring agent systems and explains why a benchmark score should be interpreted with its execution conditions.

## Evaluate the agent and its harness together

Anthropic's SWE-bench discussion illustrates that a coding evaluation measures a model operating through an agent scaffold. Tool interfaces, prompts, the interaction loop, and the repository environment influence what the model can attempt. A coding task typically involves understanding an issue, inspecting files, changing the repository, and checking the result. The evaluation must determine whether the patch satisfies the task, rather than accepting the agent's statement that it is complete. Access to the necessary context and functioning tools also affects whether a task is solvable. Consequently, a reported score is tied to the model and harness configuration used for the run. Historical benchmark results should not be read as timeless model rankings. This section summarizes source 1.

## Treat infrastructure as an experimental variable

Anthropic's infrastructure-noise analysis shows how resource conditions can affect agent benchmark outcomes. Memory, compute limits, time budgets, and concurrent activity can cause failures or alter the strategies an agent can successfully execute. A dependency installation that fails under resource pressure may work when more capacity is available. Additional resources can also enable a more expensive approach, changing behavior rather than merely removing accidental crashes. A score difference under such conditions cannot automatically be attributed to better model reasoning. Recording infrastructure and failure modes makes comparisons more interpretable. The experiment should distinguish task errors from environmental problems while retaining both in the run record. Otherwise, a change in the execution environment can masquerade as a model improvement. This section summarizes source 2.

## Observe runs, traces, and threads

LangChain distinguishes a run, a trace, and a thread when discussing agent observability. A run represents an individual operation such as a model call. A trace captures a complete execution and its nested operations. A thread connects interactions across a multi-turn conversation. These levels answer different questions. One model call may look correct while the overall execution uses a failed tool result or leaves an intended action unfinished. Traces can expose tool arguments, returned values, prompts, and state transitions. Thread-level inspection reveals dependencies across turns that an isolated final response hides. Observability supplies the evidence needed to diagnose behavior; it does not by itself define what successful behavior should be. This section summarizes source 3.

## Test the final turn against real preceding context

LangChain's evaluation-readiness guidance describes N-1 testing: retain the real preceding conversation and generate only the final turn under evaluation. This avoids compounding the behavior of a synthetic user and agent across an entirely simulated conversation. It also makes it easier to isolate a specific response while preserving the history that response depends on. Evaluation can examine the final output, the trajectory of actions, and the resulting state separately. A reference solution can help establish that a task is feasible, while positive and negative examples clarify the expected behavior. A fluent completion claim is insufficient when success requires an external artifact or a state change. The test must inspect the relevant outcome directly. This section summarizes source 4.

## Illustrative scenario: an export that was never created

Imagine a fictional assistant asked to export a comparison table. The preceding conversation specifies the required columns and destination. This is an original example. The assistant's final message says that the export succeeded, but the destination contains no file. An output-only evaluator might reward the polite response even though the requested state was never produced.

A trace shows that the export operation returned an error and that the assistant continued without checking the destination. A state check identifies the missing artifact. An N-1 test can preserve the real conversation and ask a revised agent to handle that final request, making it possible to inspect whether the behavior changes without simulating every earlier user turn.

Now suppose the export fails only when the environment has little free memory. The report should retain that failure and describe the resource conditions. If a later run succeeds with more memory, the evidence supports an environmental effect, not automatically a better reasoning model. A useful evaluation record connects the requested outcome, observed state, execution trace, and infrastructure settings. Each contributes information that the final answer alone cannot supply.

## Sources

Accessed 2026-09-05. Sections 1–4 correspond to the sources below; the scenario is original.

1. [Anthropic — SWE-bench and agent scaffolding](https://www.anthropic.com/engineering/swe-bench-sonnet)
2. [Anthropic — Infrastructure noise in agent benchmarks](https://www.anthropic.com/engineering/infrastructure-noise)
3. [LangChain — Agent observability powers agent evaluation](https://www.langchain.com/blog/agent-observability-powers-agent-evaluation)
4. [LangChain — Agent evaluation readiness checklist](https://www.langchain.com/blog/agent-evaluation-readiness-checklist)
