# Multi-Agent Research: Delegation, Shared State, and Coordination Costs

A research assistant may need to explore independent sources, reconcile partial findings, and preserve a coherent plan. These activities put different demands on an agent architecture. This note connects four designs without assuming that adding more workers always improves the result. The final scenario is an original illustration.

## Delegate bounded investigations

Anthropic describes a lead researcher that plans work and delegates independent investigations to subagents. Separate contexts let workers explore different directions without filling the leader's context with every intermediate observation. The leader must still integrate their findings. Effective assignments specify the objective, expected output, useful tools, source expectations, and boundaries. Vague delegation can produce duplicate searches or leave important topics uncovered. Workers can write artifacts and return references, avoiding repeated paraphrasing through several agents. This also makes supporting material available for later inspection. The architecture is useful when work can be divided and when the value of broader investigation justifies additional tokens. Delegation itself does not establish that a conclusion is well supported; synthesis and source checking remain necessary. This section summarizes source 1.

## Separate planning from progress tracking

Microsoft's Magentic-One uses an Orchestrator with specialist agents for browsing, files, coding, and terminal execution. It distinguishes two records. The Task Ledger holds the known facts, tentative assumptions, and overall plan. The Progress Ledger tracks the current execution, including whether work is advancing and which agent should act next. The orchestrator uses an outer planning loop and an inner execution loop. Repeated stalls can cause it to revisit the Task Ledger and devise a different plan, rather than endlessly issue the same assignment. These records serve different purposes: one describes the problem and intended approach, while the other helps assess the latest actions. A message saying that a worker is active is therefore not equivalent to evidence that the original plan is still sound. This section summarizes source 2.

## Match coordination to the task

Google Research studies agent systems under controlled configurations and finds that coordination interacts with task structure. Parallelizable work can benefit from multiple investigators, while tasks with strong sequential dependencies may suffer from communication overhead and fragmented reasoning. Tool-heavy work can incur a coordination tax when agents repeatedly exchange observations and negotiate control. The topology also affects how mistakes travel: a worker's error may be checked, contained, or repeated by other participants. Consequently, an architecture comparison should hold the task and relevant resource constraints in view. A successful multi-agent benchmark does not imply that every serial workflow should become a team. The practical design question is which useful independent work becomes possible and what communication it requires. This section summarizes source 3.

## Make handoffs explicit

LangGraph's Command design combines a state update with a routing decision. A node can return an update describing changed shared state and a goto destination identifying the next node. A handoff may also route to a parent graph. The two pieces should be understood separately: transferring control determines who acts next, while updating state determines what information that participant can use. Declared possible destinations can support graph visualization even when the next step is selected dynamically. This is a control-flow mechanism, not a guarantee that the recipient understands the task or has received complete evidence. Explicit handoffs make such responsibilities easier to inspect because the state change and destination are visible together. This section summarizes source 4.

## Illustrative scenario: comparing three deployment options

Suppose a team asks an assistant to compare three imaginary deployment options. This is a hypothetical exercise, not a reported vendor deployment. The lead assigns one worker to each option and asks for a short table of supported features, unresolved questions, and links to the relevant documentation. Workers must distinguish documented facts from assumptions. A fourth worker is not automatically useful: it needs a distinct investigation, such as checking whether the proposed comparison uses equivalent configurations.

After one worker repeatedly encounters missing documentation, the lead records that limitation and changes the research plan. It does not treat another search attempt as progress by itself. Each worker saves its findings once, and the lead reads the artifacts when composing the comparison. Before transferring control to a reviewer, the system records which options remain uncertain and which artifacts support each claim. A useful evaluation would inspect duplicated work, missing evidence, and final comparison quality alongside total resource use. Faster completion alone would not show that delegation improved the research.

## Sources

Accessed 2026-09-05. Sections 1–4 correspond to the sources below; the scenario is original.

1. [Anthropic — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)
2. [Microsoft Research — Magentic-One](https://www.microsoft.com/en-us/research/articles/magentic-one-a-generalist-multi-agent-system-for-solving-complex-tasks/)
3. [Google Research — Towards a science of scaling agent systems](https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/)
4. [LangChain — Command: a new tool for multi-agent architectures](https://www.langchain.com/blog/command-a-new-tool-for-multi-agent-architectures-in-langgraph)
