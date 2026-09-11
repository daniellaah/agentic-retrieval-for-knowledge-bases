# Prompt Injection Defenses: Provenance, Cascades, and Evaluation Limits

An assistant may read external documents that contain text resembling instructions. A defensive design needs to distinguish task authority from retrieved material and decide when additional inspection is worthwhile. The sources here describe training and monitoring approaches, including their limitations. The final example is an original, benign illustration.

## Preserve instruction authority

OpenAI's instruction-hierarchy research studies how models can prioritize instructions by their source and privilege. A lower-trust input should not acquire authority merely because it uses commanding language or appears inside a tool result. Training examples can teach a model to follow legitimate requests while disregarding conflicting instructions embedded in lower-priority material. This matters for assistants that consume retrieved passages, web pages, or other externally supplied text. The intended behavior is selective: the model should still use relevant information from the material. Treating every external document as unusable would defeat the task. Instruction prioritization is a learned behavioral defense and should be tested under realistic attacks and benign inputs, rather than treated as a mathematical guarantee. This section summarizes source 1.

## Mark external material as data

Microsoft discusses Spotlighting as a way to make lower-trust content more distinguishable to the model. Its techniques include delimiting material, adding data markers, and encoding content, together with instructions that explain the representation. The purpose is to preserve provenance when text from outside the application enters the prompt. Clear boundaries can reduce confusion between data to analyze and instructions to execute. These approaches do not create a physical security boundary around the underlying text; their effectiveness depends on how the model interprets the representation. They fit within a broader defense against indirect prompt injection. A useful assessment therefore includes both resistance to malicious material and the ability to complete legitimate tasks using ordinary external documents. This section summarizes source 2.

## Escalate suspicious cases to a stronger classifier

Anthropic's next-generation Constitutional Classifiers use a cascade to reduce the cost of inspecting traffic. A relatively inexpensive internal probe examines traffic first. Suspicious exchanges are escalated to a more capable classifier that considers the input and output together. A first-stage flag is not automatically a final refusal. This allows the initial stage to be sensitive while giving the later stage an opportunity to distinguish concerning behavior from benign content. Considering the exchange can be more informative than judging an isolated message. The design separates detection effort from the final decision, which is useful when expensive classification on every interaction would impose substantial overhead. Its measured effectiveness still depends on the evaluated distribution and adversarial testing. This section summarizes source 3.

## Keep a monitor's evidence limits visible

Anthropic's research on cheaper monitors explores reusing model representations through lightweight probes or training only later layers. Sharing earlier computation can make monitoring less expensive than running an entirely independent full model. The study focuses on input classification rather than evaluating generated outputs. It also does not establish robustness against adaptive adversarial attacks. A result on a fixed collection of concerning and benign examples is therefore narrower than a claim that an attacker cannot learn to evade the detector. Computational savings, ordinary classification quality, and adversarial robustness are separate properties. Reports should identify which properties were measured, so that an inexpensive detector is not mistaken for a complete security solution. This section summarizes source 4.

## Illustrative scenario: an assistant reviewing a support article

Imagine a support assistant reading a fictional troubleshooting page. The page contains useful descriptions of an error and an unrelated sentence asking the assistant to change its output format. This is an original illustration, not an attack procedure. The application wants the assistant to extract the troubleshooting facts while treating the page's behavioral request as source content.

A provenance marker can help show where the retrieved text begins and ends. A sensitive first-stage monitor may still flag the exchange because it contains instruction-like language. A second stage can inspect the actual request and proposed answer before reaching a decision. Refusing every flagged article would make ordinary troubleshooting less useful, while accepting every article would ignore the reason for monitoring.

The evaluation records whether the assistant used the relevant facts, followed an inappropriate embedded instruction, or unnecessarily refused the task. It also records the added monitoring cost. Repeating a fixed test suite would support regression tracking, but it would not establish resistance to attackers who adapt their inputs after observing the system. That additional claim requires a different evaluation design.

## Sources

Accessed 2026-09-05. Sections 1–4 correspond to the sources below; the scenario is original.

1. [OpenAI — The instruction hierarchy](https://openai.com/index/the-instruction-hierarchy/)
2. [Microsoft Security Response Center — Defending against indirect prompt injection](https://www.microsoft.com/en-us/msrc/blog/2025/07/how-microsoft-defends-against-indirect-prompt-injection-attacks)
3. [Anthropic — Next-generation Constitutional Classifiers](https://www.anthropic.com/research/next-generation-constitutional-classifiers)
4. [Anthropic Alignment Science — Cheap monitors](https://alignment.anthropic.com/2025/cheap-monitors/)
