# Reasoning with Learned Safety Specifications

## Core idea

Deliberative alignment teaches a reasoning model the content of safety specifications and trains it to apply those specifications when forming responses. The described method combines supervised fine-tuning on specification-referencing reasoning with later reinforcement learning. The intent is to improve decisions in difficult safety cases, including appropriate refusals and avoidance of unnecessary refusals.

## Practical implication

Evaluate both harmful compliance and refusal of benign requests. Correct calibration requires distinguishing the cases rather than maximizing refusal alone.

## Source

[OpenAI — Deliberative alignment: reasoning enables safer language models](https://openai.com/index/deliberative-alignment/)
