# Learning from Preferred and Rejected Answers

## Core idea

Direct Preference Optimization trains a language model with pairs of preferred and rejected responses to the same prompt. Its objective incorporates a reference model and increases the relative preference for the chosen response. It avoids separately fitting an explicit reward model and then running a reinforcement-learning optimization loop.

## Practical implication

Inspect preference pairs carefully. The learning signal comes from which response is chosen, so contradictory or poorly justified preferences can teach undesirable behavior.

## Source

[Hugging Face — Fine-tune Llama 2 with DPO](https://huggingface.co/blog/dpo-trl)
