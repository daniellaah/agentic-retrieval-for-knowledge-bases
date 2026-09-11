# Adapting Models with Small Trainable Additions

## Core idea

Parameter-efficient fine-tuning adapts a pretrained model while keeping most of its parameters frozen. Methods such as LoRA train a small set of additional parameters instead of producing a fully updated copy of every base weight. The resulting task-specific checkpoint can be much smaller than a complete model checkpoint.

## Practical implication

Account for the base model as well as the adapter when deploying. A small adapter represents a learned change, rather than a standalone replacement for the pretrained model.

## Source

[Hugging Face — PEFT: Parameter-Efficient Fine-Tuning of Billion-Scale Models](https://huggingface.co/blog/peft)
