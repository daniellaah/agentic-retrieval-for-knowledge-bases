# Drafting and Verifying Tokens

## Core idea

Speculative decoding uses a cheaper approximation, such as a small model, to propose upcoming tokens. A target model checks those proposals in parallel. The acceptance and correction procedure preserves the target sampling distribution while allowing multiple tokens to be produced through fewer sequential target-model steps.

## Practical implication

Its benefit depends on how often proposed tokens are accepted and the cost of drafting and verification. Faster generation is the objective, rather than teaching new knowledge to the target model.

## Source

[Google Research — Looking back at speculative decoding](https://research.google/blog/looking-back-at-speculative-decoding/)
