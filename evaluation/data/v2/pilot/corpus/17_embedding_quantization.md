# Reducing Vector Precision

## Core idea

Embedding quantization stores vector coordinates at lower numerical precision. Scalar quantization can map floating-point values into integer buckets, while binary quantization retains one bit per coordinate. Calibration data affects the scalar bucket boundaries. A search can use compressed vectors for candidate selection and higher-precision vectors for rescoring.

## Practical implication

Measure the quality and memory trade-off on representative data. Lower precision and fewer dimensions are different changes to an embedding representation.

## Source

[Hugging Face — Binary and Scalar Embedding Quantization for Significantly Faster and Cheaper Retrieval](https://huggingface.co/blog/embedding-quantization)
