# LLM Serving: Prefill, Continuous Batching, and KV Cache Management

A retrieval system's response time includes more than finding documents. The generator must process the supplied context and produce an answer while sharing finite compute and memory resources. This note separates several serving mechanisms whose names can otherwise be confused with document chunking or embedding compression.

## Distinguish prompt processing from token generation

NVIDIA describes transformer inference in terms of prefill and decode. Prefill processes the supplied prompt, exploiting parallel work across prompt positions. Decode generates subsequent tokens autoregressively and often faces memory-bandwidth constraints. The key-value cache retains attention information from earlier positions so that generation can reuse it instead of recomputing everything. This cache is distinct from the model's learned weights and grows with the active sequences and their lengths. A larger prompt can therefore affect both initial processing and runtime memory demand. Different workload shapes stress different resources, so one aggregate latency figure can conceal the source of a bottleneck. Separating time before the first generated token from later decoding makes performance investigations more informative. This section summarizes source 1.

## Schedule unfinished requests efficiently

Hugging Face explains continuous batching as a way to admit new requests when earlier requests finish, instead of making an entire fixed batch wait for its longest sequence. Packing sequences and using suitable attention masks helps avoid unnecessary padding while keeping requests separate. Chunked prefill divides prompt processing into smaller scheduling units so that prefill work can share a token budget with decoding requests. Despite the similar word, this is not retrieval chunking. Chunked prefill changes when prompt tokens are processed and cached; it does not choose semantically relevant passages or create independently searchable document fragments. Scheduling efficiency depends on workload and implementation, so a higher throughput result does not imply that every individual request receives a lower latency. This section summarizes source 2.

## Compress attention state separately from other representations

NVIDIA's discussion of low-precision KV caches addresses the memory occupied by attention keys and values, especially with long contexts and large batches. Reducing the precision of this state can relieve capacity and bandwidth pressure. It is a different intervention from quantizing model weights or compressing vectors stored in a retrieval index. Each representation has its own role and error effects. A serving experiment should therefore specify what was quantized and examine answer quality as well as memory use and speed. Hardware support and implementation affect the outcome. A format that helps one serving configuration is not evidence that the same tradeoff holds for an arbitrary model, device, or sequence length. This section summarizes source 3.

## Consider transfer cost before offloading

NVIDIA Dynamo discusses moving KV cache data across memory and storage tiers, including device memory, host memory, and other storage or network locations. Offloading can preserve reusable state when accelerator memory is scarce, but it introduces transfer and coordination costs. The useful comparison is between moving and reusing that state versus recomputing the relevant work. Access locality, available bandwidth, and expected reuse influence the choice. A cache tier is valuable only when its placement and movement fit the workload. Managing cache capacity is therefore not just a matter of saving every possible prefix. An effective system must decide which state is worth retaining and whether it can be retrieved quickly enough to help. This section summarizes source 4.

## Illustrative scenario: comparing two retrieval prompts

Suppose a local assistant receives either two compact passages or two complete manuals. This is a hypothetical comparison, not a measured result for any particular model or computer. The generated answer length is similar, but the second request carries much more input. Recording only total response time would make it hard to tell whether additional time came from input processing, decoding, loading the model, or unrelated machine activity.

An experiment records retrieved-context size, first-token timing when available, output length, and the serving configuration. It keeps cold-start status and concurrent work visible. If continuous batching improves overall request throughput, the team still checks per-request latency rather than assuming both metrics move together.

The team also avoids calling chunked prefill a retrieval improvement. A scheduler can process a long manual efficiently while the retriever still supplies irrelevant sections. Conversely, smaller retrieved passages may improve evidence focus without changing the attention-cache format. These changes deserve separate comparisons. No universal cache lifetime, query throughput, or hardware-specific speedup follows from this scenario; those would require measurements or explicit service documentation.

## Sources

Accessed 2026-09-05. Sections 1–4 correspond to the sources below; the scenario is original.

1. [NVIDIA — Mastering LLM techniques: inference optimization](https://developer.nvidia.com/blog/?p=73739)
2. [Hugging Face — Continuous batching](https://huggingface.co/blog/continuous_batching)
3. [NVIDIA — Long-context inference with NVFP4 KV cache](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/)
4. [NVIDIA — Reduce KV cache bottlenecks with Dynamo](https://developer.nvidia.com/blog/?p=106133)
