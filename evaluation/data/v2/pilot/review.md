# Pilot annotation review packet

All cases are provisional and dev-only. No human review is claimed.
Record accepted/changed/rejected, reviewer identity and rationale; update manifest hashes after adjudication.

## pilot_001: semantic_discovery

我想让助手根据刚获得的信息决定下一步，而不是执行写死的步骤。找一篇能说明这种区别的笔记。

Family: `control`; answerability: `answerable`.

Draft criterion: 模型随新信息选择下一步动作，预定义步骤属于 workflow。

Facet `facet-1` — alternatives are OR; evidence within each is AND.

- `e-f7a85cf56b8e377f`: [01_workflows_and_agents.md](corpus/01_workflows_and_agents.md) body [14, 290)

> A workflow follows a sequence of steps chosen by its developer. An agent lets the model choose its next actions and tools as new information arrives. Predictable tasks often suit workflows, while open-ended tasks can benefit from an agent responding to environmental feedback.

Review: **pending**.

## pilot_002: semantic_discovery

工具名字相近、参数含糊，模型经常选错操作。找讨论接口该如何设计的材料。

Family: `interfaces`; answerability: `answerable`.

Draft criterion: 工具职责、命名、参数与输出都应明确。

Facet `facet-1` — alternatives are OR; evidence within each is AND.

- `e-ca90e588543db123`: [02_tool_interface_design.md](corpus/02_tool_interface_design.md) body [14, 341)

> A useful agent tool has a distinct purpose, an informative name, clear parameter descriptions, and outputs that help the agent decide what to do next. Overlapping operations and ambiguous identifiers make tool selection harder. Returning a focused result can be more useful than exposing every field from an underlying service.

Review: **pending**.

## pilot_003: semantic_discovery

助手启动时加载数百个操作的说明，占掉很多上下文。找能够减少这类占用的方案。

Family: `tool_loading`; answerability: `answerable`.

Draft criterion: 按需发现工具，再加载对应 schema。

Facet `facet-1` — alternatives are OR; evidence within each is AND.

- `e-c7d3be49d912150f`: [03_dynamic_tool_discovery.md](corpus/03_dynamic_tool_discovery.md) body [14, 323)

> An agent with a large tool catalog can waste its context window on definitions it never uses. Tool search keeps most definitions deferred, discovers relevant capabilities when needed, and then loads their schemas. This reduces the amount of tool documentation competing with the current task and conversation.

Review: **pending**.

## pilot_004: semantic_discovery

大量中间数据只需要求和与筛选，不值得让模型逐行阅读。哪里讨论了这种处理方式？

Family: `aggregation`; answerability: `answerable`.

Draft criterion: 在执行环境中处理数据，只返回摘要和必要证据。

Facet `facet-1` — alternatives are OR; evidence within each is AND.

- `e-724be3cbb4a9d64e`: [04_code_execution_for_tools.md](corpus/04_code_execution_for_tools.md) body [14, 368)

> Passing every intermediate tool result through a language model can fill its context with data that only needs filtering or aggregation. An execution environment can call connected services, process their results with ordinary code, and return a small summary. Loops, joins, and conditionals can run without a separate model decision for every operation.

Review: **pending**.

## pilot_005: semantic_discovery

有哪些材料介绍只先展示操作手册的简介，等用到时再展开详细步骤？

Family: `skill_loading`; answerability: `answerable`.

Draft criterion: 技能逐层加载入口、详细说明与附加资源。

Facet `facet-1` — alternatives are OR; evidence within each is AND.

- `e-d7936bf189a4ee87`: [05_agent_skills.md](corpus/05_agent_skills.md) body [14, 398)

> An Agent Skill packages task instructions with optional references and executable resources. The agent first sees the skill name and description, then reads the detailed instructions when the task calls for them. Additional reference files can remain outside the context until needed. This layered loading keeps specialized knowledge available without reading every manual at startup.

Review: **pending**.

## pilot_006: semantic_discovery

一次任务跨越多个上下文窗口，重启后容易忘记未解决的问题。找说明摘要需要保留什么的笔记。

Family: `handoff`; answerability: `answerable`.

Draft criterion: 保留决策、未解决问题和依赖。

Facet `facet-1` — alternatives are OR; evidence within each is AND.

- `e-bd1d5450234dacdc`: [06_context_compaction.md](corpus/06_context_compaction.md) body [14, 378)

> A long-running conversation can exceed the context available for the next model call. Compaction summarizes earlier history so work can continue with a smaller active context. Persistent notes keep selected facts outside that context for later retrieval. Useful summaries preserve decisions, unresolved problems, and dependencies while removing redundant material.

Review: **pending**.

## pilot_007: semantic_discovery

上一轮助手说任务做完了，下一轮应该凭什么检查这个说法？找相关笔记。

Family: `verification`; answerability: `answerable`.

Draft criterion: 用工作应用的实际检查核对完成状态，保留未完成工作。

Facet `facet-1` — alternatives are OR; evidence within each is AND.

- `e-6b915e99721de1f8`: [07_long_running_harnesses.md](corpus/07_long_running_harnesses.md) body [14, 333)

> An agent harness can help a coding project continue across fresh context windows. An initial session prepares the environment, a feature list, and progress records. Later sessions inspect those artifacts, complete a manageable part of the work, verify it, and leave an updated account of the state for the next session.

Review: **pending**.

## pilot_008: semantic_discovery

我想在工具返回结果之后安排一个专门的思考步骤。找讨论这类设计作用和边界的资料。

Family: `intermediate_reasoning`; answerability: `answerable`.

Draft criterion: 思考步骤用于复核新信息，本身不获取外部证据或修改环境。

Facet `facet-1` — alternatives are OR; evidence within each is AND.

- `e-031ea6d8ada180d0`: [08_think_tool.md](corpus/08_think_tool.md) body [14, 345)

> The think-tool pattern gives an agent a designated step to review newly received information during a tool-use sequence. It is especially relevant when the next action depends on earlier results or detailed policies. The tool provides a place for reasoning; it does not itself fetch new evidence or modify the external environment.

Review: **pending**.

## pilot_009: semantic_discovery

只限制助手写文件，是否已经限制了它访问外部服务？找能区分这些边界的材料。

Family: `isolation`; answerability: `answerable`.

Draft criterion: 文件系统和网络限制是不同的边界。

Facet `facet-1` — alternatives are OR; evidence within each is AND.

- `e-c75459113d523b86`: [09_agent_sandboxing.md](corpus/09_agent_sandboxing.md) body [14, 321)

> An execution sandbox restricts what an agent and its subprocesses can reach. Filesystem boundaries limit readable or writable locations, while network boundaries limit reachable services. These controls address different paths by which a mistaken or manipulated action can affect resources outside the task.

Review: **pending**.

## pilot_010: semantic_discovery

每个 AI 应用都要给相同数据源重写连接器。找试图统一这层接口的笔记。

Family: `protocol`; answerability: `answerable`.

Draft criterion: MCP 用客户端与服务端统一能力连接边界。

Facet `facet-1` — alternatives are OR; evidence within each is AND.

- `e-9b2cef28ae0a4b0d`: [10_mcp_connections.md](corpus/10_mcp_connections.md) body [14, 354)

> The Model Context Protocol defines a common interface for connecting AI applications to external capabilities and data. A server exposes an integration, while a client inside an AI application connects to that server. Standardizing this boundary reduces the need to build a different connector for every application and data-source pairing.

Review: **pending**.

## pilot_011: semantic_discovery

多次请求都带相同长说明，我想复用其处理计算。找讨论这种优化及其适用条件的笔记。

Family: `cache`; answerability: `answerable`.

Draft criterion: 后续请求共享重复 prompt 内容时，可以复用对应处理计算。

Facet `facet-1` — alternatives are OR; evidence within each is AND.

- `e-ec09df1599659f74`: [11_prompt_caching.md](corpus/11_prompt_caching.md) body [14, 295)

> Prompt caching reuses computation associated with prompt material that appears repeatedly across requests. Long instructions, document collections, or examples can form a reusable context portion. This can reduce processing cost and latency when later requests share that material.

Review: **pending**.

## pilot_012: semantic_discovery

一条资料中的实体指向另一条资料，希望沿关系收集证据。哪些笔记介绍了这种能力及其风险？

Family: `graph`; answerability: `answerable`.

Draft criterion: 图关系可连接多个片段，模型抽取的实体和关系仍需质量检查。

Facet `facet-1` — alternatives are OR; evidence within each is AND.

- `e-fc53d2dadabfa9ca`: [28_graph_retrieval.md](corpus/28_graph_retrieval.md) body [14, 376)

> A knowledge graph represents entities as nodes and their relationships as connections. A retrieval system can use these links to gather connected information across passages, which is useful for questions requiring several reasoning steps. Language models can help extract entities and relations from text, but the extracted structure still needs quality checks.

Review: **pending**.

## pilot_013: exploratory_retrieval

准备一份阅读清单，分别解释谁来决定下一步动作，以及工具结果到达后如何复核信息。

Family: `control`; answerability: `answerable`.

Draft criterion: 分别覆盖控制流自主性与工具调用间的复核。

Facet `facet-1` — alternatives are OR; evidence within each is AND.

- `e-f7a85cf56b8e377f`: [01_workflows_and_agents.md](corpus/01_workflows_and_agents.md) body [14, 290)

> A workflow follows a sequence of steps chosen by its developer. An agent lets the model choose its next actions and tools as new information arrives. Predictable tasks often suit workflows, while open-ended tasks can benefit from an agent responding to environmental feedback.

Facet `facet-2` — alternatives are OR; evidence within each is AND.

- `e-031ea6d8ada180d0`: [08_think_tool.md](corpus/08_think_tool.md) body [14, 345)

> The think-tool pattern gives an agent a designated step to review newly received information during a tool-use sequence. It is especially relevant when the next action depends on earlier results or detailed policies. The tool provides a place for reasoning; it does not itself fetch new evidence or modify the external environment.

Review: **pending**.

## pilot_014: exploratory_retrieval

我要设计外部能力集成，收集连接协议的职责划分和模型工具接口的设计要点。

Family: `interfaces`; answerability: `answerable`.

Draft criterion: 覆盖客户端/服务端边界和工具名称参数输出设计。

Facet `facet-1` — alternatives are OR; evidence within each is AND.

- `e-9b2cef28ae0a4b0d`: [10_mcp_connections.md](corpus/10_mcp_connections.md) body [14, 354)

> The Model Context Protocol defines a common interface for connecting AI applications to external capabilities and data. A server exposes an integration, while a client inside an AI application connects to that server. Standardizing this boundary reduces the need to build a different connector for every application and data-source pairing.

Facet `facet-2` — alternatives are OR; evidence within each is AND.

- `e-ca90e588543db123`: [02_tool_interface_design.md](corpus/02_tool_interface_design.md) body [14, 341)

> A useful agent tool has a distinct purpose, an informative name, clear parameter descriptions, and outputs that help the agent decide what to do next. Overlapping operations and ambiguous identifiers make tool selection harder. Returning a focused result can be more useful than exposing every field from an underlying service.

Review: **pending**.

## pilot_015: exploratory_retrieval

比较大工具目录和任务手册怎样按需加载，分别给出相关笔记。

Family: `tool_loading`; answerability: `answerable`.

Draft criterion: 区分工具发现/schema 加载与技能说明分层加载。

Facet `facet-1` — alternatives are OR; evidence within each is AND.

- `e-c7d3be49d912150f`: [03_dynamic_tool_discovery.md](corpus/03_dynamic_tool_discovery.md) body [14, 323)

> An agent with a large tool catalog can waste its context window on definitions it never uses. Tool search keeps most definitions deferred, discovers relevant capabilities when needed, and then loads their schemas. This reduces the amount of tool documentation competing with the current task and conversation.

Facet `facet-2` — alternatives are OR; evidence within each is AND.

- `e-d7936bf189a4ee87`: [05_agent_skills.md](corpus/05_agent_skills.md) body [14, 398)

> An Agent Skill packages task instructions with optional references and executable resources. The agent first sees the skill name and description, then reads the detailed instructions when the task calls for them. Additional reference files can remain outside the context until needed. This layered loading keeps specialized knowledge available without reading every manual at startup.

Review: **pending**.

## pilot_016: exploratory_retrieval

我需要降低处理重复内容和中间数据的费用，分别找计算缓存与代码聚合的资料。

Family: `aggregation`; answerability: `answerable`.

Draft criterion: 缓存重复 prompt 计算；代码处理工具返回的数据。

Facet `facet-1` — alternatives are OR; evidence within each is AND.

- `e-ec09df1599659f74`: [11_prompt_caching.md](corpus/11_prompt_caching.md) body [14, 295)

> Prompt caching reuses computation associated with prompt material that appears repeatedly across requests. Long instructions, document collections, or examples can form a reusable context portion. This can reduce processing cost and latency when later requests share that material.

Facet `facet-2` — alternatives are OR; evidence within each is AND.

- `e-724be3cbb4a9d64e`: [04_code_execution_for_tools.md](corpus/04_code_execution_for_tools.md) body [14, 368)

> Passing every intermediate tool result through a language model can fill its context with data that only needs filtering or aggregation. An execution environment can call connected services, process their results with ordinary code, and return a small summary. Loops, joins, and conditionals can run without a separate model decision for every operation.

Review: **pending**.

## pilot_017: exploratory_retrieval

准备一份跨会话任务交接指南，收集摘要保留内容与下一轮工作检查流程。

Family: `handoff`; answerability: `answerable`.

Draft criterion: 摘要保存约束与未完成事项；新会话检查交接产物并验证工作。

Facet `facet-1` — alternatives are OR; evidence within each is AND.

- `e-bd1d5450234dacdc`: [06_context_compaction.md](corpus/06_context_compaction.md) body [14, 378)

> A long-running conversation can exceed the context available for the next model call. Compaction summarizes earlier history so work can continue with a smaller active context. Persistent notes keep selected facts outside that context for later retrieval. Useful summaries preserve decisions, unresolved problems, and dependencies while removing redundant material.

Facet `facet-2` — alternatives are OR; evidence within each is AND.

- `e-6b915e99721de1f8`: [07_long_running_harnesses.md](corpus/07_long_running_harnesses.md) body [14, 333)

> An agent harness can help a coding project continue across fresh context windows. An initial session prepares the environment, a feature list, and progress records. Later sessions inspect those artifacts, complete a manageable part of the work, verify it, and leave an updated account of the state for the next session.

Review: **pending**.

## pilot_018: exploratory_retrieval

评估一个助手是否可靠，找关于实际完成检查与重复尝试统计的两类资料。

Family: `verification`; answerability: `answerable`.

Draft criterion: 不能只信完成声明；重复尝试中的失败应保留。

Facet `facet-1` — alternatives are OR; evidence within each is AND.

- `e-6b915e99721de1f8`: [07_long_running_harnesses.md](corpus/07_long_running_harnesses.md) body [14, 333)

> An agent harness can help a coding project continue across fresh context windows. An initial session prepares the environment, a feature list, and progress records. Later sessions inspect those artifacts, complete a manageable part of the work, verify it, and leave an updated account of the state for the next session.

Facet `facet-2` — alternatives are OR; evidence within each is AND.

- `e-e31334e0bf7042a7`: [27_agent_trial_reliability.md](corpus/27_agent_trial_reliability.md) body [14, 348)

> An agent can behave differently across repeated attempts at the same task. The pass-at-k view asks whether at least one of several attempts succeeds. The all-trials-success view asks whether every attempt succeeds. These measure different product needs: finding one working solution and delivering dependable behavior on repeated use.

Review: **pending**.

## pilot_019: exploratory_retrieval

我想比较两种压缩向量的方法，请收集分别减少坐标数和降低每个坐标精度的材料。

Family: `vectors`; answerability: `answerable`.

Draft criterion: 截短维度与量化数值精度是不同操作。

Facet `facet-1` — alternatives are OR; evidence within each is AND.

- `e-a6434acddd07ee92`: [16_matryoshka_embeddings.md](corpus/16_matryoshka_embeddings.md) body [14, 369)

> Matryoshka embedding models are trained so useful information is retained in shorter prefixes of their output vectors. A system can use fewer dimensions for a cheaper initial search and retain longer vectors for a more detailed comparison. This behavior depends on the training method; arbitrary embedding dimensions are not automatically interchangeable.

Facet `facet-2` — alternatives are OR; evidence within each is AND.

- `e-aac3b880c4396423`: [17_embedding_quantization.md](corpus/17_embedding_quantization.md) body [14, 381)

> Embedding quantization stores vector coordinates at lower numerical precision. Scalar quantization can map floating-point values into integer buckets, while binary quantization retains one bit per coordinate. Calibration data affects the scalar bucket boundaries. A search can use compressed vectors for candidate selection and higher-precision vectors for rescoring.

Review: **pending**.

## pilot_020: exploratory_retrieval

做模型选型时，我需要区分 embedding 任务覆盖与短事实问答的评测边界，请找两类材料。

Family: `benchmark_scope`; answerability: `answerable`.

Draft criterion: embedding 多任务成绩与短事实问答不能替代目标应用测量。

Facet `facet-1` — alternatives are OR; evidence within each is AND.

- `e-4ed44f5b94be2abc`: [18_embedding_benchmarks.md](corpus/18_embedding_benchmarks.md) body [14, 345)

> The Massive Text Embedding Benchmark evaluates embeddings across tasks such as retrieval, classification, clustering, and semantic similarity. These tasks test different uses of the same numerical representation. A model that performs well on one task or language does not thereby establish its quality for every other application.

Facet `facet-2` — alternatives are OR; evidence within each is AND.

- `e-c734d829e5cc565f`: [26_simpleqa_factuality.md](corpus/26_simpleqa_factuality.md) body [14, 333)

> SimpleQA focuses on short factual questions with a single verifiable answer. Its reference construction uses independent checking, and its grading separates correct, incorrect, and unattempted answers. Restricting answer length makes factual checking more manageable than grading a long response containing many claims.

Review: **pending**.

## pilot_021: exploratory_retrieval

准备模型适配资料，分别说明小参数增量如何部署，以及偏好样本如何影响学习。

Family: `training`; answerability: `answerable`.

Draft criterion: adapter 依赖基座；DPO 的偏好对质量决定学习信号。

Facet `facet-1` — alternatives are OR; evidence within each is AND.

- `e-e7a12f55e6dcc6e4`: [19_parameter_efficient_tuning.md](corpus/19_parameter_efficient_tuning.md) body [14, 337)

> Parameter-efficient fine-tuning adapts a pretrained model while keeping most of its parameters frozen. Methods such as LoRA train a small set of additional parameters instead of producing a fully updated copy of every base weight. The resulting task-specific checkpoint can be much smaller than a complete model checkpoint.

Facet `facet-2` — alternatives are OR; evidence within each is AND.

- `e-e29d9ed2f9c0144f`: [20_direct_preference_optimization.md](corpus/20_direct_preference_optimization.md) body [14, 358)

> Direct Preference Optimization trains a language model with pairs of preferred and rejected responses to the same prompt. Its objective incorporates a reference model and increases the relative preference for the chosen response. It avoids separately fitting an explicit reward model and then running a reinforcement-learning optimization loop.

Review: **pending**.

## pilot_022: exploratory_retrieval

做生成服务优化阅读清单，区分重复输入计算复用和草稿 token 验证两种办法。

Family: `cache`; answerability: `answerable`.

Draft criterion: prompt 缓存与 speculative decoding 优化不同阶段。

Facet `facet-1` — alternatives are OR; evidence within each is AND.

- `e-ec09df1599659f74`: [11_prompt_caching.md](corpus/11_prompt_caching.md) body [14, 295)

> Prompt caching reuses computation associated with prompt material that appears repeatedly across requests. Long instructions, document collections, or examples can form a reusable context portion. This can reduce processing cost and latency when later requests share that material.

Facet `facet-2` — alternatives are OR; evidence within each is AND.

- `e-a731b3909e6814c2`: [21_speculative_decoding.md](corpus/21_speculative_decoding.md) body [14, 340)

> Speculative decoding uses a cheaper approximation, such as a small model, to propose upcoming tokens. A target model checks those proposals in parallel. The acceptance and correction procedure preserves the target sampling distribution while allowing multiple tokens to be produced through fewer sequential target-model steps.

Review: **pending**.

## pilot_023: exploratory_retrieval

比较监督信号来自中间推理步骤还是较弱模型的两种研究设定，请找对应材料并说明外推限制。

Family: `oversight`; answerability: `answerable`.

Draft criterion: 区分过程反馈和弱监督，不能直接外推到所有领域或未来超强模型。

Facet `facet-1` — alternatives are OR; evidence within each is AND.

- `e-8f106faecb0985a4`: [22_process_supervision.md](corpus/22_process_supervision.md) body [14, 329)

> Outcome supervision evaluates a final result. Process supervision provides feedback on intermediate reasoning steps, making it possible to identify where a solution becomes incorrect. OpenAI compared these approaches on mathematical problems and found benefits from process-supervised reward models in that setting.

Facet `facet-2` — alternatives are OR; evidence within each is AND.

- `e-69774729e73d97b8`: [23_weak_to_strong_generalization.md](corpus/23_weak_to_strong_generalization.md) body [14, 312)

> Weak-to-strong generalization studies whether a more capable pretrained model can learn from a less capable supervisor without simply inheriting all of its mistakes. The research uses smaller models supervising larger models as a tractable analogy for oversight by humans with limited capabilities.

Review: **pending**.

## pilot_024: exploratory_retrieval

收集两种避免片段脱离语境的方法：改切分边界和为片段补局部背景。

Family: `structure`; answerability: `answerable`.

Draft criterion: 边界保留完整含义；补充片段特定背景应对照原文。

Facet `facet-1` — alternatives are OR; evidence within each is AND.

- `e-3a512fa4908fbf98`: [15_chunk_boundaries.md](corpus/15_chunk_boundaries.md) body [14, 374)

> Chunking determines the unit stored and retrieved from a document. Fixed-size splitting controls length, while content-aware splitting follows boundaries such as paragraphs or sections. A fragment should retain enough context to be useful when retrieved independently. Very small fragments can lose meaning, while large fragments can include unrelated content.

Facet `facet-2` — alternatives are OR; evidence within each is AND.

- `e-44c7d3085e14752d`: [12_contextual_retrieval.md](corpus/12_contextual_retrieval.md) body [14, 319)

> A retrieved passage may contain a useful fact but omit the company, subject, or period needed to interpret it. Contextual Retrieval prepends a short explanation specific to that passage before building its embedding and lexical index entry. The explanation situates the passage within the larger document.

Review: **pending**.

## pilot_025: knowledge_qa

给每个检索片段都加相同的文档摘要，是否符合笔记对补充背景的建议？

Family: `context`; answerability: `answerable`.

Draft criterion: 应补片段特定的背景，而不是统一通用摘要。

Facet `facet-1` — alternatives are OR; evidence within each is AND.

- `e-10b3ecdb7221f5a1`: [12_contextual_retrieval.md](corpus/12_contextual_retrieval.md) body [347, 527)

> Use context that resolves the particular passage, rather than attaching the same generic summary everywhere. Check the added text against the source before treating it as evidence.

Review: **pending**.

## pilot_026: knowledge_qa

融合词法和向量检索时，为什么不能把融合后的分数当成余弦相似度？

Family: `fusion`; answerability: `answerable`.

Draft criterion: RRF 以排名位置融合，结果是新的排序分数。

Facet `facet-1` — alternatives are OR; evidence within each is AND.

- `e-6624bc6c49e72912`: [13_hybrid_rank_fusion.md](corpus/13_hybrid_rank_fusion.md) body [363, 531)

> Evaluate hybrid retrieval on both exact terminology and paraphrases. A fused result has a new ranking score whose interpretation differs from a cosine similarity value.

Review: **pending**.

## pilot_027: knowledge_qa

重排候选已经包含了相关片段之外的材料，它能否把完全没召回的证据变出来？

Family: `candidate_limit`; answerability: `answerable`.

Draft criterion: 重排只能重排候选集，不能恢复不在候选中的证据。

Facet `facet-1` — alternatives are OR; evidence within each is AND.

- `e-7a644dda34e94981`: [14_cross_encoder_reranking.md](corpus/14_cross_encoder_reranking.md) body [382, 535)

> Measure initial candidate coverage before evaluating reranking. Reordering a shortlist cannot recover a useful passage that never entered that shortlist.

- `e-candidate-alternative`: [40_query_planning_and_compression.md](corpus/40_query_planning_and_compression.md) body [2102, 2961)

> Microsoft's discussion of query rewriting and semantic ranking places rewriting before first-stage retrieval. That initial stage can use lexical, vector, or hybrid search to collect candidates. A later semantic ranker evaluates the shortlist more closely; a cross-encoder judges the query and candidate together rather than relying only on independently stored embeddings. These stages address complementary problems. Query rewriting can help useful evidence enter the candidate set, while reranking can improve its ordering once present. Reranking cannot recover a passage that never entered its shortlist. Its more expensive joint scoring is therefore applied to a limited set. An evaluation should inspect candidate coverage as well as final ordering so that missing evidence is not misdiagnosed as a ranking-only problem. This section summarizes source 3.

Review: **pending**.

## pilot_028: knowledge_qa

一个切片符合长度上限，却把定义与后续条件分开，为什么仍需调整？

Family: `structure`; answerability: `answerable`.

Draft criterion: 长度约束不保证语义完整，应保留限定条件的邻近上下文。

Facet `facet-1` — alternatives are OR; evidence within each is AND.

- `e-a539e5fec4569f99`: [15_chunk_boundaries.md](corpus/15_chunk_boundaries.md) body [402, 577)

> Test chunking against actual questions and document structure. Preserve nearby context when a definition or condition would otherwise be separated from the claim it qualifies.

Review: **pending**.

## pilot_029: knowledge_qa

对于需要单位向量的相似度计算，把已经归一化的向量截短后还要做什么？

Family: `vectors`; answerability: `answerable`.

Draft criterion: 在相似度要求单位长度时，截短后重新归一化。

Facet `facet-1` — alternatives are OR; evidence within each is AND.

- `e-42546abc40f0a3bc`: [16_matryoshka_embeddings.md](corpus/16_matryoshka_embeddings.md) body [397, 564)

> Measure quality at the chosen dimension. After truncating a normalized vector, normalize the shorter vector again when the similarity calculation requires unit length.

Review: **pending**.

## pilot_030: knowledge_qa

标量量化的桶边界受什么影响？它与每坐标只保留一位的方式有何区别？

Family: `quantization`; answerability: `answerable`.

Draft criterion: 标量桶受校准数据影响；二值量化每坐标一位。

Facet `facet-1` — alternatives are OR; evidence within each is AND.

- `e-aac3b880c4396423`: [17_embedding_quantization.md](corpus/17_embedding_quantization.md) body [14, 381)

> Embedding quantization stores vector coordinates at lower numerical precision. Scalar quantization can map floating-point values into integer buckets, while binary quantization retains one bit per coordinate. Calibration data affects the scalar bucket boundaries. A search can use compressed vectors for candidate selection and higher-precision vectors for rescoring.

Review: **pending**.

## pilot_031: knowledge_qa

只拿到一个很小的 LoRA adapter，就能不依赖原模型部署吗？

Family: `training`; answerability: `answerable`.

Draft criterion: 不能，部署还需要基座模型。

Facet `facet-1` — alternatives are OR; evidence within each is AND.

- `e-db5422e38921654d`: [19_parameter_efficient_tuning.md](corpus/19_parameter_efficient_tuning.md) body [365, 538)

> Account for the base model as well as the adapter when deploying. A small adapter represents a learned change, rather than a standalone replacement for the pretrained model.

Review: **pending**.

## pilot_032: knowledge_qa

偏好训练中的 chosen/rejected 标签互相矛盾，会带来什么问题？

Family: `preferences`; answerability: `answerable`.

Draft criterion: 偏好对决定学习信号，矛盾或缺乏依据的偏好会教出不良行为。

Facet `facet-1` — alternatives are OR; evidence within each is AND.

- `e-0d872c30d8458ed5`: [20_direct_preference_optimization.md](corpus/20_direct_preference_optimization.md) body [386, 559)

> Inspect preference pairs carefully. The learning signal comes from which response is chosen, so contradictory or poorly justified preferences can teach undesirable behavior.

Review: **pending**.

## pilot_033: knowledge_qa

使用较便宜的小模型起草 token，是在给目标模型训练新知识吗？

Family: `decoding`; answerability: `answerable`.

Draft criterion: 不是；目标模型检查草稿，旨在减少顺序执行的目标模型步骤。

Facet `facet-1` — alternatives are OR; evidence within each is AND.

- `e-a731b3909e6814c2`: [21_speculative_decoding.md](corpus/21_speculative_decoding.md) body [14, 340)

> Speculative decoding uses a cheaper approximation, such as a small model, to propose upcoming tokens. A target model checks those proposals in parallel. The acceptance and correction procedure preserves the target sampling distribution while allowing multiple tokens to be produced through fewer sequential target-model steps.

Review: **pending**.

## pilot_034: knowledge_qa

评估安全行为时，为什么不能只追求更高的拒绝率？

Family: `calibration`; answerability: `answerable`.

Draft criterion: 需要同时评估有害服从和对正常请求的误拒。

Facet `facet-1` — alternatives are OR; evidence within each is AND.

- `e-c61e19cd85b3d9ea`: [24_deliberative_alignment.md](corpus/24_deliberative_alignment.md) body [454, 610)

> Evaluate both harmful compliance and refusal of benign requests. Correct calibration requires distinguishing the cases rather than maximizing refusal alone.

Review: **pending**.

## pilot_035: knowledge_qa

如果错误答案与放弃回答都得零分，评测规则可能鼓励什么行为？

Family: `uncertainty`; answerability: `answerable`.

Draft criterion: 不确定时猜测，因此应区分正确、无支持错误和适当不确定。

Facet `facet-1` — alternatives are OR; evidence within each is AND.

- `e-a2c3e77425b4aa5a`: [25_hallucination_incentives.md](corpus/25_hallucination_incentives.md) body [368, 563)

> Distinguish correct answers, unsupported errors, and appropriate uncertainty in evaluation. A system should not gain an advantage merely by making confident guesses on questions it cannot answer.

Review: **pending**.

## pilot_036: knowledge_qa

短事实问答得分很高，是否就能说明长篇多论断回答也可靠？

Family: `benchmark_scope`; answerability: `answerable`.

Draft criterion: 不能，短问题的事实性评测不直接证明长篇可靠性。

Facet `facet-1` — alternatives are OR; evidence within each is AND.

- `e-b724556c3ceab292`: [26_simpleqa_factuality.md](corpus/26_simpleqa_factuality.md) body [361, 528)

> Keep the benchmark scope in view. Performance on concise fact-seeking questions does not by itself establish the factual reliability of long, multi-claim explanations.

Review: **pending**.

## pilot_037: multi_hop_qa

虚构服务 F1 当前普通记录与事件记录分别保留多久？沿服务说明查找现行规范，并排除旧版本。

Family: `synthetic_retention_family`; answerability: `answerable`.

Draft criterion: 先确认服务所引用的现行规范：普通记录 21 天，事件记录 3 天；旧的 81 天不适用。

Facet `joint-evidence` — alternatives are OR; evidence within each is AND.

- `e-721cf72ba095cda9`: [pilot_service_1.md](corpus/pilot_service_1.md) body [0, 129)

> Synthetic evaluation fixture. Service F1 delegates its retention policy to pilot_policy_1.md. The archived policy is not current.

- `e-cfcd2712f06eddaf`: [pilot_policy_1.md](corpus/pilot_policy_1.md) body [0, 181)

> Synthetic evaluation fixture. The current retention period is 21 days. For incident records, the exception is 3 days. Both rules apply only to the service that links to this policy.

Review: **pending**.

## pilot_038: multi_hop_qa

虚构服务 F2 当前普通记录与事件记录分别保留多久？沿服务说明查找现行规范，并排除旧版本。

Family: `synthetic_retention_family`; answerability: `answerable`.

Draft criterion: 先确认服务所引用的现行规范：普通记录 22 天，事件记录 4 天；旧的 82 天不适用。

Facet `joint-evidence` — alternatives are OR; evidence within each is AND.

- `e-b4c27d3a4d3682d4`: [pilot_service_2.md](corpus/pilot_service_2.md) body [0, 129)

> Synthetic evaluation fixture. Service F2 delegates its retention policy to pilot_policy_2.md. The archived policy is not current.

- `e-33c69cb7b78f2a59`: [pilot_policy_2.md](corpus/pilot_policy_2.md) body [0, 181)

> Synthetic evaluation fixture. The current retention period is 22 days. For incident records, the exception is 4 days. Both rules apply only to the service that links to this policy.

Review: **pending**.

## pilot_039: multi_hop_qa

虚构服务 F3 当前普通记录与事件记录分别保留多久？沿服务说明查找现行规范，并排除旧版本。

Family: `synthetic_retention_family`; answerability: `answerable`.

Draft criterion: 先确认服务所引用的现行规范：普通记录 23 天，事件记录 5 天；旧的 83 天不适用。

Facet `joint-evidence` — alternatives are OR; evidence within each is AND.

- `e-94fdf7880774b5ed`: [pilot_service_3.md](corpus/pilot_service_3.md) body [0, 129)

> Synthetic evaluation fixture. Service F3 delegates its retention policy to pilot_policy_3.md. The archived policy is not current.

- `e-2a1364b7672fab10`: [pilot_policy_3.md](corpus/pilot_policy_3.md) body [0, 181)

> Synthetic evaluation fixture. The current retention period is 23 days. For incident records, the exception is 5 days. Both rules apply only to the service that links to this policy.

Review: **pending**.

## pilot_040: multi_hop_qa

虚构服务 F4 当前普通记录与事件记录分别保留多久？沿服务说明查找现行规范，并排除旧版本。

Family: `synthetic_retention_family`; answerability: `answerable`.

Draft criterion: 先确认服务所引用的现行规范：普通记录 24 天，事件记录 6 天；旧的 84 天不适用。

Facet `joint-evidence` — alternatives are OR; evidence within each is AND.

- `e-ccd89664b7aa497a`: [pilot_service_4.md](corpus/pilot_service_4.md) body [0, 129)

> Synthetic evaluation fixture. Service F4 delegates its retention policy to pilot_policy_4.md. The archived policy is not current.

- `e-67a2ae68af499ec4`: [pilot_policy_4.md](corpus/pilot_policy_4.md) body [0, 181)

> Synthetic evaluation fixture. The current retention period is 24 days. For incident records, the exception is 6 days. Both rules apply only to the service that links to this policy.

Review: **pending**.

## pilot_041: multi_hop_qa

虚构服务 F5 当前普通记录与事件记录分别保留多久？沿服务说明查找现行规范，并排除旧版本。

Family: `synthetic_retention_family`; answerability: `answerable`.

Draft criterion: 先确认服务所引用的现行规范：普通记录 25 天，事件记录 7 天；旧的 85 天不适用。

Facet `joint-evidence` — alternatives are OR; evidence within each is AND.

- `e-c5f9b74d800baf8e`: [pilot_service_5.md](corpus/pilot_service_5.md) body [0, 129)

> Synthetic evaluation fixture. Service F5 delegates its retention policy to pilot_policy_5.md. The archived policy is not current.

- `e-d5032110c447a87d`: [pilot_policy_5.md](corpus/pilot_policy_5.md) body [0, 181)

> Synthetic evaluation fixture. The current retention period is 25 days. For incident records, the exception is 7 days. Both rules apply only to the service that links to this policy.

Review: **pending**.

## pilot_042: multi_hop_qa

虚构服务 F6 当前普通记录与事件记录分别保留多久？沿服务说明查找现行规范，并排除旧版本。

Family: `synthetic_retention_family`; answerability: `answerable`.

Draft criterion: 先确认服务所引用的现行规范：普通记录 26 天，事件记录 8 天；旧的 86 天不适用。

Facet `joint-evidence` — alternatives are OR; evidence within each is AND.

- `e-78e1526fe3bb7f8e`: [pilot_service_6.md](corpus/pilot_service_6.md) body [0, 129)

> Synthetic evaluation fixture. Service F6 delegates its retention policy to pilot_policy_6.md. The archived policy is not current.

- `e-7c5790aae297358a`: [pilot_policy_6.md](corpus/pilot_policy_6.md) body [0, 181)

> Synthetic evaluation fixture. The current retention period is 26 days. For incident records, the exception is 8 days. Both rules apply only to the service that links to this policy.

Review: **pending**.

## pilot_043: exact_lookup

列出正文中匹配 'MCP' 的所有笔记；使用字面子串匹配，区分大小写。没有匹配请明确说明。

Family: `exact_enumeration`; answerability: `answerable`.

Draft criterion: 列出的文件集合必须与实际匹配完全一致。

Review: **pending**.

## pilot_044: exact_lookup

列出正文中匹配 'LoRA' 的所有笔记；使用字面子串匹配，区分大小写。没有匹配请明确说明。

Family: `exact_enumeration`; answerability: `answerable`.

Draft criterion: 列出的文件集合必须与实际匹配完全一致。

Review: **pending**.

## pilot_045: exact_lookup

列出正文中匹配 'prefixes' 的所有笔记；使用字面子串匹配，区分大小写。没有匹配请明确说明。

Family: `exact_enumeration`; answerability: `answerable`.

Draft criterion: 列出的文件集合必须与实际匹配完全一致。

Review: **pending**.

## pilot_046: exact_lookup

列出文件名中匹配 '^1[67]_' 的所有笔记；使用正则匹配，区分大小写。没有匹配请明确说明。

Family: `exact_enumeration`; answerability: `answerable`.

Draft criterion: 列出的文件集合必须与实际匹配完全一致。

Review: **pending**.

## pilot_047: exact_lookup

列出正文中匹配 'ARKB-PILOT-NO-SUCH-TOKEN-20260910' 的所有笔记；使用字面子串匹配，区分大小写。没有匹配请明确说明。

Family: `exact_enumeration`; answerability: `answerable`.

Draft criterion: 列出的文件集合必须与实际匹配完全一致。

Review: **pending**.

## pilot_048: exact_lookup

列出正文中匹配 'scalar quantization' 的所有笔记；使用字面子串匹配，不区分大小写。没有匹配请明确说明。

Family: `exact_enumeration`; answerability: `answerable`.

Draft criterion: 列出的文件集合必须与实际匹配完全一致。

Review: **pending**.

## pilot_049: direct_read

只读取 10_mcp_connections.md，说明客户端与服务端的分工。 不要搜索。

Family: `protocol`; answerability: `answerable`.

Draft criterion: 服务端暴露集成；AI 应用内客户端连接它。

Facet `facet-1` — alternatives are OR; evidence within each is AND.

- `e-9b2cef28ae0a4b0d`: [10_mcp_connections.md](corpus/10_mcp_connections.md) body [14, 354)

> The Model Context Protocol defines a common interface for connecting AI applications to external capabilities and data. A server exposes an integration, while a client inside an AI application connects to that server. Standardizing this boundary reduces the need to build a different connector for every application and data-source pairing.

Review: **pending**.

## pilot_050: direct_read

只读取 19_parameter_efficient_tuning.md，说明部署小 adapter 时是否还需要基座模型。 不要搜索。

Family: `training`; answerability: `answerable`.

Draft criterion: 小 adapter 是增量，部署仍需基座。

Facet `facet-1` — alternatives are OR; evidence within each is AND.

- `e-db5422e38921654d`: [19_parameter_efficient_tuning.md](corpus/19_parameter_efficient_tuning.md) body [365, 538)

> Account for the base model as well as the adapter when deploying. A small adapter represents a learned change, rather than a standalone replacement for the pretrained model.

Review: **pending**.

## pilot_051: direct_read

只读取 25_hallucination_incentives.md，说明评测应如何区分猜测与适当的不确定。 不要搜索。

Family: `uncertainty`; answerability: `answerable`.

Draft criterion: 区分正确、无支持错误和适当不确定。

Facet `facet-1` — alternatives are OR; evidence within each is AND.

- `e-a2c3e77425b4aa5a`: [25_hallucination_incentives.md](corpus/25_hallucination_incentives.md) body [368, 563)

> Distinguish correct answers, unsupported errors, and appropriate uncertainty in evaluation. A system should not gain an advantage merely by making confident guesses on questions it cannot answer.

Review: **pending**.

## pilot_052: direct_read

只读取 17_embedding_quantization.md，说明量化与减少维度是不是同一件事。 不要搜索。

Family: `quantization`; answerability: `answerable`.

Draft criterion: 降低精度与减少维度不同。

Facet `facet-1` — alternatives are OR; evidence within each is AND.

- `e-d9f1caeca1d20834`: [17_embedding_quantization.md](corpus/17_embedding_quantization.md) body [409, 564)

> Measure the quality and memory trade-off on representative data. Lower precision and fewer dimensions are different changes to an embedding representation.

Review: **pending**.

## pilot_053: evidence_gap

根据知识库，给出这个本地系统昨天测得的平均检索延迟，精确到毫秒。

Family: `missing_measurement`; answerability: `unanswerable`.

Draft criterion: 笔记没有提供本系统昨天的延迟测量，不能用一般性说明推算。

Review: **pending**.

## pilot_054: evidence_gap

知识库记录的虚构项目 ZX-NOT-RECORDED 的验收口令是什么？

Family: `missing_private_fact`; answerability: `unanswerable`.

Draft criterion: 知识库没有这个项目或口令，明确无法确定。

Review: **pending**.

## pilot_055: evidence_gap

解释截短 embedding 维度后的归一化要求，并报告本系统截短后实际节省了百分之多少延迟。

Family: `vectors`; answerability: `partial`.

Draft criterion: 可以说明必要时重新归一化；本系统实测延迟节省比例未知，分开回答。

Facet `facet-1` — alternatives are OR; evidence within each is AND.

- `e-42546abc40f0a3bc`: [16_matryoshka_embeddings.md](corpus/16_matryoshka_embeddings.md) body [397, 564)

> Measure quality at the chosen dimension. After truncating a normalized vector, normalize the shorter vector again when the similarity calculation requires unit length.

Review: **pending**.

## pilot_056: evidence_gap

请告诉我那个服务的新版保留期限。

Family: `ambiguous_version`; answerability: `needs_clarification`.

Draft criterion: 缺少服务身份，应询问具体服务，而不是任选 F1 到 F6。

Review: **pending**.

## pilot_057: no_retrieval

只按给出的数字计算：4 加 9 等于多少？

Family: `provided_text`; answerability: `not_applicable`.

Draft criterion: 13

Review: **pending**.

## pilot_058: no_retrieval

只处理给定列表，按字母排序：delta, alpha, gamma。

Family: `provided_text`; answerability: `not_applicable`.

Draft criterion: alpha, delta, gamma

Review: **pending**.

## pilot_059: no_retrieval

不查资料，把“延迟降低了”翻译成英文。

Family: `provided_text`; answerability: `not_applicable`.

Draft criterion: Latency decreased.

Review: **pending**.

## pilot_060: no_retrieval

只按这段话回答：方案甲用 2 秒，方案乙用 5 秒。哪个更快？

Family: `provided_text`; answerability: `not_applicable`.

Draft criterion: 方案甲

Review: **pending**.
