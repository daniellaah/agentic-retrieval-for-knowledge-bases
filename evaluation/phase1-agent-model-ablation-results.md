# Phase 1：Agent Model Ablation 实测报告

**正式矩阵 3 models × 40 cases × 3 trials = 360 次真实 Agent 运行，全部完成并通过复算与控制变量核验。**
实验 ID：`20260909-phase1-formal`。运行时间：2026-09-09 13:19:53–16:45:50 America/Los_Angeles（20:19:53–23:45:50 UTC，约 3 小时 26 分钟）。

27B 的任务成功率为 **95.0%**，探索检索成功率为 **100%**，本次没有 max-turn 或运行错误；4B 对应为 77.5%、40.0%。证据支持 Agent 模型能力是当前**多步检索与停止控制的重要瓶颈之一**。但工具调用成本未随容量单调降低，27B 的总召回率也低于 9B，不能把结果归纳为所有指标都随模型变大而改善。

建议后续 Embedding Ablation 固定 **`qwen3.5:27b`**，以减少 Agent 停止失败和工具参数错误的干扰。其平均运行延迟约为 9B 的 3.06 倍，这是一项实验质量与运行成本的取舍。本阶段没有改动产品默认模型、Agent 或检索策略。

## 实验配置与当前模型模块

| 模块 / 变量 | 实际配置 |
| --- | --- |
| Agent generation / 工具选择、查询、阅读、停止 | `qwen3.5:4b`、`qwen3.5:9b`、`qwen3.5:27b`，仅此变量改变 |
| 向量检索的文档 / 查询 embedding | `qwen3-embedding:0.6b`，1024 维，沿用原快照的模型 digest 与 tokenizer |
| 实验当时的 Reranker 配置 | `cross-encoder/ms-marco-MiniLM-L6-v2`；该次实验的 Runtime.ask 路径始终禁用 reranking |
| Dataset | `evaluation/data/agent_v1.jsonl`，40 个原始 case，不改标签 |
| Dataset SHA-256 | `8babda40f2a74506c3fefe5159a9094a91c86e4e5562443a09c7b35790a0e8af` |
| 语料 / 快照 | 同一 example_notes：40 documents、160 chunks；`facaea4136d1421389ca640051e1e55c` |
| Vault / Qdrant collection | `agent-eval-integration` / `obsidian_rag_336df70314593b9703bda55d8fe6640b` |
| Trials | 每个 case / model 3 次，单次新建对话状态；无重试 |
| Agent 控制 | `think=True`、`max_turns=8`、`temperature=0`、`stream=False` |
| 其余采样参数 | 三个模型 provider 参数相同：presence_penalty=1.5、top_k=20、top_p=0.95；provider temperature=1 被 Runtime 的 0 覆盖 |
| 上下文 / token 限制 | Runtime 未指定 num_ctx、num_predict、seed；三个模型实际加载 context 均观测为 262144 |
| 检索配置 | 默认 semantic，同一工具可选 bm25 / semantic / hybrid；candidate_k=20、RRF k=60、BM25 k1=1.2、b=0.75 |
| Reranker 参数（未启用） | 候选 20，max_length=512，revision=233902d25c440f23af6f7d6e94d2946bac0bee0a |
| 硬件 / Runtime | Apple M4 Max、128 GiB、arm64；Python 3.13.5、Ollama server 0.33.2、Ollama Python 0.6.2、Qdrant 1.19.0 |
| 代码版本 | `eval` 分支，commit `4a31281515e2f014b8c82b97b1b73dde11e4a89d`，加本次 evaluation 编排 / 分析代码，完整 source hashes 已保存 |

模型使用准确 tag，没有替代模型；9B、27B 在用户授权后下载。三个 Agent 模型均为 Q4_K_M，其制品 digest 为：

| 模型 | Digest |
| --- | --- |
| qwen3.5:4b | `2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd` |
| qwen3.5:9b | `6488c96fa5faab64bb65cbd30d4289e20e6130ef535a93ef9a49f42eda893ea7` |
| qwen3.5:27b | `7653528ba5cba4dd8e19da24aaddc7f4d0b5ecd93571c0825dfd4137958ec06e` |

Embedding digest 为 `ac6da0dfba84a81fdbfbaf330198c33cd77c4cdfc53e8bc50eb581914a15621d`。模型自选的 search query、mode、limit 属于被测 Agent 行为；宿主提供的模式、默认值、检索配置与工具契约均相同。

原 integration 实验的临时 Qdrant 容器已移除。本次从原 SQLite 缓存恢复**同一个逻辑快照**到原 URL / collection，未重新 embedding 或重新切分；160 个 point 的 payload 和 vector 全量核验通过。原 SQLite、语料及历史 dataset 指纹一致。恢复记录见 [snapshot-restoration.json](results/agent_model_ablation/snapshot-support/snapshot-restoration.json)。

18 次真实 smoke run 单独保存在 [smoke 目录](results/agent_model_ablation/20260909-phase1-smoke/comparison.md)，不计入以下任何正式统计。历史 4B 的单次结果也未混入；本次 4B 的 120 条 v1 指标均重复了对应历史 case 的指标，见 [baseline-replication.json](results/agent_model_ablation/20260909-phase1-formal/baseline-replication.json)。

## 全量汇总

| 指标 | qwen3.5:4b | qwen3.5:9b | qwen3.5:27b |
| --- | --- | --- | --- |
| 成功 trial 数 / 120 | 93 | 102 | 114 |
| Task success | 77.50% | 85.00% | 95.00% |
| Mean source recall | 85.88% | 89.61% | 88.53% |
| Exploratory success | 40.00% | 80.00% | 100.00% |
| Avg tool calls | 3.0250 | 3.3750 | 3.0750 |
| Avg turns / Agent model requests | 3.5500 | 3.2000 | 2.9250 |
| Max-turn failure | 7.50% | 5.00% | 0.00% |
| Normal final | 90.00% | 92.50% | 100.00% |
| Runtime / tool error | 2.50% | 2.50% | 0.00% |
| Unnecessary retrieval | 0.00% | 0.00% | 0.00% |
| Avg match calls | 0.1500 | 0.1500 | 0.1500 |
| Avg search calls | 1.4750 | 1.6000 | 1.2000 |
| Avg read calls | 1.4000 | 1.6250 | 1.7250 |
| Avg evidence-sufficient turn（条件均值） | 1.2143 | 1.5806 | 1.2812 |
| Evidence-sufficient 有定义 trials | 84 | 93 | 96 |
| Avg wasted calls（条件均值） | 1.6296 | 2.0333 | 2.0625 |
| Wasted calls 有定义 trials | 81 | 90 | 96 |
| Expected sources found / tool call | 0.7455 | 0.7203 | 0.7328 |
| Avg latency / 秒 | 14.25 | 21.84 | 66.86 |

成功率以每模型 120 个 trial 为分母，recall 以 102 个需要来源的 trial 为分母；无需检索的 18 个 trial 的 recall 为 null。Unnecessary retrieval 以这 18 个无需检索 trial 为分母。所有失败均纳入正式成功率，六个异常都有部分 trace，没有缺失 trace 的 trial。

`evidence_sufficient_turn` 是首次达到原 case 的 source recall / read 标签要求的 observation 所在 turn，不代表答案正确。`wasted_tool_calls_after_sufficient_evidence` 只统计其后有完成 observation 的调用，含同一批次内的后续调用；它不证明这些调用可以省略。未达到证据阈值、或后续调用的执行情况不明时为 null。条件均值的分母随模型而变，不能单看均值判断效率。

Prompt / completion / total tokens 全部为 **null**，没有估算。延迟是 Runtime.ask 的完整墙钟时间，包含模型加载、检索、失败及缓存影响。固定模型运行顺序为 4B → 9B → 27B；每次 fresh conversation 不等于清空 provider cache。本结果不是纯推理吞吐基准。

## 按任务类型比较

| Task type | 模型 | 成功 | Recall | Tools | Turns | Max-turn | 延迟 / 秒 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| exact_lookup | qwen3.5:4b | 15/18 | 90.00% | 1.00 | 2.00 | 0.00% | 5.06 |
| exact_lookup | qwen3.5:9b | 15/18 | 90.00% | 1.00 | 2.00 | 0.00% | 6.22 |
| exact_lookup | qwen3.5:27b | 15/18 | 90.00% | 1.00 | 2.00 | 0.00% | 18.59 |
| semantic_discovery | qwen3.5:4b | 15/18 | 79.44% | 4.50 | 4.83 | 16.67% | 18.88 |
| semantic_discovery | qwen3.5:9b | 12/18 | 93.33% | 4.83 | 4.50 | 16.67% | 26.86 |
| semantic_discovery | qwen3.5:27b | 18/18 | 79.44% | 4.00 | 4.00 | 0.00% | 70.65 |
| direct_read | qwen3.5:4b | 18/18 | 100.00% | 1.00 | 2.00 | 0.00% | 8.33 |
| direct_read | qwen3.5:9b | 18/18 | 100.00% | 1.00 | 2.00 | 0.00% | 11.85 |
| direct_read | qwen3.5:27b | 18/18 | 100.00% | 1.00 | 2.00 | 0.00% | 32.27 |
| exploratory_retrieval | qwen3.5:4b | 12/30 | 75.33% | 6.60 | 6.10 | 20.00% | 28.17 |
| exploratory_retrieval | qwen3.5:9b | 24/30 | 79.67% | 7.60 | 4.80 | 10.00% | 45.97 |
| exploratory_retrieval | qwen3.5:27b | 30/30 | 84.33% | 7.20 | 4.20 | 0.00% | 156.79 |
| knowledge_qa | qwen3.5:4b | 15/18 | 91.67% | 2.67 | 3.67 | 0.00% | 14.54 |
| knowledge_qa | qwen3.5:9b | 15/18 | 91.67% | 3.00 | 3.83 | 0.00% | 21.67 |
| knowledge_qa | qwen3.5:27b | 15/18 | 91.67% | 2.50 | 3.50 | 0.00% | 56.48 |
| no_retrieval | qwen3.5:4b | 18/18 | null | 0.00 | 1.00 | 0.00% | 1.28 |
| no_retrieval | qwen3.5:9b | 18/18 | null | 0.00 | 1.00 | 0.00% | 2.38 |
| no_retrieval | qwen3.5:27b | 18/18 | null | 0.00 | 1.00 | 0.00% | 6.44 |

差异集中在 exploratory_retrieval 和 semantic_discovery。Exact lookup、direct read、knowledge QA、no-retrieval 的成功率在三个模型之间相同。27B 的语义发现 recall 与 4B 相同，却由 15/18 成功提高到 18/18；9B 的该组 recall 更高，但只有 12/18 成功。这直接说明 recall、停止与工具执行必须分开评价。

## Exploratory retrieval：30 trials / 模型

| 指标 | qwen3.5:4b | qwen3.5:9b | qwen3.5:27b |
| --- | --- | --- | --- |
| Success | 40.00% | 80.00% | 100.00% |
| Source recall | 75.33% | 79.67% | 84.33% |
| Normal final | 70.00% | 90.00% | 100.00% |
| Max-turn | 20.00% | 10.00% | 0.00% |
| Runtime / tool error | 10.00% | 0.00% | 0.00% |
| Avg tools | 6.6000 | 7.6000 | 7.2000 |
| Avg turns | 6.1000 | 4.8000 | 4.2000 |
| Avg search | 4.1000 | 4.1000 | 3.5000 |
| Avg read | 2.5000 | 3.5000 | 3.7000 |
| Avg wasted calls（条件均值） | 3.0000 | 4.0000 | 4.1000 |
| Wasted calls 有定义 trials | 15 | 27 | 30 |
| Avg latency / 秒 | 28.17 | 45.97 | 156.79 |

从 4B 到 27B，探索成功率提高 **60 个百分点**，recall 提高 **9 个百分点**，平均轮数从 6.1 降到 4.2。Search 减少而 read 增多，平均总调用从 6.6 增到 7.2；同一 turn 可以发起多个工具调用，因此轮数降低不能等价为调用更少。

在三个模型的 wasted 指标都有定义的**同一批 15 个 exploratory trial**（explore_003/004/008/009/010）上，均值仍是 **3 → 5 → 5**。因此本次并无“大模型普遍减少充分证据之后调用”的证据；`semantic_005` 等局部停止改善应与全局调用成本分开陈述。[补充计算](results/agent_model_ablation/20260909-phase1-formal/supplemental-analysis.json)保留配对 case 集合与分母。

## 三次 trial 的稳定性与 scaling 分类

| 模型 | Empirical pass@1 / 跨 case 平均成功概率 | Always pass | Flaky | Always fail |
| --- | --- | --- | --- | --- |
| qwen3.5:4b | 77.50% | 31 | 0 | 9 |
| qwen3.5:9b | 85.00% | 34 | 0 | 6 |
| qwen3.5:27b | 95.00% | 38 | 0 | 2 |

40 个 case 均有 3 个 trial，因此 case-weighted probability 与 pooled task success 恰好相同。所有模型的每个 case 在三次运行中的**完整 v1 指标和工具参数序列均相同**，没有观察到 flaky case；这不证明自然语言回答完全相同，也不证明换 seed、环境或数据后仍如此。三次重复仅支持本次配置下的经验概率，不做显著性或 pass@k 外推。

- **Scaling wins：7 个**：`semantic_005`, `explore_001`, `explore_002`, `explore_004`, `explore_005`, `explore_006`, `explore_007`。这些 case 中至少一个更大模型优于 4B。
- **Scaling insensitive：31 个**：29 个共同通过，`exact_001`、`qa_006` 共同失败。“Insensitive”只按成功率分类，不意味着 recall、工具成本或延迟相同。
- **Scaling regressions：2 个**：`semantic_002`、`semantic_004`，均为 **3/3 → 0/3 → 3/3**。退步发生在 4B → 9B；27B 恢复通过，没有观察到 9B → 27B 的成功率退步。

完整 40-case 成功概率、每类代表 case 的 query、expected sources、逐 trial 工具序列、轮数、stop reason 见 [comparison.md](results/agent_model_ablation/20260909-phase1-formal/comparison.md)。所有准确 search query、mode、limit、read selectors、missing sources 见 [comparison.json](results/agent_model_ablation/20260909-phase1-formal/comparison.json)。

## 九个既有失败 case

下表每格的成功次数均以 3 次 trial 为分母；recall / turns / stop 为三次一致的观测。

| Case | 成功：4B → 9B → 27B | Recall：4B → 9B → 27B | Turns / stop：4B → 9B → 27B |
| --- | --- | --- | --- |
| semantic_005 | 0/3 → 3/3 → 3/3 | 1.000 → 1.000 → 1.000 | 8 / max_turns → 4 / final → 3 / final |
| explore_001 | 0/3 → 3/3 → 3/3 | 0.333 → 0.667 → 0.667 | 8 / final → 3 / final → 4 / final |
| explore_002 | 0/3 → 0/3 → 3/3 | 0.200 → 0.400 → 0.800 | 8 / final → 7 / final → 6 / final |
| explore_004 | 0/3 → 3/3 → 3/3 | 1.000 → 0.600 → 1.000 | 8 / max_turns → 3 / final → 3 / final |
| explore_005 | 0/3 → 3/3 → 3/3 | 0.500 → 0.750 → 0.750 | 8 / max_turns → 6 / final → 5 / final |
| explore_006 | 0/3 → 3/3 → 3/3 | 1.000 → 1.000 → 0.667 | 8 / error → 4 / final → 4 / final |
| explore_007 | 0/3 → 0/3 → 3/3 | 0.500 → 0.750 → 0.750 | 3 / final → 8 / max_turns → 5 / final |
| qa_006 | 0/3 → 0/3 → 0/3 | 0.500 → 0.500 → 0.500 | 4 / final → 5 / final → 4 / final |
| exact_001 | 0/3 → 0/3 → 0/3 | 0.400 → 0.400 → 0.400 | 2 / final → 2 / final → 2 / final |

**semantic_005：更早停止的清晰实例。** 三个模型在第 1 turn 都已取得满足标签的来源，recall 都是 1。4B 继续到第 8 turn 耗尽预算，9B 在第 4 turn final，27B 在第 3 turn final。工具调用 8 → 5 → 4，充分证据后的调用 7 → 4 → 3。这里成功率改善并非检索从“找不到”变成“找到”。

**explore_004：search x8 的循环减少，但总调用未同幅减少。** 4B 执行 8 次 search、0 次 read，recall=1 仍 max_turns；9B 为 3 search + 3 read，27B 为 4 search + 4 read，均在第 3 turn final。该 case 不要求 read，最低 recall=0.6；4B 第 2 turn 已足够，9B/27B 第 1 turn 已足够。27B 将 8 次调用安排到更少轮数，其充分证据后的调用为 7，高于 4B 的 6。可确认的是搜索循环与停止改善，不能声称所有额外调用都减少。

**explore_005：覆盖与 max-turn 同时改善。** 4B recall=0.5，从未达到 0.75 的阈值，8 turn 后 max_turns；9B/27B recall 都为 0.75，分别 6 / 5 turn final。4B 阅读 02、05、38；较大模型阅读 02、27、38，补上重复试验相关来源 `27_agent_trial_reliability.md`，但仍未找齐全部 expected sources。工具调用分别为 8、5、6。

**explore_001：阅读选择改善，无法只归因于 query 字面措辞。** 4B 多次 hybrid search，最终读到 34 和非预期来源 38；9B 与 27B 读到预期的 06、34，满足至少两篇的要求，recall 从 1/3 升到 2/3。三者都以 `Agent Memory` 起步；4B 用 hybrid，9B 用默认 semantic，27B 首次 semantic limit=10。27B 另查 `memory lifecycle semantic episodic procedural agent`。差异同时涉及模式、候选范围和 read selection。

**explore_002：更有针对性的检索扩大覆盖。** 4B 搜索 `Agentic Retrieval`、`tool interface retrieval agent`，读取 9 篇但只覆盖 1/5 预期来源；9B 使用 `Agentic Retrieval tool interface candidate retrieval` 等查询，覆盖 2/5；27B 增加 `candidate retrieval search results`、`read original document continue reading`、`agent retrieval pipeline reading original document evidence`，覆盖 4/5。27B 读取 02、12、32、37、40，其中 02、40 是 expected sources，缺失的 expected source 为 13。应区分 search 找回的覆盖率与实际阅读；较大模型并未把每次 read 都选为 expected source。

**explore_007：9B 与 27B recall 相同，停止及阅读对象不同。** 4B recall=0.5，不足所需 0.75；9B 达到 0.75 后仍 8 turn max_turns，读取 31、34、非预期的 38。27B 也是 0.75，读取 07、31、34，5 turn final；其后续 query 包含 `Agent handoff shared memory coordination cost` 和 `shared memory state context window agent autonomy independence`。这支持对交接 / 记忆来源的选择与控制改善，但不足以将每个差异归因到某一个内部推理机制。

**qa_006：更大模型没有补齐缺失来源。** 三者全为 recall=0.5 / final，均未取回 `32_document_ingestion_and_structure.md`，只覆盖另一 expected source `37_multimodal_evidence_retrieval.md`。4B/27B 各 1 search + 2 read，9B 为 2 search + 2 read；首查均为 `表格页面检索 错误证据 生成器`，4B 显式 hybrid，其余默认 semantic。该来源在其他 case 中能被检索到，不能据此断言文档或索引缺失。后续应区分 query、检索排序与标签覆盖要求，单靠本实验不能断定是 embedding 的原因。

**exact_001：共同的 match 完整枚举问题。** Query 要求列出正文中大小写一致包含 `RAG` 的所有笔记，expected 为 30、32、33、37、40。三个模型都仅一次 `match(query="RAG", target="content", case_sensitive=True)`，使用默认 limit=5；结果按 occurrence 返回，其中 4 条来自同一篇 32，加上 30，共覆盖 2/5。三者都在第 2 turn final。优先检查 match 的枚举契约、limit 与完整性反馈比继续扩大模型更有依据；本阶段没有修改工具。

**explore_006：单独归为 runtime / tool failure。** 4B 三次均为 `LookupError: Unknown section`，失败 read 的 section_id 为 `fb60551581e3b974054177bddee096ffc965c5b2a5aceda201c40570e0db3878`，部分 trace 已保留。4B 已检索到所有 expected sources，recall=1；9B、27B 没有该错误并 final，但 recall 分别为 1 和 2/3。这个 case 的成功变化不能作为“找回更多证据”或“基础设施改善证明推理增强”的证据。

上述数字编号对应完整文件名见 comparison 的 expected_sources / read_sources；每个 case 的原始 query、完整文件路径集合和逐次调用都保存在同一报告及原始 trace 中。

## 退步与不敏感代表

`semantic_002` 要找“换一种说法后仍能找到”的材料，expected 为 `13_hybrid_rank_fusion.md` 与 `29_learned_sparse_retrieval.md`。4B/27B recall=0.5、4 次工具调用并 final（分别 5 / 4 turn），满足该 case 的原成功阈值。9B recall=1，却 7 search + 1 read、8 turn max_turns，故 0/3。更高召回不自动意味着更高任务成功率。

`semantic_004` 要找“单次完成不证明每次可靠”的材料，expected 为 07、27、38。4B/27B 均 1 search + 3 read、5 turn final、recall=2/3；9B recall=1，但第 2 turn 出现 `ValueError: document_id must be a lowercase SHA-256 hex digest.`，3 次均失败。原错误及不完整调用结果均保留。这是工具参数 / 执行失败，不能隐藏在平均分中。

成功不敏感的例子是 `exact_002`：查询精确文本 `Reciprocal Rank Fusion`，expected 为 `13_hybrid_rank_fusion.md`；三者均 3/3、recall=1、一次 match、2 turn / final。`exact_003` 查询 `LangMem`，expected 为 `34_agent_memory_lifecycle.md`，轨迹指标也完全相同。与共同失败的 exact_001 / qa_006 一起，这些 case 显示模型容量并非所有任务的限制因素。

`explore_010` 也均通过且 recall=1，但工具调用为 **3 → 13 → 10**，充分证据后的调用为 **2 → 11 → 8**。这是成功率不敏感而成本明显不同的反例。

## 错误、控制能力与效率的归因

| 模型 | 停止分布 final / max_turns / error | match / search / read 总数 | 异常 |
| --- | --- | --- | --- |
| qwen3.5:4b | 108 / 9 / 3 | 18 / 177 / 168 | LookupError × 3 |
| qwen3.5:9b | 111 / 6 / 3 | 18 / 192 / 195 | ValueError × 3 |
| qwen3.5:27b | 120 / 0 / 0 | 18 / 144 / 207 | 0 |

没有记录到 timeout、OOM 或 Ollama 服务异常；没有自动重试。两个有异常的 case 是 explore_006 和 semantic_004。作为**补充敏感性检查**，在三个模型中同时排除这两个 case（每模型剩余 114 trials），成功率仍为 **90/114=78.95% → 99/114=86.84% → 108/114=94.74%**。主结果没有排除它们；此检查只说明总体收益不全来自避免这两个运行错误。

27B 总 recall=88.53%，高于 4B 的 85.88%，却低于 9B 的 89.61%；总体成功率则为 77.5% → 85% → 95%。这与 semantic_005、semantic_002、explore_007 的逐轨迹证据共同支持：**已有检索结果如何被读取、何时结束及工具参数是否有效，是重要差异来源。** 探索组 recall 的上升则说明查询 / 候选 / 阅读选择也有贡献，不能归结为只有 stopping。

工具效率不单调：27B 平均 search 少于另外两者，但 read 更多；总调用 3.075 略高于 4B 的 3.025，低于 9B 的 3.375。每次调用找到的 expected sources 均值与充分证据后调用也未整体改善。更大的模型主要提高任务完成与轮数效率，不能声称降低了总体调用或延迟成本。

## 固定 baseline 与下一阶段

**后续实验固定 qwen3.5:27b。** 选择依据是 38/40 case 始终通过、探索组 10/10 case 始终通过、0 max-turn、0 运行错误。相比 9B 增加 4 个始终通过的 case，整体成功率高 10 个百分点，探索成功率高 20 个百分点；本次不符合“9B 与 27B 质量接近”的条件。

9B 的平均 latency 为 21.84 秒，27B 为 66.86 秒（约 3.06 倍）；4B 为 14.25 秒。27B 的选择适合需要减少 Agent 干扰的后续科学比较，不等于已证明它适合所有产品延迟目标。产品默认模型仍保持原值 4B。运行时 /api/ps 观测的 model size_vram 约为 12.92 / 15.07 / 32.50 GiB；这些是加载状态的模型占用字段，**不是峰值系统内存测量**，原始观测保存在 snapshot-support/runtime-observations.jsonl。

**可以进入 Embedding Ablation。** Phase 1 已提供完整、可复算的基线及固定 Agent 的依据。下一阶段应另建实验，固定 27B 的准确 tag/digest、prompt/tools/loop、原 dataset 和语料、采样与检索策略，仅将 embedding 及其对应向量索引作为变化因素，并继续分别报告 source recall、任务成功、错误、调用与延迟。

目前 27B 的探索成功率已到 100%，且 v1 标签允许部分 source recall 达标，因此只比较成功率可能看不出 embedding 差异；需要同时保留连续的来源覆盖指标、逐 case 缺失来源及检索行为。qa_006 可作为共同检索缺口观察点；exact_001 是 match 枚举问题的候选，不能期待仅换 embedding 就解决。二者的根因尚需各自验证。本阶段不修改标签或修复 Agent。

## 实现、复算与产物

新增 `src/arkb/evaluation/model_ablation.py`，薄层调用已有 `run_agent_evaluation`；新增 `ablation_analysis.py` 复用 v1 evaluator / aggregation。原 runner 仅增加命名空间 metadata 与 trial analysis 输出接口，没有复制或修改 Runtime.ask、Agent loop、工具执行、retrieval 或成功规则。

新增的 12 个确定性测试覆盖模型配置、传参、360 次展开、输出隔离、失败继续、序列化、汇总、case 分类、report、缺失模型、控制变量漂移与中断恢复；测试不调用真实 LLM。最终默认确定性 suite：**908 passed，60 integration tests deselected**。

```
.venv/bin/python -m arkb.evaluation.model_ablation \
  --baseline-metadata evaluation/results/integration-20260909T190858Z-089e02/evaluation/run_metadata.json \
  --output evaluation/results/agent_model_ablation/<new-experiment-id> \
  --num-trials 3
```

复跑应使用新 output 目录，并保留本报告的模型、服务和快照条件。实验 runner 的使用及字段定义见 [agent-model-ablation.md](agent-model-ablation.md)。

- [统一 comparison.md](results/agent_model_ablation/20260909-phase1-formal/comparison.md)：整体 / task type / 40-case / 代表轨迹比较。
- [comparison.json](results/agent_model_ablation/20260909-phase1-formal/comparison.json)：完整聚合、每个 case 的 query、sources、tool arguments 与行为指标。
- [qwen3.5:4b 原始 results.jsonl](results/agent_model_ablation/20260909-phase1-formal/qwen3.5-4b/results.jsonl)：120 条原始 trace、v1 metrics、错误和 annotations。
- [qwen3.5:9b 原始 results.jsonl](results/agent_model_ablation/20260909-phase1-formal/qwen3.5-9b/results.jsonl)：120 条原始 trace、v1 metrics、错误和 annotations。
- [qwen3.5:27b 原始 results.jsonl](results/agent_model_ablation/20260909-phase1-formal/qwen3.5-27b/results.jsonl)：120 条原始 trace、v1 metrics、错误和 annotations。
- [config.json](results/agent_model_ablation/20260909-phase1-formal/config.json)、[run_metadata.json](results/agent_model_ablation/20260909-phase1-formal/run_metadata.json)：实际配置、时间、source hashes、模型 / 索引指纹。
- [validation.json](results/agent_model_ablation/20260909-phase1-formal/validation.json)：360 个唯一模型/case/trial、逐条评分与分析复算、全量比较复算、控制变量及 Qdrant vector/payload 核验。
- [measured-source.zip](results/agent_model_ablation/20260909-phase1-formal/measured-source.zip)：实验完成时的准确 Python 源码、source hashes、依赖文件与新增测试。

正式实验结束、核验并归档源码后，仅修正了 Markdown formatter 的一处展示问题：分 task 表格不再显示不存在的 exploratory-success 字段。没有重跑模型，没有更改 trial、评分或分析；原始 Markdown 另存 comparison.measured.md，展示改动及前后 hashes 见 [postprocessing.json](results/agent_model_ablation/20260909-phase1-formal/postprocessing.json)。全部原始 results.jsonl hashes 与核验时一致。

这些结论仅对应本次 40-case、固定语料、Q4_K_M 模型和近确定性配置。v1 成功衡量来源 / 工具 / 预算 / 停止标签，不是自由文本答案质量评估，也不是对更广泛 Agent 任务的统计保证。
