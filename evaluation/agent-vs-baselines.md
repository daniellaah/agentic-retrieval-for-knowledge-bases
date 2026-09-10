# Agent 与固定 Retrieval Baselines 对比

本次回放已有结果，未重新调用模型。已逐一验证 dataset、标准化后的 case query/labels、
知识库 before/after 指纹和 snapshot 一致，并从全部 360 条 AgentTrace 重算 v1
source recall 与 task success。

## 同一批 34 个检索任务

排除 6 个 no_retrieval case。Agent 每个 case 先平均 3 次 trial，再对 34 个 case
等权平均；baseline 每个 case 执行 1 次。全部 source recall 均有定义。

| Method | Source recall | Mean request/task time |
| --- | ---: | ---: |
| BM25 | 30.39% | 0.0002 s |
| Semantic | 82.79% | 0.0992 s |
| Hybrid | 84.61% | 0.0773 s |
| Hybrid + Rerank | 87.30% | 2.8156 s |
| Agent qwen3.5:4b | 85.88% | 16.5436 s |
| Agent qwen3.5:9b | 89.61% | 25.2762 s |
| Agent qwen3.5:27b | 88.53% | 77.5262 s |

Baseline recall 来自一次返回的 top-10 chunks，按 source 去重；Agent recall 来自完整
trajectory 的 match/search/read 累计来源。两者是来源覆盖对照，检索预算不同。
Agent 时间包含推理、生成与工具；baseline 时间只包含一次 engine request，初始化另计。
不同运行的缓存/加载影响没有消除，不能把时间比值当作受控性能测试。

Agent 9B 相对 Hybrid + Rerank 的总体来源召回高 **2.30 个百分点**；平均任务时间为
**25.28 秒 vs 2.82 秒**。这批结果不支持 Agent 在所有任务上都优于固定检索的结论。
历史 Agent 实验的 reranking 均未启用；Hybrid + Rerank 使用固定 Qwen3 reranker。

## 按任务看 Agent 9B 的来源覆盖

| Task type | Hybrid + Rerank | Agent 9B |
| --- | ---: | ---: |
| direct_read | 100.00% | 100.00% |
| exact_lookup | 100.00% | 90.00% |
| exploratory_retrieval | 68.83% | 79.67% |
| knowledge_qa | 100.00% | 91.67% |
| semantic_discovery | 80.00% | 93.33% |

Agent 9B 的覆盖收益主要出现在 semantic_discovery 和 exploratory_retrieval。
这些是来源覆盖结果，不证明最终回答质量。

## Agent task success 单独保留

| Agent model | 原 v1 success：全部 40 cases × 3 | 检索子集：34 cases × 3 |
| --- | ---: | ---: |
| qwen3.5:4b | 77.50% | 73.53% |
| qwen3.5:9b | 85.00% | 82.35% |
| qwen3.5:27b | 95.00% | 94.12% |

Task success 还检查正常停止、工具限制、read 覆盖等。部分工具出错前已取得的证据
仍计入 source recall，但任务可以失败。因此不能把 baseline Recall 当作其 task success，
也不能仅凭 source recall 判断 Agent 9B 与 27B 的任务完成能力。
MRR/nDCG 仍只用于 baseline，没有为 Agent 构造单次排名。

## rg 在哪里

当前 ripgrep 路径为 `AgentTools.match → ExactRetriever.search → _matches → rg`：

- `src/arkb/retrieval/exact.py`：调用本机 `rg`，支持 literal/regex、大小写选项及 content/source 目标。
- `src/arkb/agent/tools.py`：Agent 通过 `match` 工具调用；它与 `search`、`read` 并列。
- `src/arkb/runtime.py` / CLI：直接入口为 `Runtime.match` 和 `arkb match "RAG" --json`。

它读取实时笔记，不依赖 embedding 或向量索引，按文件及命中位置排序，没有相关性分数。
它未进入 RetrievalEngine 的 bm25/semantic/hybrid mode，也未作为本次四种 baseline 的第五组。
CLI 的 `search --exact` 表示 Qdrant exact vector search，不是 rg。

历史三个 Agent 模型每组 120 次 trial 都实际调用了 **18 次 match**，即 6 个
exact_lookup case × 3。例如 `exact_002` 使用 `query="Reciprocal Rank Fusion"`，
成功找到对应笔记。

还确认了 `exact_001` 的失败机制：目标是找全正文包含 RAG 的 5 篇笔记，Agent 调用
`match(query="RAG")` 时未指定 limit，默认只取前 5 个 occurrence；其中 4 次来自同一篇
笔记，最终覆盖 2/5 篇，source recall=40%，随后未继续补齐。这是已有的 occurrence limit
行为，不是 rg 没有参与 Agent。评估没有暗中扩大 limit 或从 expected_sources 反推 pattern。

直接把原始自然语言问题交给 literal rg，匹配的是整句；Agent 在 match 中选择关键词的
行为不同。本报告说明现有能力与覆盖范围，没有新增关键词提取策略或修改 match 算法。

## 产物

- [结构化对照、逐 case 指标和 match 调用记录](results/agent-vs-baselines-20260909/comparison.json)
- [固定 baseline 实测](deterministic-baseline-results.md)
- [Agent Model Ablation 原始结论](phase1-agent-model-ablation-results.md)
