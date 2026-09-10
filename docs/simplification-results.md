# 架构简化执行记录

分支 `refactor`，基线 `7bbf4ce`。按“单项变更、常规测试、真实集成验证、提交”的顺序执行。六个能力边界、索引格式、默认模型和 Agent 的 8 轮预算保持不变。

## 已完成步骤

| 项目 | 变更 | 提交 | 验证 |
| --- | --- | --- | --- |
| 1 | 删除无调用别名和重复 bookkeeping | `75fbeb9` | 964 常规 + 2 真实 Qdrant |
| 2 | 修复旧引用评估子进程入口 | `76299a9` | 964 常规 + 10 真实生成 |
| 3 | RRF 实现并入 hybrid，保留独立函数 | `19b77d5` | 964 常规 + 2 CLI/Qdrant；冻结输出一致 |
| 4 | 删除一次性 search_index 转发接口 | `6ff442b` | 964 常规 + 2 真实 Qdrant |
| 基线修复 | 搜索反复返回旧证据时提醒 Agent 判断是否继续 | `7bd3d63` | 972 常规 + 59 真实集成；详见[基线验证](agent-search-stopping-validation.md) |
| 5 | 纯函数生成工具 schema，评估不再构造假 capability | `cb8080b` | 972 常规 + 5 真实 Agent；冻结输出一致 |
| 6 | evaluation.runs 管执行，agent_metrics 管指标与报告 | `32dc90c` | 972 常规 + 真实 Agent/BM25 评估 CLI |
| 7 | context/citation 组合评估共用一次上下文构建 | `d888af0` | 972 常规 + 10 真实生成 |
| 8 | SQLite 记录解码和 Qdrant payload 各保留一处定义 | `8bb9d5a` | 972 常规 + 2 真实 Qdrant |
| 9 | 有序快照记录一次事务写入；embedding 批次继续独立落盘 | `98e79fc` | 973 常规 + 3 真实 Qdrant/CLI；新增记录批次回滚测试 |
| 10 | 删除 EvidenceBlock，最终上下文直接保存 CitationSource/CitationOrigin | `ff6585c` | 974 常规 + 10 真实生成；冻结输出一致；新增检索元数据修改不影响已构建上下文的回归测试 |
| 11 | 删除内部重复结果检查；引用映射在构造时检查一次 | `aba4756` | 968 常规 + 12 真实集成；冻结输出一致 |
| 12 | 复核诊断历史，保留当前检索导出格式；生成边界的分离由第 10 项完成 | 本记录 | 真实 Qwen reranker 排序、批处理和冻结候选评估通过 |

第 11 项删除了 5 个通过假 VectorIndex 违反内部契约触发上层检查的参数化测试，以及 1 个通过替换 SQLiteStorage.load_snapshot 伪造缺失记录的测试。真实存储损坏、快照计数、发布回滚、过滤、embedding 空间兼容、模型输出校验仍有覆盖。坏分数仍会失败，只是错误现在来自唯一的 snapshot_result 检查。

## 第 12 项的范围决定

没有引入第二套 RetrievalTrace 模型或通用 metadata 序列化框架。

- `interfaces/cli.py` 用 `asdict` 导出 SearchResponse；`evaluation/retrieval.py` 的 `evaluate_retrievers` 和 `evaluate_reranker` 保存完整检索与候选结果。
- `retrieval/hybrid.py:rrf` 的贡献列表和 `retrieval/rerank.py:Reranker.rerank` 的输入排名、分数、scorer 身份进入这些已有产物；真实 reranker 集成测试核对它们对应实际送入模型的候选。
- 常规生产组合只执行一次融合和一次重排。递归历史不参与排序或 Agent 决策，但删除它会改变现有 JSON；在本轮“保持行为”约束下，再增加 trace 容器和兼容投影会增加概念。
- 第 10 项已切断生成层与完整 SearchResult 的耦合。上下文仅保留引用需要的身份、位置和最终分数，修改检索 metadata 不会改变提示词、context_id 或引用来源。
- Agent 工具继续只投影实际观察字段，不把融合和重排诊断发送给模型。

后续若确有检索结果体积/性能需求，适合通过明确版本化的诊断输出协议删除递归字段；不在本次重构中增加预备接口。

## 保留的边界

保留独立 BM25、Semantic、Hybrid/RRF、Reranker API，供评估与消融调用；RetrievalEngine 只组合确定性策略。Qwen 模型加载继续与通用排序逻辑分开。SQLite 写锁、embedding 校验、候选快照、Qdrant 配置/内容核验与原子发布继续保留。Generation 的真实 token 计数、引用解析、原文重叠检查和模型外部响应校验继续保留。
