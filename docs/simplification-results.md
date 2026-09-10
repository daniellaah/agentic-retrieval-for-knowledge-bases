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
| 12 | 复核诊断历史，保留当前检索导出格式；生成边界的分离由第 10 项完成 | `aef9d34` | 真实 Qwen reranker 排序、批处理和冻结候选评估通过 |
| 13 | live read 直接返回 DocumentSlice，共用文档身份与修订计算 | `d8c7df5` | 968 常规 + 6 真实 Agent/CLI；冻结工具输出一致 |
| 14 | 文档加载、扫描指纹和评估共用 flat Markdown 文件枚举 | `9dac603` | 969 常规 + 真实索引生命周期 + Agent 评估 CLI；新增评估指纹范围回归测试 |

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

`DocumentAccess.records` 继续提供 whole-note 记录给独立检索及评估。`read` 返回实际文档片段，没有 chunk_id；Agent 的已有 JSON 字段仍明确输出 `chunk_id: null`。文档加载与索引原来允许文件符号链接，live 工具和评估指纹排除根目录外链接，本轮保持各自范围。DocumentAccess 的选择器枚举保留原有过滤、路径解析顺序；没有用统一扫描修改其错误或越界行为。

`_LazySnapshotRetriever` 保留：它让问候和 live match/read 无需初始化向量、embedding 服务，并在首次搜索后保留快照。没有为省去一次轻量 engine 构造增加另一套 capability factory。包入口继续公开实际使用的确定性检索、Agent 和数据契约；Embedder、VectorIndex 和 CandidateScorer 隔离外部 SDK 和模型，测试可通过同一边界提供确定性实现。

## 当前目录与概念

```text
src/arkb/
├── config.py / runtime.py
├── knowledge/    documents, chunking, embeddings, indexing, sqlite, qdrant, models
├── retrieval/    exact, bm25, semantic, hybrid (含 rrf), rerank, qwen_rerank, engine, models
├── generation/   context, citations, generate, models
├── agent/        loop, state, tools
├── interfaces/   cli
└── evaluation/   datasets, metrics, agent_metrics, baselines, retrieval, generation,
                  runs, model_ablation, ablation_analysis, models
```

删除没有实现、注册或调用者的 MCP 占位文件；README 明确 MCP 尚未实现。历史实验产物和历史目录迁移记录保留原始命令，当前文档和入口使用新路径。

Knowledge 准备、持久化并读取知识。Retrieval 执行可独立测试的确定性检索。Generation 将检索结果转换为不可变引用证据，打包预算并验证回答。Agent 只决定动作与停止。Interfaces 适配用户请求。Evaluation 直接使用同一能力和组合根。Runtime 管资源、延迟初始化与固定快照，不承担检索算法。

## 最终验收（2026-09-10）

| 验证 | 结果 |
| --- | --- |
| 常规测试 `pytest -q -m 'not integration'` | **969 passed**；63 个真实集成用例在独立命令中执行 |
| 全部真实集成 `pytest -q -m integration`，启用模型、Qdrant 和 reranker 开关 | **63 passed，0 skipped**，192.65 秒 |
| 原始基线 `7bbf4ce` 与最终代码的固定检索/context/tool JSON 比较 | 完全一致，包含 BM25/semantic/hybrid、重排、过滤、重复与重叠证据、两种引用模式和工具 schema/read/search/match |
| 源码分发包及 wheel | `uv build --offline` 成功；wheel 安装到独立临时目录 |
| 仓库外从 wheel 加载 CLI 与评估入口 | 产品 CLI、runs、retrieval baseline/ann、generation、model_ablation 共 6 个 `--help` 入口通过；评估指南导入通过 |
| 旧模块与语法检查 | 42 个 Python 源文件解析通过；当前 source/tests/入口文档没有退休模块引用；`git diff --check` 通过 |

全部真实集成包含 4 个原本可选的 Qwen reranker 测试，使用独立的 160-chunk example_notes 快照，并核对 BM25、semantic、hybrid 的候选、过滤和真实重排分数。真实 Agent 探索场景正常结束；明确限制轮次的测试仍按预期返回 max_turns。模型行为只按这些实际用例的断言验证，不据此声称所有任意问题都能稳定结束。

验证使用已缓存本地模型和本任务独立创建的 Qdrant 容器，没有修改用户索引或下载模型。JUnit XML、固定输出对照和最终 Agent 轨迹另存本地 `.arkb/refactor/final-validation/`（忽略提交的测试产物）；本文件保留可随代码审阅的结论。原始验证目录为 `/private/tmp/arkb-refactor-validation/`。
