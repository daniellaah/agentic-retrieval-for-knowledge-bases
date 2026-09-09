# Capability 目录重构结果

已按[目录重构方案](directory-refactor-plan.md)完成五个阶段。分支：`refactor/capability-layout`；迁移前基线：`9d47667`。

## 交付结构

生产包按 Knowledge、Retrieval、Generation、Agent、Interfaces、Evaluation 六个能力组织，根部只保留 `__init__.py`、`runtime.py` 和 `config.py`。

- Knowledge 收拢文档加载、唯一 chunker、embedding/tokenizer、索引构建、SQLite 与 Qdrant。`indexing.py` 负责构建和发布，`qdrant.py` 负责后端读写；Knowledge 不依赖 Retrieval。
- Retrieval 保留独立 BM25、Semantic、Hybrid、RRF、Reranker 和 Engine。纯结果契约在 `models.py`，Qdrant 快照读取适配器在 `semantic.py`，cross-encoder 与 reranker 合并到 `rerank.py`。
- Generation 收拢 Context、预算、证据与引用契约、引用验证、生成和 generation tokenizer。独立的 `models.py` 不加载生成客户端。
- Runtime 管理自己创建的客户端和 tokenizer，按需创建、复用并在退出时关闭；接收的外部 storage/client 仍由调用者管理。CLI 和检索评估共用组装代码，评估继续直接调用独立算法。原有一次性 `search_index` 与可复用 `SnapshotSemanticRetriever` 归入 Runtime。
- Evaluation 拆为 `datasets.py`、`metrics.py`、`retrieval.py`、`generation.py`。保留相关性、exact/ANN、冻结候选重排、Context 和引用评估，以及组合报告与不可覆盖的实验产物。
- Agent、MCP 和 exact 文本检索仅创建职责说明占位。Qdrant 原有 `--exact` 向量搜索不受影响。

旧文件和转发路径已删除，没有保留两套导入方式。README、benchmark 脚本、评估命令和安装入口均使用新路径。

## 主要导入迁移

| 原位置 | 当前位置 |
| --- | --- |
| `arkb.schema` | `arkb.knowledge.models`；Qdrant point ID / identity 归 `knowledge.qdrant` |
| `arkb.indexing.loaders` | `arkb.knowledge.documents`；Note 归 `knowledge.models` |
| `arkb.chunking` / `arkb.indexing.chunking` | `arkb.knowledge.chunking` |
| `arkb.embeddings` / `arkb.tokenization` / `retrieval.ollama` | `arkb.knowledge.embeddings` |
| `arkb.indexing` / `arkb.storage` | `knowledge.indexing` / `knowledge.sqlite`；Qdrant 后端归 `knowledge.qdrant` |
| `retrieval.contracts` / `retrieval.snapshot` | `retrieval.models`；`snapshot_result` 归 `retrieval.semantic` |
| `retrieval.qdrant` | 原始向量读归 `knowledge.qdrant`；快照适配归 `retrieval.semantic`；应用组装归 `runtime` |
| `retrieval.reranker` / `retrieval.cross_encoder` | `arkb.retrieval.rerank` |
| `arkb.context` / `arkb.generation` | `generation.models` / `generation.context` / `generation.citations` / `generation.generate` |
| `arkb.cli` | `arkb.interfaces.cli` |
| `arkb.retrieval_evaluation` / 原 `arkb.evaluation` 文件 | `evaluation.datasets` / `evaluation.metrics` / `evaluation.retrieval` / `evaluation.generation` |

产品命令仍是 `arkb index/query/status`，参数和输出保持。更新代码后运行 `uv sync --locked`；使用可选重排依赖时加 `--extra rerank`。

评估入口现在是：

```sh
python -m arkb.evaluation.retrieval baseline --help
python -m arkb.evaluation.retrieval ann --help
python -m arkb.evaluation.generation --help
```

`baseline` 保留相关性 runner 的参数与默认值；`ann` 保留原 exact/ANN runner，包括组合使用 `--context`、`--citations`。Generation 入口默认启用 Context 评估，`--citations` 才启用回答生成；它复用同一快照实验流程，保留 exact 命中和原有报告格式。

## 测试与验证

Tests 一级目录大致镜像六个能力。真实服务检查迁入各能力的 `integration/`，统一标记 `integration`，原有环境开关不变。底层向量读测试归 Knowledge，快照检索测试归 Retrieval，资源关闭测试归 Runtime。数据集和指纹检查由评估 runner 的既有行为测试覆盖，没有新增空测试文件。

| 验证 | 结果 |
| --- | --- |
| 常规测试：`pytest -q -m "not integration"` | 561 passed，38 deselected |
| 全量收集，显式关闭真实服务开关 | 561 passed，38 skipped |
| 迁移前后参数化用例逐项核对 | 原功能用例全部保留；删除 1 项仅验证旧 chunking 别名的测试，新增 1 项 Runtime 所有权测试 |
| 旧索引兼容对照 | 旧代码创建临时 SQLite + Qdrant Local 索引，新代码直接复用，无新 embedding 请求 |
| 同一临时索引上的行为对照 | Manifest、chunk 身份、向量、四种检索输出、Context 和带引用输出与旧代码完全一致 |
| 安装包验证 | 从源码分发包构建 wheel，安装到独立临时目录，核对全部模块和 CLI entry point |
| 安装后入口验证 | 在仓库外启动产品 CLI、模块 CLI、baseline、ann、generation 入口，均成功 |
| 导入边界 | Knowledge 纯模型/chunker、Retrieval、Generation 模型、Runtime/Config 导入不加载 Ollama、Qdrant、tokenizer SDK 或重排模型 |

结构搬迁使用既有回归测试保护行为。对新增 Runtime 公共资源边界采用 TDD：先添加“按需创建、复用、异常关闭”的失败测试，再实现并验证。没有为每次移动文件增加重复测试。

本次验证使用合成数据、模拟模型和 Qdrant Local；38 项真实 Ollama、Qdrant Server、tokenizer 或 cross-encoder 检查未启用，因此不据此判断真实 ANN 性能或模型质量。

## 后续工作

此次聚焦职责与目录，不改变索引格式、身份命名空间、缓存键、算法和引用规则；无需因目录迁移重建现有兼容索引。

[架构审查](architecture-review.md)中的 Context 来源可变性、原文坐标与文档访问能力、tokenizer 指纹开销等仍需独立处理。资源组装重复已在本次迁移中收拢。后续行为修复适合先补最小失败用例，再修改实现；新增 Agent/MCP/exact 能力按各自公开行为建立测试。
