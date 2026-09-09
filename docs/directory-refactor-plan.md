# 六个能力边界下的目录重构方案

本方案采用用户确定的 Knowledge、Retrieval、Generation、Agent、Interfaces、Evaluation 六个一级能力边界，根部保留 runtime.py 和 config.py。它取代上一版按 domain/documents/indexing/embeddings/storage/application 分为多个一级包的方案。

当前生产代码基线为 9d47667。本轮只重新制定方案，尚未移动生产文件、创建功能占位或修改测试。实施时按下述结构完成迁移与占位。

## 1. 架构边界

| 能力 | 职责 | 边界 |
| --- | --- | --- |
| Knowledge | 文档加载、切块、embedding、索引构建与生命周期、持久化、原始数据访问 | 提供 Note、Chunk、记录、向量和快照；不依赖 Retrieval、Generation 或 Agent |
| Retrieval | exact、BM25、semantic、hybrid、fusion、rerank；返回有来源的检索结果 | 消费 Knowledge；不构建索引、不生成回答、不运行 Agent 决策 |
| Generation | 从证据构建 Context、管理预算与引用、调用生成模型、验证和输出带引用结果 | 独立能力；无须先运行 Agent，也不负责再次检索 |
| Agent | 按任务选择搜索、阅读、生成操作，累计状态、判断是否继续或停止 | 调用已有能力；不实现算法、存储或引用校验 |
| Interfaces | CLI、MCP 的参数/协议适配、输出与错误表达 | 使用 runtime 准备能力；不复制算法或 Agent loop |
| Evaluation | 数据集、指标、检索及 Agent 实验；保留现有 Context/引用评测 | 独立调用能力 API，生产能力不反向依赖评测 |
| runtime.py | 组装能力、准备和复用资源、固定快照、管理关闭、承接现有应用查询组合 | 不放算法、prompt、数据集指标或 Agent 决策；不成为全功能业务类 |
| config.py | 应用配置、默认值与显式参数合并/校验 | 纯配置，不创建客户端、不下载模型、不在 import 时读取/修改外部状态 |

Knowledge 包含“准备知识”和“访问知识”两部分；整个包不应被理解为只能离线使用。read_note、list_notes 未来直接使用 knowledge.documents，exact 使用同一文档访问边界。

依赖方向如下：

~~~text
interfaces ──> runtime ──> knowledge / retrieval / generation / agent
evaluation ──> runtime 的资源准备 + 各能力的独立 API

agent ──> knowledge 的文档访问 + retrieval + generation
generation ──> retrieval.models + 必要的 knowledge.models
retrieval ──> knowledge
knowledge ──> 自身实现与外部服务
~~~

能力对象由 runtime 创建并注入。Agent loop 和 tools 不反向导入 runtime 来临时获取全局服务。CLI 可直接查询或生成，不要求通过 Agent。

## 2. 目标生产目录

保留用户给出的结构。仅额外列出 evaluation/generation.py，用来承接项目已经实现的 Context、引用与生成评测；它不是新增功能。其余尚未开发的文件明确占位，各包包含轻量 __init__.py。

~~~text
src/arkb/
├── __init__.py
│
├── knowledge/
│   ├── models.py           # Note、Chunk、EmbeddingSpec、记录、manifest、身份规则
│   ├── documents.py        # load_notes、scan_notes
│   ├── chunking.py         # 唯一 chunking 实现
│   ├── embeddings.py       # 输入格式、tokenizer、批处理/重试、query embedder
│   ├── indexing.py         # 构建、增量、核验与发布编排
│   ├── sqlite.py           # SQLiteStorage、缓存、事务、锁、快照
│   └── qdrant.py           # collection/point 操作、原始向量查询、后端检查
│
├── retrieval/
│   ├── models.py           # 搜索契约、结果构造、请求校验
│   ├── exact.py            # 占位：源文件精确检索
│   ├── bm25.py
│   ├── semantic.py         # semantic 流程与后端命中适配
│   ├── hybrid.py
│   ├── fusion.py
│   ├── rerank.py           # reranker、组合器、可选 cross-encoder scorer
│   └── engine.py
│
├── generation/
│   ├── models.py           # 共用的 Context/引用数据与计数契约
│   ├── context.py          # 证据处理、预算、BuiltContext、消息构建
│   ├── citations.py        # 引用协议、解析、校验、渲染
│   └── generate.py         # 模型调用、CitedGeneration、生成模型计数适配
│
├── agent/
│   ├── tools.py            # 占位
│   ├── state.py            # 占位
│   └── loop.py             # 占位
│
├── interfaces/
│   ├── cli.py
│   └── mcp.py              # 占位
│
├── evaluation/
│   ├── datasets.py         # 读取、验证现有样本，数据集身份
│   ├── metrics.py          # 排名、邻居、覆盖与引用统计
│   ├── retrieval.py        # 检索/ANN/冻结候选重排实验及其入口
│   └── generation.py       # 已有 Context/引用/生成实验及其入口
│
├── runtime.py
└── config.py
~~~

不额外建立 domain、documents、indexing、embeddings、storage、context、application 等一级包，也不保留旧文件作为长期兼容转发。chunking 的唯一位置是 knowledge/chunking.py。

Evaluation 的 Agent runner 在 Agent 行为真正实现时加入 evaluation/agent.py。当前预留能力边界即可，不设计虚构指标或实验结果。

## 3. 模块内部的具体归属

### Knowledge

- models.py：原 schema.py 中的 Note、Chunk、EmbeddingSpec、ChunkRecord、IndexManifest、VectorHit，以及稳定身份、指纹、记录校验。保持纯数据与轻依赖；不放连接或索引构建流程。
- documents.py：原 loader 与稳定扫描，保持当前单层目录、标题提取和正文处理行为。
- chunking.py：完整迁入根部实现，删除根部实现与 indexing/chunking.py 转发入口。章节识别、片段位置、重叠和身份计算规则均保持。
- embeddings.py：合并根 embeddings.py、embedding 专用 tokenization.py 和 OllamaQueryEmbedder。保留输入构造、模型解析、批处理、重试、向量校验和 tokenizer 身份验证。生成模型的消息计数归 Generation。
- indexing.py：保留 build_index、BuildReport、增量/无变更判断、失败恢复、候选核验与发布顺序。
- sqlite.py：保留 SQLite 数据结构、缓存、事务、writer lock 和原子 publish 操作。
- qdrant.py：集中 Qdrant collection 创建、点写入、原始查询、核验、就绪检查、连接与后端身份映射。对外返回 VectorHit 等 Knowledge 数据，不导入 SearchResult。

QdrantConfig 是现有能力配置，建议作为纯配置类型归 knowledge.models；from_metadata 使用纯 metadata 校验，避免反向导入会加载 SDK 的 qdrant 实现。原 point_id/qdrant_identity 等具体后端映射归 knowledge.qdrant。通用指纹仍留 knowledge.models，不能复制到多个包。

### Retrieval

- models.py：原 contracts.py 的 SearchResult、SearchResponse、Retriever、请求校验；合入 snapshot.py 的纯 chunk_result 构造。它可以依赖 knowledge.models，但不做 I/O。
- semantic.py：保留 Embedder、VectorIndex 协议与 SemanticRetriever；接纳 QdrantSnapshotIndex 及对应结果翻译。SDK/具体存储依赖只在需要该适配器时加载，不能让导入 SemanticRetriever 就导入完整 Knowledge 构建器或模型客户端。
- bm25.py：保持现有内存 postings、评分和 snapshot 初始化路径。初始化查询实例的内存结构不等于发布新的知识库索引，本轮不额外拆出 lexical indexing 框架。
- hybrid.py、fusion.py、engine.py：保持现有算法和组合边界。
- rerank.py：合并 Reranker、RerankedRetriever、CandidateScorer 与 CrossEncoderScorer。后者仅在显式构造时加载可选模型依赖；合并文件不改变依赖可选性。
- exact.py：本轮仅占位。它未来表示对源文档的字面/正则等精确检索，不能拿现有 Qdrant exact 向量查询冒充这项能力。

SnapshotSemanticRetriever 和一次性 search_index 是应用组装，迁入 runtime.py。空快照仍校验输入并跳过模型调用，调用方拥有客户端等现有接口语义需保留。

### Generation

原 context 包整体归入 Generation，生成能力继续可以独立调用：

~~~text
SearchResult / Evidence
    -> generation.context.build_context
    -> generation.generate.generate_cited_answer
    -> 带引用的结果
~~~

models.py 放共享的数据/协议：ContextConfig、GenerationCounter、EvidenceBlock、CitationOrigin、CitationSource、Claim、CitedAnswer、CitationValidation 等。

模型文件不反向导入 context、citations 或 generate。BuiltContext 包含最终消息映射与渲染核验行为，可继续定义在 context.py；CitedGeneration 包含引用验证与展示行为，可继续定义在 generate.py。不为“所有 dataclass 都必须放 models”制造循环依赖。

context.py 保留现有证据校验、去重、合并、预算和消息构建；citations.py 保留引用解析与校验；generate.py 保留生成模型调用和对应 token 计数适配。此阶段不同时修复已发现的 Context 可变性或放宽证据契约。

### Agent 与 Interfaces 占位

实施结构迁移时创建 agent 包及 tools/state/loop 文件、interfaces/mcp.py、retrieval/exact.py。占位文件仅包含清晰的模块说明，标注职责与未实现状态。

不导出假成功函数、空工具注册表或虚构的可用 MCP 服务；不将占位能力注册到 CLI/Engine。已有 index/query/status 功能保持可用。确需导出占位函数时应明确报未实现，本轮优先不设计尚无依据的函数签名。

## 4. runtime.py 与 config.py

runtime.py 只承接现有、可复用的组装：

1. 按输入配置和模式打开 SQLite、解析并固定 manifest。
2. 按需准备 embedding client、tokenizer、Qdrant client、BM25、semantic 与 reranker。
3. 组装 RetrievalEngine；生成所需资源也按需准备。
4. 用明确的生命周期管理关闭资源，调用者传入的客户端不擅自关闭。
5. 保留一次性搜索和复用已准备能力的使用方式。

同一会话使用固定 snapshot；新会话可以显式选择最新或历史 snapshot。BM25-only 不连接 Ollama/Qdrant，未启用 rerank 不加载其模型，纯检索不准备生成模型。runtime 不对外暴露一个必须全部初始化才能工作的全能对象，也不新增全局单例或 DI 框架。

config.py 聚合应用配置与默认值。已有 QdrantConfig、ContextConfig 保留唯一类型来源，分别位于 Knowledge/Generation 的 models；顶层配置引用或组合它们，不复制其定义。能力实现接受显式参数或配置对象，不自行读取全局 runtime 状态。

CLI 参数与现有配置优先关系应原样迁移；本轮不新增 TOML/YAML 配置系统或新的环境变量覆盖规则。算法评测默认 exact、应用可用 ANN，以及两者不同 top_k 等有意差异，应显式保留。

所有包初始化与 models/config 的导入保持轻量。迁移时延迟 SDK import 必须限定到确实需要的实现，避免知识模型导入触发客户端、模型或文件访问。

## 5. 生产文件迁移表

| 当前内容 | 新位置 |
| --- | --- |
| schema.py 的记录、通用身份与指纹 | knowledge/models.py |
| schema.py 的 Qdrant point/collection 身份映射 | knowledge/qdrant.py |
| indexing/loaders.py | knowledge/documents.py |
| 根 chunking.py 的真实实现 | knowledge/chunking.py |
| indexing/chunking.py 兼容入口 | 删除，全部调用方改用唯一新路径 |
| 根 embeddings.py、tokenization.py、retrieval/ollama.py | knowledge/embeddings.py |
| indexing/index.py 中构建编排与 BuildReport | knowledge/indexing.py |
| 同文件 QdrantConfig | knowledge/models.py |
| 同文件 QdrantIndex | knowledge/qdrant.py |
| storage.py 中 SQLiteStorage/DDL/序列化 | knowledge/sqlite.py |
| storage.py 中具体 Qdrant 连接/检查 | knowledge/qdrant.py；供模型解码共用的纯 metadata 校验归 knowledge/models.py |
| retrieval/qdrant.py 中 search_qdrant/底层参数检查 | knowledge/qdrant.py |
| 同文件 QdrantSnapshotIndex/snapshot_result | retrieval/semantic.py |
| 同文件 SnapshotSemanticRetriever/search_index | runtime.py |
| retrieval/contracts.py 与 snapshot.py | retrieval/models.py |
| retrieval/reranker.py 与 cross_encoder.py | retrieval/rerank.py |
| 其余 retrieval 算法文件 | 保留位置，迁移导入 |
| context/builder.py | generation/context.py，共享纯数据提取到 generation/models.py |
| context/citation.py | generation/citations.py，共享纯数据提取到 generation/models.py |
| 根 generation.py | generation/generate.py |
| 根 cli.py | interfaces/cli.py，资源组装抽取到 runtime.py |
| retrieval_evaluation.py 的排名统计 | evaluation/metrics.py |
| 同文件 retriever/reranker 实验与 main | evaluation/retrieval.py，公共样本读取提取到 datasets.py |
| evaluation.py 的邻居/证据/引用统计 | evaluation/metrics.py |
| 同文件 compare_retrieval | evaluation/retrieval.py，保留 exact/ANN 实验 API |
| 同文件 Context/引用实验与相应 runner | evaluation/generation.py |
| CLI/评测中应用级配置与重复资源准备 | config.py 与 runtime.py；不抹平实验特有设置 |

旧 indexing、context 目录及根部旧功能文件在迁移完成后删除。所有新增包 __init__.py 只提供必要且轻量的公开入口，不维护新旧两套别名。

## 6. tests 按 capability 组织

tests 的一级目录大致镜像 src/arkb，runtime/config 的测试对应根部文件。真实服务测试放入相关 capability 下的 integration 子目录，跨能力测试按其主要验证的行为归组。

~~~text
tests/
├── __init__.py
├── conftest.py
├── test_runtime.py
├── test_config.py
│
├── knowledge/
│   ├── test_models.py
│   ├── test_documents.py
│   ├── test_chunking.py
│   ├── test_embeddings.py
│   ├── test_tokenization.py
│   ├── test_indexing.py
│   ├── test_sqlite.py
│   ├── test_qdrant.py
│   └── integration/        # 真实 embedding/tokenizer/Qdrant 构建
│
├── retrieval/
│   ├── test_models.py
│   ├── test_bm25.py
│   ├── test_semantic.py
│   ├── test_hybrid.py
│   ├── test_fusion.py
│   ├── test_rerank.py
│   ├── test_engine.py
│   ├── test_snapshot.py
│   ├── test_pipeline.py
│   └── integration/        # 真实 Qdrant 检索、cross-encoder
│
├── generation/
│   ├── test_models.py
│   ├── test_context.py
│   ├── test_citations.py
│   ├── test_generate.py
│   └── integration/        # 真实生成/token 计数/引用
│
├── agent/
│   └── __init__.py         # 当前仅占位，不添加空测试用例
│
├── interfaces/
│   ├── test_cli.py
│   └── integration/        # 真实 CLI 生命周期
│
└── evaluation/
    ├── test_datasets.py
    ├── test_metrics.py
    ├── test_retrieval.py
    └── test_generation.py
~~~

各测试包/子包都添加 __init__.py，避免多个 test_models.py 在现有 pytest 导入方式下重名。示意中的测试文件由现有用例迁入或按职责拆分；不为凑齐目录新增空测试或镜像实现的测试。test_runtime/test_config 承接被抽取逻辑的相关既有用例，并仅在出现新的资源生命周期风险时补必要覆盖。

关键迁移映射：

| 现有测试 | 目标 |
| --- | --- |
| test_schema/loaders/chunking/embeddings/tokenization/indexing/storage | knowledge/ 下相应测试 |
| test_retrieval_contracts | retrieval/test_models.py |
| test_retrieval.py | 按用例拆到 knowledge/test_qdrant.py、retrieval/test_snapshot.py、test_runtime.py |
| test_bm25/semantic/hybrid/fusion/engine | retrieval/ 下同名测试 |
| test_reranker 与 test_cross_encoder | retrieval/test_rerank.py |
| test_retrieval_pipeline | retrieval/test_pipeline.py，保持离线运行 |
| test_context/citation/generation | generation/ 下相应测试 |
| test_cli | interfaces/test_cli.py；抽取逻辑对应断言随 runtime/config 迁移 |
| test_retrieval_evaluation 与 test_evaluation | evaluation/ 下按数据、指标、检索、生成实验拆分 |
| integration/test_qwen_chunking/embeddings/tokenizer、test_qdrant_indexing | knowledge/integration/ |
| integration/test_qdrant_retrieval、test_cross_encoder_model | retrieval/integration/ |
| integration/test_qwen_citations | 生成行为归 generation/integration；其中 CLI 流程按用例归 interfaces/integration |
| integration/test_indexing_lifecycle | interfaces/integration/ |
| test_context 中 4 个真实生成计数参数化样本 | generation/integration/，仍保留全部样本 |

测试约束：

- 根 conftest 仅放跨能力共享 fixture，能力私有 fixture 放各自 conftest；跨能力共享 Qdrant fixture 不放到只有 Knowledge 可见的位置。
- 普通 mock/Qdrant Local 测试继续属于常规测试；不能因为是 pipeline 就改成需外部服务的测试。
- 真实服务用例保留现有显式环境开关，并统一添加、注册 integration 标记。常规验证用 pytest -m "not integration"，不再依赖旧顶层 tests/integration 排除路径。
- 迁移所有 monkeypatch/mock 字符串、subprocess 模块入口和测试 helper 的导入。
- 核对迁移前后的实际用例清单与参数化样本，不仅比较总数。此前基线 561 passed、4 skipped；4 个样本转到 integration 后，常规集合的跳过数变化有明确原因，不应误认为覆盖被删除。

## 7. 实施顺序与每批交付

### 第一批：Knowledge 收拢

建立 knowledge，迁移模型、文档、唯一 chunker、embedding/tokenizer、SQLite 和 Qdrant；拆出索引编排。同步更新当前消费者和 Knowledge 测试。保留序列化、指纹、坐标、缓存和发布语义。

### 第二批：Retrieval 整理

迁移 models，合并 rerank，划清 semantic 与 Knowledge 原始后端操作；占位 exact。继续保留独立算法 API、可选模型加载和 provider 无关的基础检索能力。同步迁移 Retrieval 测试。

### 第三批：Generation 收拢

将 Context、引用、生成归入 generation。先提取共享模型，确保模型层不反向导入实现；再迁移 context/citations/generate。同步迁移 Generation 测试，保留原有 prompt、预算、引用及生成行为。

### 第四批：Runtime、配置与 Interfaces

抽取重复资源组装，归位 SnapshotSemanticRetriever/search_index。建立 config 聚合，移动 CLI，更新 pyproject 的脚本入口为 arkb.interfaces.cli:main。创建 Agent/MCP 占位。更新 CLI、资源生命周期和配置测试。

### 第五批：Evaluation 与最终清理

迁移样本、指标和实验，保持 retrieval/ANN/frozen-rerank/Context/citation 能力。更新 benchmark、README、文档示例、测试和安装入口；清除旧目录、文件、转发与 import 路径。

迁移后的评测命令采用 python -m arkb.evaluation.retrieval 和 python -m arkb.evaluation.generation。Retrieval 入口用 baseline/ann 子命令区分原来的相关性对比和 exact/ANN 对比，每种实验保留自己的参数与默认值。原 ANN runner 同时启用 Context/引用评测的能力继续保留：调用 evaluation.generation 的实验函数，并共享同一快照、query 向量和命中，保持组合报告内容。详细调用映射同步更新到 benchmark 与文档。产品 arkb index/query/status 命令和输出保持。

每批都迁移实际消费者并运行对应测试，临时过渡导入必须在整体交付前清除。占位文件创建属于目录交付，功能开发属于后续任务。

## 8. 验收标准与范围

- 一级能力包恰为 knowledge/retrieval/generation/agent/interfaces/evaluation，根部功能文件只有 runtime.py、config.py。
- chunking 只有一个规范实现；旧 indexing、context 和根部功能文件不再保留转发壳。
- Knowledge 不导入 Retrieval/Generation/Agent；Generation 与 Retrieval 不导入 Agent；生产能力不导入 Evaluation。
- models、config 和包初始化不会加载模型或创建服务连接；导入纯检索 API 不要求安装可选 reranker。
- 排名、分数、模型输入、片段坐标、Context/引用和序列化行为保持；持久化 schema、hash 命名空间、cache key、Qdrant 身份不变。
- 目录迁移不要求重建现有索引；固定 snapshot、只读查询、失败不覆盖有效版本与资源关闭继续得到验证。
- tests 按能力组织；服务测试的标记、条件和 fixture 范围正确；原有常规用例完整执行。
- wheel 中新入口可用，不包含旧重复模块；CLI、评测脚本与文档均使用新路径。
- Agent、exact、MCP 占位可导入但不会被宣称为已实现，不增加新依赖或假成功行为。

Python import 与评测启动路径的调整会写入迁移表；功能与持久化兼容保持。已发现的 Context 可变性、原文坐标、多轮证据、中文分词与性能问题留作后续独立变更，不在本次结构迁移中顺带修改。
