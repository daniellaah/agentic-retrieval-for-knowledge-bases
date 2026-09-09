# 面向 Agentic Retrieval 的架构审查

审查基线：`9d47667`。范围包括 `src/arkb`、常规测试、集成测试代码和已有 benchmark；本轮只新增审查文档，没有修改实现、接口或索引。

后续调整：当前采用用户确定的 Knowledge、Retrieval、Generation、Agent、Interfaces、Evaluation 六个能力边界，先完成目录组织。唯一 chunking 实现归 knowledge/chunking.py，Context 与引用归 Generation，根部 runtime.py/config.py 负责组装与应用配置，tests 按能力归组。具体迁移与执行顺序见[目录重构方案](directory-refactor-plan.md)。该方案取代本文此前的目录建议和实施优先级；本文已验证的行为发现仍然有效。

当前代码有 29 个 Python 文件、4,228 行。直接内部 import 的静态扫描未发现循环，扫描包含函数内 import，但不代替完整的运行时依赖分析。检索核心已有脱离 provider、storage、context 和 generation 的测试。

**判断：保留当前检索核心；优先修复证据可变性、收拢应用组装，并为文档访问和多轮证据明确接口。目录调整应随这些职责变化发生。**

## 1. 当前能力与后续任务

| 目标 | 当前状态 | 审查结论 |
| --- | --- | --- |
| 独立比较 BM25、semantic、hybrid、rerank | 已有 primitives、Engine、评测入口 | 保留；独立调用是研究能力 |
| 查询不修改索引，失败不替换有效快照 | 已有只读查询、候选构建、核验、原子发布 | 保留，作为重构验收条件 |
| 从检索片段构建带引用回答 | 已实现，要求有效 snapshot chunk | 维持现有路径，补齐下述证据边界 |
| `knowledge_search` 高层工具 | 可建立在现有 Engine 上 | 工具负责行动参数与结果表达，固定算法策略由应用配置 |
| `exact_search`、`read_note`、`list_notes` | 尚未实现 | 先定义文档访问、版本和坐标，不能只增加工具包装 |
| 多轮发现、读取、再搜索、停止 | 尚未实现 Agent loop 和证据累计 | 属于后续功能；不要求现在搭建通用 Agent 框架 |
| CLI、API、MCP 共用业务流程 | 当前只有 CLI；资源组装散在多个入口 | 先复用当前 CLI 和评测所需的组装逻辑 |

不应把“尚未开发未来功能”记作当前检索算法的缺陷。以下分别标注当前问题、简化机会和接入未来功能前的准备。

## 2. 发现与处理建议

### F01：Context 的引用来源仍能被外部修改

**类别：当前可复现问题。建议：最先独立修复。**

位置：[SearchResult](../src/arkb/retrieval/contracts.py) 的 `__post_init__`（51–71 行）、[EvidenceBlock / BuiltContext](../src/arkb/context/builder.py)（95–165 行）。

`SearchResult` 使用 frozen dataclass，构造时复制 metadata，但其内部字典仍然可变。`EvidenceBlock.origins` 直接保留输入 hit，`BuiltContext.citation_sources` 又在访问时从这些 hit 读取版本等信息。

最小验证：构建 context 后，修改输入 hit 的 `metadata['document_revision']`。结果是：

- 已渲染的 messages 没变。
- citation source 的 document revision 改变。
- `context_id` 改变。
- `verify_citation_mapping()` 仍然通过。

当前 CLI 顺序执行时未发现主动进行此修改的路径；但这不符合 Context 的不可变语义，多轮状态复用也会增加误修改的机会。

建议在证据进入 Context 时保存独立、不可变的来源快照，避免后续从调用方可变字典重建引用信息。单纯缓存 `context_id` 会掩盖变化，单纯 deepcopy 输入也不能防止通过 `context.evidence_blocks[*].origins` 再次修改内部字典。需要同时明确内部存储与对外访问的边界。

验收：修改原始 metadata、输入 hit.metadata，以及尝试修改对外暴露的 metadata，都不能静默改变已构建 Context 的引用和 ID；未修改输入时，既有 JSON、引用定位和 context_id 计算结果保持一致。

### F02：通用检索结果与当前引用模型没有贯通

**类别：接入工具前需要完成的接口工作。**

位置：[contracts.py](../src/arkb/retrieval/contracts.py)（17–71 行）、[builder.py](../src/arkb/context/builder.py)（264–308 行）、[citation.py](../src/arkb/context/citation.py)（34–55 行）。

`SearchResult` 允许没有 score、chunk 和 index 的结果；但 Context 要求有限分数、可重建的 ChunkRecord、chunk ID、索引版本。`CitationOrigin` 也强制要求这些字段。因此不能只放宽 Context 的一次校验；引用类型和证据来源转换也需要一起审查。

另一个已验证的限制：同一个 chunk 分别作为 semantic、BM25 命中传给 `build_context`，只保留首次命中的 origin，第二次只留下 duplicate 决策。当前“单次排序列表 → Context”的行为有明确代码和测试依据；它不能直接承担多轮搜索轨迹的完整保存。RRF 自己保留的算法贡献信息仍然存在，这与多轮工具调用来源是两个问题。

建议区分三类数据：

| 数据 | 最小职责 |
| --- | --- |
| 文档/片段证据 | 文档身份、版本、坐标体系、原文内容；能验证来源 |
| 检索或读取记录 | 哪一步、哪个 query/tool 发现了证据；可选分数与排名 |
| 最终 Context | 从证据中选出什么、合并了什么、预算舍弃什么、最终引用编号 |

先从 `context/builder.py` 提取现有证据校验、去重、重叠合并到 `context/evidence.py`，保持原有行为。新增无分数证据、跨轮来源累计应作为后续功能提交，避免把语义变更混入文件移动。

`list_notes` 的导航结果未必应该直接成为可引用正文；可以先供 Agent 选择，再调用 `read_note`。不要为了统一工具输出而强迫所有元数据列表进入 Context。

验收：search → read_note 能落到同一文档版本和坐标；相同内容只占一次正文预算，多次发现记录仍可追踪；不靠伪造 score 或 chunk ID 接入原文读取。

### F03：目前缺少独立的文档访问与原文坐标契约

**类别：接入 exact/read/list 前的必要设计。**

位置：[loaders.py](../src/arkb/indexing/loaders.py)（8–53 行）、[Note](../src/arkb/schema.py)（18–32 行）、[storage DDL](../src/arkb/storage.py)（20–37 行）。

当前 loader 只扫描一层 Markdown，source 保存文件名；移除第一条 `# ` 标题行后还会裁剪正文空白。索引坐标明确对应 `Note.content`，不是原始文件。验证样本 `# Title\n\nalpha body\n` 中，alpha 位于原始文件第 3 行、字符位置 9，却位于加载后正文位置 0。子目录中的笔记没有被加载。

SQLite 保存 chunk 记录、embedding 缓存和构建信息，没有独立保存完整原文及其目录/标签/时间索引。不能假定拼接 chunk 可以无损恢复原始文件。BM25 查询虽可脱离模型和 Qdrant 连接，但当前持久化构建仍需要 embedding 和 Qdrant；`snapshot_chunks.embedding_key` 也依赖 embedding 缓存。

接入文档访问时需要明确：

- source 使用相对知识库根目录的规范路径，嵌套同名文件不冲突。
- 行号是原始文件行号还是标准化正文行号；引用使用哪一种字符坐标。
- 搜索旧快照后，read_note 如何读取该版本；读取最新文件时如何检测和表示版本变化。
- 未建向量索引时，exact/read/list 的能力边界。
- 标签、修改时间、链接由文档解析/访问层提供，检索过滤只消费其约定。

建议第一步明确坐标和版本规则，再决定保存原文快照或维护可验证的位置映射。原始路径、字符位置进入现有身份计算和评测标注；修改 loader 不能当作纯搬目录，需要明确旧索引重建或迁移规则。

不建议为了让导航工作，先把现有向量构建流程拆成全新的多后端索引框架。可先让文件读取与导航拥有自己的小接口，保留现有 snapshot 查询路径。

### F04：资源组装与配置来源存在真实重复

**类别：现在可以进行的简化。**

位置：[cli.py](../src/arkb/cli.py)（178–205 行）、[retrieval_evaluation.py](../src/arkb/retrieval_evaluation.py)（166–209 行）、[qdrant.py](../src/arkb/retrieval/qdrant.py)（133–168 行）。

CLI 和评测都需要读取 manifest、解析 backend 信息、打开客户端、加载 tokenizer、核对模型和构造 retriever。`retrieval/qdrant.py` 还同时承担 Qdrant adapter 与 Ollama + Qdrant 应用组合。

建议增加一个小的 `application.py` 作为当前应用的组装入口，只负责明确的资源创建、模式所需能力和关闭边界。复用适配器初始化与默认常量；CLI 继续解析参数和展示结果，评测继续选择自己的算法组合、参数与指标。先不增加 DI 容器、provider registry 或通用插件机制。

可直接收拢的例子：reranker 的默认模型和 revision 已在 `cross_encoder.py` 定义常量，CLI 与评测仍重复书写；评测入口 164–165 行已验证 `top_k <= rerank_candidates <= candidate_k`，204–205 行又在同一路径、相同参数上验证其上界。后者可以消除重复，但跨公共入口或外部服务返回边界的校验仍需保留。

应保持的有意差异：

- CLI 可使用 ANN，现有算法评测使用 exact；不要通过共用默认值悄悄改变实验。
- BM25-only 查询不打开模型或向量服务。
- 评测可以直接用 primitives，不强制一切经过 Engine。
- `SnapshotSemanticRetriever` 的空快照路径会校验输入并跳过模型；它有实际行为，不能仅因是包装类就删除。
- `search_index` 是已有的一次性入口，迁移内部位置时应保留其委托入口。

验收：CLI 与评测使用同样配置时构造相同能力；客户端在一次会话中可复用并正确释放；同一会话固定 snapshot；错误与空结果继续区分。

### F05：每次查询全量计算 tokenizer 指纹有可测开销

**类别：量化优化候选；需要先解决资源所有权。**

位置：[ollama.py](../src/arkb/retrieval/ollama.py)（24–29 行）、[tokenization.py](../src/arkb/tokenization.py)（63–64 行）。

每次 `prepare()` 都执行 tokenizer.to_str()、UTF-8 编码与 SHA-256。使用本地已缓存的固定 tokenizer、无网络与模型调用，预热后各重复 7 次：

| 操作 | 中位耗时 |
| --- | ---: |
| tokenizer 指纹 | 31.318 ms |
| 短查询 token 计数 | 0.015 ms |

序列化 tokenizer 大小为 5,363,472 字节。这是本机微测量，不是端到端提速预测，也不能直接与历史 benchmark 相除得到当前耗时占比。

这里的重复校验还承担检测 tokenizer 变化的职责。直接按对象 ID 缓存指纹会漏掉内部状态变化。应先让应用会话拥有配置固定的 tokenizer，再考虑把完整身份校验移到资源准备阶段；现有允许调用方持有可变 tokenizer 的接口不能在无替代保证时取消逐次检查。

索引写入的重复读取、逐 chunk 事务另有[既有测量](../benchmarks/indexing-overhead.md)。它们和发布前完整性核验有关，建议单独优化和验证，不把删除检查当作目录简化的附带操作。

### F06：评测能保护算法回归，但尚不足以覆盖目标使用场景

**类别：现在补充少量目标样本；完整 Agent 评测随功能落地。**

位置：[BM25 tokenizer](../src/arkb/retrieval/bm25.py)（21–22 行）、[ranking evaluation](../src/arkb/retrieval_evaluation.py)（40–86 行）、[测量样本](../benchmarks/retrieval-example.md)。

当前样本为 40 文档、6 查询，已有文档明确限定其用途。BM25 采用 Unicode word token，无中文分词。最小验证中，正文“知识库检索可以结合向量搜索。”无法被短查询“向量搜索”命中，而整句查询可以。这是已声明分词规则的限制，是否替换分词器要由目标语料和对照实验决定。

建议增加中文/混合语言、精确标识符、多个相关片段来自同一笔记、无答案、过滤范围和跨轮补证据场景。先固定旧 baseline，再单独评估分词调整；它会改变排名，不能归入行为不变的重构。

需明确的评测口径：

- 当前先取 K 个 chunk，再按文档去重，不等价于返回 K 篇不同笔记。研究 chunk 检索与产品“找笔记”应分别报告。
- 没有正标签时，排名指标为 undefined；不代表已评估“是否应停止或拒答”。
- query 耗时不包含模型/tokenizer 加载和 BM25 postings 构建；首次使用和多轮复用需要分别测量。
- 普通持久化评测记录配置、源码哈希、manifest；本地示例 runner 额外记录部分依赖版本。可共用最小实验环境记录函数，使报告口径一致。
- ANN 邻居 Recall、相关性 Recall、引用结构正确性、答案受证据支持的程度是不同指标，不应因名字相近合并实现。

## 3. 目录与冗余设计的逐项判断

| 当前区域 | 决策 | 理由和边界 |
| --- | --- | --- |
| `retrieval/engine.py`、`hybrid.py`、`fusion.py`、`reranker.py` | 保留 | 模式选择、候选组合、纯排序融合、独立重排承担不同职责；Engine 仅 41 行 |
| `Retriever`、`Embedder`、`VectorIndex`、`CandidateScorer` | 保留 | 简单结构化协议，支撑测试替身和独立实验，没有必要改成复杂基类 |
| `retrieval/snapshot.py` | 保留 | BM25 与 semantic 共用的证据翻译，集中保护字段一致性 |
| `retrieval/qdrant.py` 的应用组装部分 | 合并到应用组装入口 | 后端读取 adapter 留在原处；迁移后保留现有公共委托接口 |
| `indexing/index.py` | 保留，暂缓进一步分包 | 当前只有一个生产向量后端；构建、核验、发布具有紧密生命周期关系 |
| `indexing/loaders.py` | 目前保留；文档访问功能落地时归位 | 若新增 `documents/`，在此集中原文读取、解析和 metadata；不要再平行创建重复的 ingestion 流程 |
| 根目录 `chunking.py` | 保留 | 已是唯一 chunker 实现，无需为对称目录再次移动 |
| `indexing/chunking.py` | 迁移内部消费者；暂留公共兼容入口 | 仅 6 行，是转发而非算法重复；README 和多个测试、脚本仍使用 |
| `schema.py` | 暂不机械拆分 | 核心身份和版本规则应只有一个来源；新增文档模型后再按稳定职责拆分 |
| `storage.py` | 保留 SQLite 主体 | 约 30 行 Qdrant 辅助函数归属可随应用组装整理；不足以单独引入大型 adapters 层 |
| `context/builder.py` | 分离证据处理与最终预算/渲染 | 支撑未来多轮累计；先修复 F01，再做行为不变的提取 |
| `evaluation.py`、`retrieval_evaluation.py` | 保留能力，共用组装与实验记录 | 前者含 ANN/Context/引用评测，后者含相关性排名评测；不能按相似命名删掉其中一个 |
| `cli.py` | 保留单文件入口 | 暂无第二个用户接口，不必只为一个 CLI 建 `interfaces/` 包 |
| `README.md` | 后续压缩入门路径，详细说明归入 docs | 保留现有示例和兼容说明，避免结构重构时同步大幅重写使用文档 |
| `tools/`、`agent/`、`interfaces/` | 随真实功能建立 | 不提前创建空目录、空 registry 或抽象执行框架 |

当前最小的结构增量是 `application.py`，以及在职责提取阶段引入 `context/evidence.py`。原文访问开发时再增加 `documents/`；API/MCP 开发时再考虑 `interfaces/`。无需立即把所有共享模块移动到 `core/` 或 `common/`。

## 4. 实施顺序与验收边界

| 批次 | 工作 | 验收 |
| --- | --- | --- |
| A：正确性修复 | F01，隔离并稳定 Context 引用来源 | 构建后外部修改不改变引用与 ID；原有序列化和引用结果不变 |
| B：行为不变的简化 | 应用组装共用、默认常量共用、内部 import 迁移、现有证据函数提取 | CLI/评测等价；保持 exact/ANN 差异、客户端释放、空快照行为、兼容入口 |
| C：文档与证据接口 | 原文/正文坐标、版本读取、无分数证据、多轮来源 | search → read → context 可验证；任何存储/身份变化有独立迁移或重建说明 |
| D：工具与 Agent | 高层工具、累计证据、查询改写、停止条件 | 保留独立算法 baseline；开始记录步数、查询、选择与预算 |
| 独立性能批次 | tokenizer 身份校验、索引重复工作 | 在保留数据核验保证后，用同样工作负载复测；不与目录调整混在一起 |

公共 import 兼容、CLI 参数和输出、存储/身份兼容是三种不同承诺。迁移内部消费者不等于可以删除外部入口。当前没有充分依据认定这些入口都没有外部使用者，因此本报告不建议直接删兼容 API。

## 5. 本轮验证与局限

常规测试命令：

```sh
OBSIDIAN_RAG_RUN_MODEL_TESTS=0 HF_HUB_OFFLINE=1 \
  .venv/bin/python -m pytest -q --ignore=tests/integration
```

结果：**561 passed, 4 skipped，2.64 秒**。4 项跳过均为真实 Ollama generation token 计数样本。集成测试目录本轮未执行，不据此判断真实 Qdrant Server、ANN 或模型推理表现。

独立最小探针还验证了：无分数证据被当前 Context 拒绝；重复命中只保留首个 origin；Context 来源可变；原文与正文位置不同；子目录未扫描；中文 BM25 短查询限制。探针仅使用临时文件、合成数据和本地代码，没有修改现有知识库或索引。tokenizer 微测量仅加载本地缓存。

F01 的最小复现（从项目根目录的 Python 环境执行）：

```python
from arkb.chunking import whole_note_chunks
from arkb.context import build_context
from arkb.retrieval.snapshot import chunk_result
from arkb.schema import ChunkRecord, Note

note = Note('Title', 'alpha body', 'a.md')
record = ChunkRecord.from_note(
    whole_note_chunks([note])[0], note=note, vault_id='v')
hit = chunk_result(record, method='semantic', index_id='snapshot',
                   score=.8, score_type='cosine_similarity')
context = build_context('alpha', [hit])
before = context.context_id
hit.metadata['document_revision'] = 'modified-after-construction'
context.verify_citation_mapping()  # 当前仍通过
assert context.context_id != before  # 当前已发生变化
```

本轮未发现需要重写 Retrieval Engine 的证据。目录简化的主要收益来自明确应用组装、文档访问与证据处理的职责，而现有算法组合和索引正确性保证应继续作为稳定基础。
