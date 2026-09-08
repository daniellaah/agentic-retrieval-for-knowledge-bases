# Agentic Knowledge Retrieval：架构与后续开发边界

项目目标是让 Agent 根据用户意图选择检索、读取原文、继续搜索或生成回答。
当前阶段已经交付可独立调用的知识与检索设施；Agent 控制循环尚未实现。
不要把 CLI 当前的固定 vector → context → generation 流程当成未来所有任务的统一策略。

## 当前职责

| 层 | 当前实现 | 约束 |
| --- | --- | --- |
| Knowledge | Note/Chunk、身份、知识快照、加载、切块、metadata、lexical/vector index | 数据准备和确定性计算；不决定用户任务的检索策略 |
| Retrieval | grep、metadata、BM25、vector | 每种方法有自己的参数，统一返回 SearchResponse |
| Context | 摘录去重、重叠合并、预算与消息构造 | 只使用实际摘录；无摘录的笔记定位需要先读取原文 |
| Citation/Response | 结构化回答、引用与原文引句校验 | 引用结构正确不等于事实支持已验证 |
| Evaluation | 固定语料/问题/参数，保留结果、来源和运行元数据 | 区分近邻召回、证据覆盖与答案正确性 |
| Agent / Evidence Store / Tool Registry | 后续阶段 | 不在现有底层函数中夹入 LLM 决策或自动循环 |

模块文件使用 snake_case，类使用 CapWords，函数使用 snake_case。
`retrieval/vector.py` 中的入口是 `vector_search`；其他方法对应 `grep.py`、
`metadata.py`、`bm25.py`。有状态设施采用 `KnowledgeSnapshot`、`LexicalIndex`、
`SQLiteStorage`、`QdrantIndex` 等具体名称。类型随职责放置，不添加 `Model` 后缀，
不为每个 dataclass 新建文件。知识记录与检索契约分别集中在各自域内的 models.py。

## 统一协议的含义

统一的是结果语言，不是强行统一所有算法的输入。
vector 需要模型、tokenizer 与 Qdrant 客户端；metadata 使用结构化过滤；
grep 接受字面字符串；BM25 使用词法统计。这些差异保留在各自的参数中。

SearchResult 必须说明 source、target、method、rank，可以附带 score 和 excerpts。
source 保留 vault、文档身份、文档版本与快照。target 明确是整篇笔记、任意正文范围，
还是有稳定 ID 的 chunk。score 必须声明 metric 和排序方向，也可以完全没有。
不要求 metadata 或 grep 为了适配向量检索而创建伪 chunk、伪分数。

SearchResponse 说明 query、method、scope、items、limit 和完整性。
grep/metadata/BM25 的分页能明确说明还有没有结果；vector top-k 的 has_more 为 unknown，
不能据此声称“整个知识库都没有其他相关信息”。页内 rank 从 1 开始。
不同方法的原始分数不能直接相加或横向比较；未来融合需要明确的算法与评估。

## 同一任务如何固定来源

1. 新加载语料：通过 `KnowledgeSnapshot.from_notes` 创建不可变内容快照。
2. 已有向量索引：先固定 READY `index_version`，通过
   `SQLiteStorage.knowledge_snapshot(version)` 得到同版本知识视图。
3. 使用该视图调用 grep、metadata、BM25、read_note/read_span，并在 vector_search
   中固定同一 index_version。不要重新读取 live 文件来解释旧检索结果。
4. SourceRef 可直接用于同一知识视图的原文读取；不同版本的引用不得静默替换。

旧 SQLite v1 保存的 chunk 覆盖完整正文，因此可以在不读取 embedding BLOB 的情况下
恢复完整笔记。恢复过程验证连续覆盖、重叠一致性、文档 revision 与 corpus fingerprint。
如果来源缺失或校验不符，明确报错；不能凭空补文本或直接读取当前文件顶替。

字符坐标始终对应规范化加载后的 Note.content，end exclusive。它不是原文件字节坐标。
加载器目前保持既有的单层目录行为。元数据提取不移除 frontmatter，也不改变这些坐标。
BM25 的 NFC/casefold 仅影响词项；返回的正文保持原样。

## 检索能力当前范围

- grep：literal 搜索正文，默认区分大小写；返回匹配行或多行原文范围，同一范围去重。
- metadata：标题包含、路径前缀、YAML tags/aliases 的 AND 过滤；返回笔记候选。
  标题/tags/aliases 忽略大小写，路径保持字面比较；不把正文 hashtag 当作 YAML 属性。
- BM25：可在整篇笔记或共享 chunk 上建统计；默认 k1=1.2、b=0.75；中文使用汉字
  unigram/bigram，其他字母数字按词项处理。当前是进程内不可变索引，修改语料后显式重建。
- vector：唯一后端为 Qdrant，负责问题嵌入、过滤、top-k、exact/ANN 和来源解析；
  embedding/cache/chunking/collection lifecycle 不放在 retrieval/vector.py 中。

grep/metadata/BM25 当前提供 Python 能力接口，尚未新增通用工具 CLI。
递归 Obsidian vault 扫描、完整 Obsidian metadata/link 语义、词法索引持久化与专用分词器
都需要独立的后续设计及测试，不能在此阶段声称已支持。

## Context 与后续 Evidence Store

搜索候选和回答证据是不同阶段。没有 excerpts 的 NoteTarget 先经过 read_note/read_span；
Context 不自动读取文件，也不把标题、tag 命中当成正文事实。当前用 needs_inspection 决策
标记这类结果。读取整篇笔记可以生成 method=read_note 的结果，附原文摘录进入 Context。

相同文档/版本的相交范围可以合并，前提是重叠文字完全一致。同一正文由 vector、grep、
BM25 找到时保留各方法来源。Context 当前按照输入优先级选择证据，保留原始分数信息，
不做跨方法分数融合。预算针对最终实际发送的模型消息。

后续 Evidence Store 应记录操作 ID、来源、读取动作、证据覆盖和重复度，并保持搜索日志
与当前可用证据的区别。不要把现有 BuiltContext 直接扩张成 Agent 状态容器。

## 后续开发顺序

基础模块完成后，再逐项讨论并实现以下阶段：

1. Tool Registry 与能力 schema：描述每个工具的输入、响应、完整性和错误；支持仅返回
   搜索结果，以及从候选继续 inspect。避免暴露底层数据库内部结构给 Agent。
2. Evidence Store 与 trajectory：固定任务快照，记录工具调用、参数、检索来源、原文读取、
   延迟、预算和阶段状态。确定停止/继续的可观察依据。
3. 最小 Agent 循环：意图判断、工具选择、观察、继续/停止、输出模式；限制步数与资源预算。
   先验证简单 grep discovery 与 vector → read_note 两类任务，再增加多轮研究任务。
4. 融合、reranking 与检索扩展：根据固定问题、必要证据和实际失败样例选择实现，不预先
   把所有查询绑定到 hybrid，也不把所有底层算法步骤交给 LLM。
5. Agent 评估：在检索质量之外验证工具选择、无效调用、过早停止、证据缺口、回答支持与成本。

新增行为采用公开接口的 red → green → refactor 小步开发。只在模型或外部服务边界使用替身；
数据库、快照与来源校验优先运行真实临时实例。完成一个模块后检查 diff、运行相关回归并提交。
真实服务测试必须显式启用，普通套件跳过这些测试不能代替集成验收。
