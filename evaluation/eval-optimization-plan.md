# ARKB Evaluation 优化方案

日期：2026-09-10。审查代码：`0a3e89d`。状态：调研与设计完成，v2 数据标注和运行能力尚未实施。

建议先建立可信的测量，再用测量选择优化：**冻结 v1 作为回归集，建立独立的 v2 数据与证据标注，分开评估检索排序、多步取证、最终输出和运行成本，最后在受控实验中决定改什么。** 当前最值得优先验证的方向是 match 的完整枚举、跨语言查询、候选覆盖、段落选择与停止控制。扩大模型或替换 embedding 应放在这些诊断之后。

本方案暂按“个人技术知识库，中文提问，检索中英文笔记，同时支持找资料与知识问答”设计。尚未获得真实用户查询分布、标注人力和产品延迟目标；下文的数据配比、时间、预算和门槛均是**待 pilot 校准的工程提案**，不是行业统一标准，也不是已经测得的效果。公开基准与项目数据分别报告，不合成一个总分。

## 1. 当前系统与评测审计

### 1.1 已有资产

| 资产 | 已确认的内容 | 适合回答的问题 |
| --- | --- | --- |
| [Agent v1](data/agent_v1.jsonl) | 40 个 case；6 exact、6 semantic、6 direct read、10 exploratory、6 QA、6 no retrieval | 来源覆盖、工具约束、read 是否返回目标来源、是否正常停止 |
| [检索回归样例](../benchmarks/retrieval-cases.jsonl) | 6 个英文问题，source 相关性等级 1/3；其余未列来源按 0 评分 | 最小排序回归与示例，不能代表真实质量分布 |
| [Agent 指标](../src/arkb/evaluation/agent_metrics.py) | 成功工具 observation 中 source 的集合并集；失败、分母、null 明确 | 可确定性复算 Agent 轨迹 |
| [固定 baseline](../src/arkb/evaluation/baselines.py) | BM25 / Semantic / Hybrid / Hybrid + Qwen Rerank；原始 query 单次搜索 | 固定查询下的来源排序与延迟 |
| [retrieval 诊断](../src/arkb/evaluation/retrieval.py) | 分级 qrels、冻结候选重排、exact/ANN 对照、原始排名 | 候选与重排、近似索引损失 |
| [evidence/context/citation 指标](../src/arkb/evaluation/metrics.py)、[generation 评估](../src/arkb/evaluation/generation.py) | 来源 OR 分组、正文区间并集覆盖、context 重复率、引用 ID 校验、外部 review 字段 | 已有可复用基础，但不等于主 Agent 数据已有这些标注或语义评分 |
| [运行器](../src/arkb/evaluation/runs.py) | dataset 副本/hash、源码指纹、语料边界指纹、完整最小 trace、错误保留、唯一产物目录 | 离线评分回放和运行审计 |

v1 的优点应保留：标签不会传给被测 Runtime；不用唯一工具顺序约束模型；失败不重试、不从 Agent 成功率中删除；没有语义评分时不会伪造答案准确率；保存原始观察，能重新评分。优化不是推倒重建。

### 1.2 本次独立核对

可复现入口：[审计脚本](audits/20260910_review.py)；完整机器结果：[audit.json](audits/20260910-audit.json)。脚本只读取文件并调用纯评分函数，不调用模型或打开数据库。

- 原始 v1 SHA-256 为 `8babda40f2a74506c3fefe5159a9094a91c86e4e5562443a09c7b35790a0e8af`，与四组历史运行保存的数据一致。
- 复算 **360 条 Agent 记录**的全部 v1 case 指标与 v1 汇总字段；从原始 response 重建 **160 条 baseline 记录**的来源排名，复算逐条指标及 overall 指标均值。全部一致；baseline 包含 136 次检索与 24 条不适用记录。
- 四组 run metadata 的逻辑 snapshot 内容相同：40 documents / 160 chunks；保存的 before/after 指纹一致，当前 example_notes 的文件 hash 仍与当时一致。本次没有重新核验运行中的 Qdrant vector，也没有复跑实时模型。
- 当前评测确定性测试：`.venv/bin/python -m pytest -q tests/evaluation -m 'not integration'`，**156 passed**。这证明已测评分契约与接线通过，不证明模型回答正确。
- 三个模型各自 40 个 case 的三次 trial 都具有相同的 v1 指标及工具调用参数/轮次。不能把 120 次重复当成 120 个独立信息需求。
- 另做两个**合成评分契约探针**：对 `qa_001` / `read_001` 返回正确 source 名、无关正文和故意错误回答，v1 仍判成功。这符合现有设计，直接说明其成功率不包含证据/答案语义验证；探针未混入历史实验分数。

### 1.3 语料与查询分布

40 篇 Markdown 合计 80,700 个原始字符，最短/中位数/最长为 606 / 733 / 6,240。人工查看与字符检查一致：语料为英文，39/40 个 query 含中文。正例标签涉及 33/40 篇，7 篇从未成为 expected source；`40_query_planning_and_compression.md` 出现在 9 个 case 中。

这批数据存在几个局限：

1. **语言与算法混杂。** 原始中文 query 对英文词项匹配不友好。当前 BM25 使用 Unicode `\w+`，没有中文分词、词干化或跨语言转换。其低分支持“这个输入条件下不够好”，不能推出“BM25 普遍无效”；仅添加中文分词也不能解决中文 query 对英文 corpus 的语义对应。
2. **小语料、有限主题、表述相近。** 许多笔记使用相似教学结构。扩大 query 数而不扩展语料、用户意图和困难负例，可能只增加近重复样本。7 篇无正例不必强行补齐，但应检查是否存在重要场景盲区。
3. **缺少与知识库相关但无法回答的查询。** no_retrieval 是问候、改写等“不需要查”，不等于“查过仍没有答案”。当前 loader 也不允许普通 retrieval case 的 expected_sources 为空。
4. **多来源不一定是多跳。** 多主题阅读清单和必须连接 A→B 的证据推理应分开。当前多来源集合不能表达“找到这一篇才知道下一篇是什么”。
5. **缺少标注过程和留出集。** 当前文件不记录 query 来源、独立复核者、分歧、pool、split、同源改写关系。不能从 curated notes 推断已经双人盲标。既有 40 case 已用于多轮诊断，适合回归，不适合重新命名为盲测。

### 1.4 历史实验应该怎样解释

| 历史实验 | 核对结果 | 能支持的结论与边界 |
| --- | --- | --- |
| [模型消融](phase1-agent-model-ablation-results.md) | 4B / 9B / 27B 成功率 77.5% / 85.0% / 95.0%；source recall 85.88% / 89.61% / 88.53% | 模型影响取证和停止；27B 不是所有指标都最好，95% 不是答案准确率 |
| [固定检索](deterministic-baseline-results.md) | 34 个适用 case：Semantic / Hybrid / Hybrid + Rerank 来源 R@10 为 82.79% / 84.61% / 87.30% | 重排增加已标注来源覆盖，但 MRR 略降；单次实验不建立通用排名 |
| [Agent 对照](agent-vs-baselines.md) | 9B 相对 Hybrid + Rerank 来源覆盖高 2.30 个百分点 | Agent 累积多轮 match/search/read，baseline 只取一次 top-10 chunks；不属于预算匹配的优劣结论 |
| [Qwen reranker 验证](reranker-integration.md) | 4 个真实集成测试通过；一个知识库问题验证三条链路 | 证明模板、候选、返回证据及实际接线，不能替代重排质量 benchmark |
| [thinking 诊断](../docs/agent-thinking-validation.md)、[停止修复](../docs/agent-search-stopping-validation.md) | 跨文档事实检查与重复搜索案例已做真实验证；当前 loop 已增加停滞提醒 | 新版本已有行为变化；历史模型消融只能代表当时版本，不能沿用为当前 release baseline |

还有两项容易混淆的统计口径：

- baseline 的 top_k 是 **chunk 数**。历史 Hybrid + Rerank 的 top-10 chunks 平均只有 **4.91 个不同 source**（范围 2–8），因此目前的 source R@10 并非“真正检索 10 篇文档”的召回率。需要同时记录 raw depth、unique-source depth 和 evidence token 数。
- 从 34 个 retrieval case 中排除 exact 与已给文件名的 direct read，剩下 22 个 semantic/exploratory/QA case，Hybrid + Rerank 与 9B 的来源覆盖分别为 **80.38% / 86.67%**。这是本次事后诊断切片，预算仍不同，不能当成预先注册的主实验胜负。

历史产物保留在 gitignored 的 `evaluation/results/`；机器结果能在本机审计，但单靠 clone 不保证可获得这些原始产物。今后应把正式 run bundle 放到有校验和的长期制品存储，仓库保留索引与 manifest。

## 2. 主流方法与数据集：采用什么，保留什么边界

以下资料均于 2026-09-10 检索，优先采用论文、作者仓库和官方工具。表中“用于 ARKB”是结合当前项目的建议，不是论文直接证明的效果。

| 方法 / 数据集 | 官方设计的重点 | 用于 ARKB 的方式 |
| --- | --- | --- |
| [BEIR](https://github.com/beir-cellar/beir)、[原论文](https://arxiv.org/abs/2104.08663) | 多领域零样本检索；统一 corpus / queries / qrels，nDCG、Recall、MAP 等 | 采用兼容数据接口；先跑 SciFact、NFCorpus，验证实现与领域迁移。不要用 embedding 总榜替代本项目评测 |
| [MS MARCO](https://microsoft.github.io/msmarco/)、[TREC DL](https://github.com/microsoft/msmarco/blob/master/TREC-Deep-Learning-2021.md) | 完整语料检索和固定候选重排分开；TREC 人工分级判断以 nDCG 为重点，稀疏 MS MARCO 标签另看 MRR | 借鉴候选召回与最终排序分离。稀疏 qrels 不能直接充当穷尽的相关来源清单 |
| [MIRACL](https://github.com/project-miracl/miracl)、[MTEB 中文任务](https://leaderboard.mteb.org/benchmark/MTEB%28cmn%2C%20v1%29) | 多语言/中文 embedding 与检索能力；MIRACL 有人工正负判断 | 选 zh→zh、en→en 作语言对照；MIRACL 多语言单语检索不能单独证明 zh→en 能力 |
| [MKQA](https://arxiv.org/abs/2007.15207) | 多语言、语言无关答案的开放域 QA，非一套现成的完整 passage qrels | 可借鉴跨语言问题对齐；引入时必须另定英文语料、证据标注/答案匹配口径，不能自动视为金标准检索标签 |
| [BRIGHT](https://brightbenchmark.github.io/) | 1,385 个需要较深入推理才能建立 query-document 关联的查询 | 用于困难语义/推理检索；按域评估，不能因英文通用 benchmark 高分就宣称本地 Agent 更好 |
| [Bright-Pro 作者仓库](https://github.com/yale-nlp/Bright-Pro) | 2026 年扩展；方面标注、静态与 Agent 搜索两类协议，并有固定轮数/自适应轮数对照 | 借鉴 aspect coverage 和受控搜索预算。属于较新补充，先固定数据与代码 revision 再使用 |
| [HotpotQA](https://aclanthology.org/D18-1259/)、[HoVer](https://aclanthology.org/2020.findings-emnlp.309/) | supporting facts 与跨多个文档的证据推理 | 借鉴句子证据和证据依赖；区分 full-corpus 检索与给定 distractor 的较小候选场景 |
| [TREC RAG 2025](https://pages.nist.gov/trec-browser/trec34/rag/overview/)、[2025 overview](https://arxiv.org/abs/2603.09891) | 检索、给定证据生成、端到端 RAG 分开；考察相关性、完整性与归因 | 采用按事实要点检查完整性、检查引用支撑的分层设计 |
| [TREC RAG 2026](https://trec-rag.github.io/) | 当前官网已转为 Agent-first，语料换成 ClimbMix；列出 Retrieval 与 RAG 任务，judgments 返回时间仍为 TBD | 跟踪新协议；近期优先采用已发布、可冻结的历史 judgments，不把 2025/2026 配置混用 |
| [ALCE](https://aclanthology.org/2023.emnlp-main.398/) | 同时评估回答正确性与引用质量 | 引用存在、引用能支持 claim、关键事实回答完整应分别评分 |
| [Ragas 指标](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)、[ARES](https://aclanthology.org/2024.naacl-long.20/) | Ragas 提供 context/faithfulness 等评分；ARES 用人工标注校准自动评估并做统计推断 | 先实现项目 rubric 与人工校准，框架只做可替换 judge adapter；导入框架不会自动得到可信标签 |
| [ir-measures](https://ir-measur.es/en/latest/getting-started.html)、[trec_eval](https://github.com/usnistgov/trec_eval/blob/main/m_ndcg_cut.c) | 标准排名指标、分 query 输出、qrels 与 run 格式 | 用作第二评分器交叉验证，固定 provider、版本、gain、cutoff、相关性阈值及 tie 规则 |

公开集的建议顺序是 **SciFact / NFCorpus → 一个 BRIGHT 域 → MIRACL zh/en 或合适中文 retrieval 子集**。这些只是外部有效性检查，主优化仍由独立 ARKB 场景决定。大规模 MS MARCO/TREC 全量暂放后面，避免先投入大量索引资源。

外部数据 adapter 应把官方 document ID 映射为安全的平面 Markdown 文件名，保留原 ID、title、正文、split、许可证、下载 revision/hash 与 qrels 映射。输出需聚合回官方的检索单位；若改变官方段落边界或裁剪语料，应显式标为自定义实验。可以抽样 query 降低成本，正式可比评估应保留对应完整 corpus；只留下正例和少量 distractor 会让任务变简单。公开训练集、已知模型训练暴露风险与未公开的项目盲测结果分别说明。

## 3. 数据优化：从来源列表升级为可审计证据

### 3.1 建立四类评测资产

| 集合 | 规模提案 | 用途与管理 |
| --- | --- | --- |
| v1 regression | 保留原 40 + 6 | 不改原标签与分数；已知失败可成为显式回归项，不能冒充 holdout |
| v2 pilot / dev | 先 60 个新的信息需求 | 校准 schema、标注指南、难度、成本；所有已看过的 pilot 留在 dev |
| v2 core | 目标 300 个独立信息需求，包含前述 60 个 pilot | dev 120 / validation 60 / test 120；先按意图与主题分组再切分 |
| challenge / external | challenge 起步 40 个；外部按数据集单列 | 版本变更、缺证据、近重复、桥接、长文、规模压力；不按实际用户频率解释其平均分 |

300 指 **独立信息需求族**，不是重复 trial，也不是把同一问题翻译三遍算成三个独立样本。同义改写、翻译、关联多跳题共用 `intent_family_id`，全部进入同一 split。初期这是能力覆盖集；没有用户流量证据时不声称代表生产任务概率。

拟定 core 配比：

| 主任务 | 独立信息需求数 | 要验证的能力 |
| --- | ---: | --- |
| semantic discovery | 60 | 不给标题/文件名的意图检索，术语与自然表达转换 |
| exploratory retrieval | 60 | 多方面材料覆盖，避免重复返回同一观点 |
| 单步 knowledge QA | 40 | 具体事实、适用条件、例外和来源 |
| multi-hop QA | 40 | 必须连接两处以上证据，包含桥接和比较 |
| exact lookup | 30 | literal/regex、大小写、零匹配、全量枚举、出现次数与来源数 |
| direct read | 20 | 文件/section/range 选择正确，所读片段确实支持问题 |
| 证据缺口或歧义 | 30 | 无答案、部分可答、需要澄清各约 10；分别标注预期行为 |
| no retrieval | 20 | 用户给足信息或明确无需知识库，直接完成任务 |
| 合计 | 300 | 配比先在 pilot 验证，再冻结 |

另记录交叉切片：query/corpus language、术语/口语、文档长度、证据位置、否定与数值、近重复、版本、难度、数据来源。先为 40 个信息需求编写经过人工核对的 zh/en 配对查询，测同语与跨语差值；不把翻译版当新增独立样本。中文语料需补充真实笔记后才有 zh→zh 结论。

### 3.2 语料不能只靠扩充问题

建议 pilot 在隔离语料目录中保留现有 example_notes，并增加 60–160 篇真实、结构不完全统一的技术笔记；core 再按实际可获得内容扩至约 300–1,000 篇。规模是预算提案，不是足够泛化的保证。数据来源优先：真实个人查询及对应笔记 → 人工独立编写场景 → LLM 辅助补充困难切片。来源比例实际记录；合成问题须人工验收，不能标成用户日志。

加入具有明确诊断目的的样本：

- 同主题但答非所问、旧版本但关键词更匹配、相同缩写不同含义、缺少限定条件的片段。
- 相邻段落才能组合的“规则 + 例外”、Markdown 表格的“表头 + 单元格”、多处重复文字。
- A 笔记只说明“遵循 B 规范”，关键规则仅在 B；答案实体不能直接出现在 query。
- 对同一事实替换为私有或随机值、移动证据位置、删除关键证据，检查输出是否随证据变化。这些作为独立的受控诊断，不冒充自然用户样本。
- t0 / t1 文档更新、删除、重命名、旧索引配实时 read 的受控状态集。静态排名测量仍固定一个状态，不能把 freshness 场景混入普通 baseline。

当前产品只接收平面 UTF-8 Markdown。先测 Markdown 表格与结构，不把“笔记讨论 PDF/多模态”写成“系统已具备 PDF/图像检索评估”。规模压力实验可从 1k / 10k 文档起步，等真实语料足够再扩大；随机复制文档不能代表真实增长。

### 3.3 标注内容

每个问题至少保存三层标签：

1. **Document relevance，0–3 级。** 0 不相关；1 同主题背景；2 直接支持一个所需要点；3 对核心要点有充分、直接证据。另存 `judgment_status=judged/unjudged`，未标注不伪装成人工判负。v2 Recall 的正例阈值先定为 ≥2，nDCG 保留完整 0–3；v1 的 >0 口径不改。
2. **Evidence / facets。** 文档 revision、正文 span、精确 quote/hash、支持哪个 facet，以及哪些证据可替代、哪些必须同时存在。以去除 H1 后的 `Note.content` 字符坐标为基础，保存解析规范；不能把原始 Markdown 偏移混进来。
3. **Task output rubric。** 对 QA 标关键事实、必需限定、可接受等价答案和缺口行为；对找资料标所需方面、来源列表及理由；对 exact 标正确文件集合；对 direct read 标目标段落中的信息。参考答案可辅助，但不应只有一段必须逐字匹配的范文。

示意结构如下，**不是已经实现的 v2 loader 输入**；缩写 ID 和范围要在标注后替换为真实值：

```json
{
  "schema_version": "arkb-eval-v2",
  "id": "qa_v2_example",
  "intent_family_id": "rerank_candidate_limit",
  "task_type": "knowledge_qa",
  "query": "为什么重排无法修复首轮缺失的证据？",
  "query_language": "zh",
  "corpus_id": "technical-notes-v2",
  "split": "dev",
  "answerability": "answerable",
  "evidence_requirements": [
    {
      "facet_id": "candidate_boundary",
      "critical": true,
      "weight": 1,
      "alternatives": [{"all_of": ["e14"]}, {"all_of": ["e40"]}]
    }
  ],
  "required_facts": ["重排只能调整已经进入候选集的证据顺序"],
  "query_origin": "human_authored",
  "annotation_status": "draft"
}
```

`e14` / `e40` 在独立 evidence 文件中定位真实 span。一个 facet 的 alternatives 是 OR，每个 alternative 的 all_of 是 AND；critical facets 之间也必须全部满足。这样既能表达替代证据，又能表达真正多跳的组合要求，不能把所有 evidence 简单并成一个 OR 集合。

例如 v1 `qa_003` 同时要求 14 和 40；本次阅读发现两篇都明确表达“候选之外的证据无法靠重排恢复”。它们可能是同一要点的替代证据，值得人工复核。此处是标签设计诊断，**不直接改 v1 gold，也不因此重写历史失败**。`qa_006` 的表头、位置映射等不同要点则可能确实需要组合，应逐项标注后再决定。

gold span 不使用 chunk_id 作为唯一身份，因为 chunking 实验本来就会改变 chunk。用稳定 doc ID + revision + body span 定义，再将不同 chunk 配置投影到同一 evidence。文本修订触发新语料与标签版本，旧偏移失效时必须报错。

### 3.4 标注与复核流程

1. 在看被测模型结果前写出信息需求、目标范围、answerability 和 rubric，记录作者与来源。生成 query 的模型不自动担任最终裁判。
2. 建 pool：BM25、dense、hybrid、rerank、多 query/Agent 轨迹各取固定深度；加入标注者独立找到的证据、强负例与少量 pool 外随机文档。不能只用现有冠军的候选构建 gold。
3. 隐藏系统名、分数和原排名，打乱候选；按 document→span→facet 标注。保存 pool 来源供审计，但不展示给标注者。
4. pilot 可先对 40 篇语料做全库判断；扩库后采用深度 pool。初步按每 query 去重后 30–60 篇规划，pool 实际大小和 pool 外补标新增正例率必须记录。
5. test 的 gold 证据/要点由第二名审阅者复核；document relevance 至少双标随机 20%，所有争议、负例和多跳关键证据复核。发布前完成最终系统所有实际返回 top-10 的判断，保存争议、仲裁理由及审阅覆盖率。
6. 报等级混淆矩阵、原始一致率、ordinal weighted κ 或 Krippendorff α、证据 span/facet 的分歧。pilot 暂以 weighted κ≥0.7 为检查点；不足则改指南并重标，不能靠删除难样本提高一致率。即使一致率高，也需检查共同偏差。
7. 冻结数据后发现未判断的潜在正例，记录为数据问题，盲化补标并发布新 qrels revision；所有比较系统在相同 revision 上重算。不能只给新模型补正例、旧模型继续用旧标签。

没有独立审阅人时先交付 `provisional` 数据，明确其局限；LLM 多次判断不等于双人标注。标注规模按 pilot 的真实耗时规划，不以快速批量生成 query 代替金标准建设。

### 3.5 防止开发集泄漏

v1 与 pilot 只用于开发。v2 按信息需求族与主题分组切分；共享事实、模板和近重复文档的组一起切。正常“固定知识库、新问题”测试允许不同 split 搜索同一个 corpus，不能把 test 正例文档从可搜索语料删掉；另设 held-out-topic / 新语料测试衡量泛化。

dev 用于调参；validation 用于选候选，限制尝试次数并留日志；test 只供最终比较，不根据逐题失败修改系统后再次宣称独立验证。详细失败一旦交给开发用于修复，下个 release 使用新的盲测版本或未见 reserve。held-out labels 与候选开发过程分离，被测检索 corpus 不含 qrels、参考答案和历史 traces。

## 4. 指标优化：每层只回答自己的问题

### 4.1 建议的主要指标

| 评测层 | 主指标 | 诊断 / 成本指标 | 定义与适用范围 |
| --- | --- | --- | --- |
| 语料 / 索引 | Gold evidence indexability | 丢失/过期文档率、span 校验失败、索引规模 | 被标必需证据是否真的保留在索引内容中；不计作检索质量 |
| 单次检索 | Evidence Recall@candidate depth、nDCG@10 | doc R@5/10/20、MRR@10、Judged@10、重复来源率 | 候选是否有足够证据；前排是否有用 |
| 冻结候选重排 | ΔnDCG@10、Δevidence coverage | 关键证据跌出 top-k、截断覆盖、重排延迟 | 同一候选集合上的前后变化；记录候选上限 |
| Context / read | Required-evidence coverage@token budget | full-evidence success、context 保留率、重复 token、关键限定丢失 | 只评模型实际可见文本；source 命中不等于完整证据 |
| Agent | Grounded task success@budget | 首次满足证据的时间/轮次、停止错误、无新增证据调用、错误类型 | 取证、任务输出、约束同时达标；按任务类型判定 |
| 答案 / 引用 | 必需事实完整率、claim 支持率 | answer correctness、citation correctness/completeness、无证据硬答率 | 判断最终输出，独立于 source recall |
| 运行成本 | p50 / p95 端到端延迟 | 原生 tokens、工具数、累计证据 tokens、模型加载、内存、错误率 | 明确 warm/cold、环境、分母；不把缺失值估成 0 |

核心看板不先引入任意加权“综合分”。对检索模型选择看 evidence recall、nDCG 和成本；对 Agent 产品选择看 grounded task success 和成本；其余指标用于解释变化。

### 4.2 排名指标统一契约

- 同时保留 `chunk@K` 与 `doc@K`。兼容旧数据的名称写成 `source_recall_from_top10_chunks`。若要真正 doc@10，先预定义 source 聚合、允许的 chunk 深度/补取策略与成本，再取 10 个 source；不能只改显示名称。
- MRR 必须标 cutoff：现有 `ranking_metrics` 的 MRR 读完整传入列表，并不按 `k` 截断；v2 的 MRR@10 要明确截断到 10。
- v1 nDCG 使用 `2^grade−1`。官方 trec_eval 的 `ndcg_cut` 默认直接使用 qrels grade 作为 gain，二者在 0/1 标签下一致，在分级标签下不同。[官方实现](https://github.com/usnistgov/trec_eval/blob/main/m_ndcg_cut.c)
- 保留 v1 原口径；v2 选择一种并写入 metric spec，例如 `ndcg_exp@10`，用 [ir-measures 的 gain/DCG 配置](https://ir-measur.es/en/latest/measures.html#ndcg) 对齐。外部 benchmark 采用其官方 metric recipe。禁止把不一致的实现都写成同一个 nDCG 数。
- Recall 的分母是已标相关项，不能称为“全世界真实相关项”。主表与 qrels revision、judgment coverage 一起展示；未判断项可按官方保守规则作 0 计算，但必须同时披露它们没有被判负。
- 初期正式比较要求实际返回 top-10 全部完成 judgment，并另报 returned_count 与 Judged@10；不足 10 个结果不能用短列表规避质量检查。缺口较大的初步结果标为 provisional。[Judged / Bpref 定义](https://ir-measur.es/en/latest/measures.html)
- 冻结 candidate list 评分前后需采用同一聚合方式；RRF、BM25、dense 的原始分数不直接比较。平分采用稳定 tie 规则，跨评分器输入保留同一排名。

### 4.3 证据与输出指标的具体计算

设任务有若干所需 facet。某 facet 的任一 alternative 所需 spans 全部被返回证据覆盖，才算满足；覆盖基于同 revision 的原始正文区间并集，重叠不重复计算。只覆盖一个 span 的部分字符单独报告，不自动当作事实完整。

`Evidence coverage = 已满足 facet 的权重之和 / 所需 facet 权重之和`。同时报告 `all_critical_facets_covered`，避免同一个主题多取几段就抵消另一个必需主题缺失。权重在看结果前冻结，默认等权。对宽泛找资料任务，这比要求找齐所有替代文档更贴合需求；复杂多样性排名可在标签成熟后补 α-nDCG，v2 首版无需堆叠所有指标。

现有 source group 和字符覆盖指标可复用，但 `source_group_recall` 只是 OR 来源命中，尚不实现上述多级逻辑。新指标要单独命名，不能误称旧代码已经支持全部需求。摘要/压缩文本若不是原文切片，区间不能证明其保留语义，须另做要点核对。

Agent 的 grounded task success 按任务分开：

- 找资料：最终交付来源集合/说明满足所需方面，证据充分且来源有效；轨迹中曾找到、最终没交付的来源只算过程指标。
- exact：最终文件集合与预期集合一致；零匹配是有效结果；全量枚举还检查完整性声明，不只看召回。
- direct read：确实读取支持问题的目标区间，输出满足 rubric；读取任意一段目标文件不算通过。
- QA / multi-hop：critical evidence 完整、关键事实正确、必需限定完整、引用能支持论断，并正常结束且不超预算。
- 无答案/部分可答：根据 snapshot 的证据正确说明缺口，对已有部分给出受支持回答；不强制“完全不搜索”。歧义任务允许并检查澄清。no_retrieval 则单独检查不必要检索和文本任务完成。

回答中的事实 claim 拆成可独立核对的单元；引用 correctness 评“所引用的实际可见证据能否支持此 claim”，引用 completeness 评“需外部证据的 claim 是否得到充分引用”。同时以 gold required facts 检查遗漏；只输出一句正确话或完全拒答不能凭高 faithfulness 通过完整性检查。工具返回的 source、最终引用 ID 和可访问 revision 都必须校验。[ALCE](https://aclanthology.org/2023.emnlp-main.398/)

### 4.4 错误与分母

当前 baseline 运行错误的排名分数为 null，主平均排除错误；当前 Agent 的成功率包含失败。历史 baseline 恰好没有错误，但未来有错误时两者直接比较会有偏差。

v2 保存两个视图：**条件质量**（成功执行且标签适用的请求）与 **交付效用**（所有计划执行且适用的任务，系统执行失败记 0 效用）。原始未知 recall 仍为 null，不伪装成已观测 0；效用指标明确是错误惩罚口径。报告 attempted / completed / failed / missing / not applicable，以及逐指标分母。

预先计划的任务因实验被中断而根本未执行，标 `run_incomplete`，不出正式胜负；不能仅统计跑完的容易任务。artifact 丢失、标签泄漏或语料漂移属于实验无效，与模型任务失败分开。不要重试到成功再删掉原记录；若要评 retry 策略，将它作为单独配置，完整记录成本。

## 5. 公平比较：让差异对应可解释的变量

### 5.1 三条独立实验线

| 实验线 | 固定项 | 变化项 | 主要决策 |
| --- | --- | --- | --- |
| Retrieval component | query、语料、labels、评测单位、K、检索模式、chunking | 一次只换 embedding 或某个检索参数 | 候选覆盖与排序是否改善 |
| Reranker component | 冻结 query + 完整候选池 + evidence 原文 | 是否重排、输入长度/截断策略等单因素 | 候选已存在时，重排是否帮助 |
| End-to-end task | 信息需求、snapshot、judge/rubric、预算、环境 | 固定 RAG、query rewrite、Agent 策略/模型 | 额外搜索与推理是否带来值得付出的完成率收益 |

Agent 模型可以改变 query、mode、read 和停止，这些属于处理效应，不能把最终差异直接归因给 embedding。先通过固定 query 的 component track 筛选 embedding，再固定 Agent 测对最终任务的影响。历史 27B 可作为高能力 reference 候选，但应在当前代码重建 baseline 后确认；4B 的产品路径仍需保留测量，不能只在接近饱和的 27B 上看 embedding 差异。

### 5.2 baseline 矩阵

首版保留现有四组，并逐项补：

- 原始 query 的 BM25、dense、hybrid、hybrid+rerank，适用于自然语言 retrieval 排名。
- zh/en 人工配对 query 对照，区分语言差值；翻译 BM25 是单独的 `translate→BM25` pipeline，记录 translator、prompt、输入输出和成本。
- 一次受控 query rewrite + hybrid，必要时再加固定数量的 multi-query fusion，用于判断收益是否来自改写或扩大预算，而非 Agent 的自适应决策。
- exact 路径：独立的 `pattern→match` 工具评测使用预先标明的 pattern；自然语言→pattern 的模型抽取另算 pipeline。direct read 用输入中给定的 selector。不能从 expected_sources 反推 pattern/文件名作为自然语言 baseline。
- 固定 RAG + 与被比较配置相同的最终生成模型。比较原生 Agent 产品路径时保留真实行为；要隔离取证质量时，将各策略的最终 evidence 交给同一个冻结 synthesis prompt/模型，另列为受控实验，避免把两种实验混为一谈。
- 诊断 upper bound：gold evidence→固定 generator；frozen candidates 的 oracle rerank；必要时全库上下文仅用于小语料诊断。所有 gold/oracle 只在 evaluator 使用，不作为可部署 baseline。

raw natural-language exact / direct-read 问题可以保留旧来源覆盖对照，但不混进“自然语言语义检索算法”主表。任务路由能力由单独 track 评估，固定检索器跳过 no_retrieval 不能算路由正确。

### 5.3 统一预算与可见证据

建议 pilot 试三个 evidence budget：2k / 4k / 8k tokens；查询型工具调用上限 1 / 3 / 6，并单列 read 次数。具体允许的组合在 pilot 后冻结，不直接运行所有笛卡尔积。当前代码没有完整的累计 evidence token / wall-clock 强制预算；实现前不能把仅有 `max_turns=8` 的运行称为预算匹配。

每次记录：原始候选数、每次返回 chunks、累计返回 tokens（包含重复）、去重后的证据 tokens、最大单轮上下文、全部模型 input/output tokens、总工具次数、完整 wall time。所有工具 schema、错误和提醒进入模型上下文的开销也应可追踪。使用固定参考 tokenizer 约束跨模型 evidence budget，另报各 provider 原生 token 用量，避免用不同 tokenization 假装预算相同。

总预算需要在线强制且不读取 gold；达到上限时的失败不能从质量均值删掉。离线“证据曲线在第 B token 的值”可以诊断，但不能当作强制预算下重新运行的真实任务成功率。相同上限不保证实际开销相等，因此报告质量—成本曲线和达到某个质量水平所需成本，不只看一个平均值。

ANN 另设 track：相同向量与 query vectors，对照 exact 和真实 Qdrant Server ANN；报 neighbor recall、任务证据损失与 latency。在只有 160 chunks 或 Qdrant Local 的情况下，不宣称得到生产 ANN 规模结论。

## 6. 让结果可信：冻结、校准、统计与复现

### 6.1 运行级冻结与追踪

每个正式实验 bundle 保存：

- dataset / qrels / evidence / rubric / split manifest 的版本与 hash；query-family 分组；完整执行矩阵与随机化 seed。
- 原始语料归档/hash、正文解析规范、SQLite schema、logical snapshot、Qdrant collection 与 vector hash、embedding tokenizer/template/维度/revision。
- Agent / embedding / reranker / judge 的准确制品 revision/digest、量化、运行参数、prompt/tool schema hash；实际 provider、依赖 lock、源码包/hash、硬件/线程数。
- raw query、所有候选与可见证据、逐次工具执行状态、最终输出、provider usage、分阶段时间、judge 输入/原始输出/解析状态。
- 逐 case score、聚合分母、配对差值、统计配置、原始制品校验清单、错误与无效原因。

使用隔离且冻结的 corpus 副本和与之匹配的新索引，match/read/search 都指向同一状态。当前 source_scope 绑定目录，复制后不能盲用旧数据库；按 schema v2 新建索引，记录快照和内容等价关系。纯检索固定 run-wide snapshot；Agent 在整个 run 中也保持这一约束。边界 hash 是补充检查，不代替冻结；发现漂移则 run 标为不可比较。

执行顺序采用分块随机化或轮换。单机大模型切换成本高，可以按 trial/block 轮换模型顺序，并在每块做独立 warmup；不要求无意义地每 query 卸载/加载。warm 与 cold 分开报告，warmup 不计质量样本，加载时间仍保留。缓存策略预先声明，不能用一个模型冷启动和另一个缓存命中推导速度倍数。

### 6.2 LLM judge 的可信边界

排名、exact 集合、span 覆盖、预算、执行错误优先确定性评分；事实支持与输出完整性才使用人工/LLM rubric。现有 citation review 接口可以扩展，主 Agent runner 需新增适配；不能直接拿固定 generation 的结构化引用格式去解析所有自由文本 Agent 回答。

先积累约 100 条人工判定的 query–answer–evidence 样本，覆盖模型、任务、语言和明显好/坏回答，划分 judge 调试与独立校准集合。量不够时只声明 pilot。judge 与被测生成模型尽量异构，但异构不能代替校准；本地模型不能因此自动视为可信。

评价 judge 的“支持/不支持”“完整/不完整”混淆矩阵，特别报告错误答案被判通过的 false-accept rate、召回率与不确定区间。对成对比较交换 A/B 顺序，隐藏模型名；检查长回答偏好、改写一致性与跨语言误判。研究已观察到位置、冗长与自我偏好等问题。[LLM-as-a-Judge 研究](https://arxiv.org/abs/2306.05685)

judge 必须输出判定所依赖的 evidence span/claim 对应及短理由；模型自报 confidence 不视为校准概率。解析失败、上下文超限与无法判断都保存为明确状态，并报告 coverage，不能反复抽样直到得到有利分数。对高影响结论、系统之间的胜负翻转、judge 分歧和随机样本做人工复核。

人工 gold 只在 evaluator/judge 可见；faithfulness 判定只允许用候选回答实际引用/看到的证据，不能用 gold 文本补足模型未取得的依据。answer correctness 可以另对照 gold，两者不合并。判分材料作为不可信数据处理，防止检索正文或候选回答中的指令控制 judge。

ARES 的启示是校准自动评分，而非“模型评分天然准确”；只有满足代表性抽样与方法假设时才使用其 PPI 等推断。随机人工审核用于总体校准，专挑失败的审核用于诊断，两类统计分开。[ARES](https://aclanthology.org/2024.naacl-long.20/)

### 6.3 统计报告与决策

**基本单位是独立 query family。** 对一个 family 内的变体先按预注册权重聚合，trial 先在同 case 内平均；再对 case/family 进行配对比较。不可把同题三个 trial 或二十个候选文档当成独立任务扩大样本量。报告 family 数、query 数、trial 数和可用配对数。

建议：

- 对 nDCG、evidence coverage、成功概率均值的差值做配对 cluster bootstrap（如 10,000 次，固定 seed），以 family 为重采样单位；相关文档/主题簇另做更保守的敏感性检查。
- 在均衡试验下先报告每题 trial 成功比例；若 trial 不等，不能按 trial 数给容易题更多权重。想估计运行随机性时做层级重采样，需说明假设。
- 单次配对二元结果可用 exact McNemar；连续指标可用配对置换测试。先确定一个主指标与少量主要比较，多个正式假设采用 Holm 等校正；诊断切片标为探索性。
- 报效应大小、95% CI、wins/ties/losses、逐题差值、任务/语言切片和成本；CI 包含 0 时写“证据不足”，不是“等价”。正式比较前确定最小有意义改善和非劣界限。
- 3 次 trial 可用于发现不稳定和运行错误，不保证估计了低概率故障。temperature=0 也不承诺未来确定性；多做全相同 trial 不能解决 query 多样性不足。

IR 显著性研究与配对推断方法可参考 [Urbano 等](https://arxiv.org/abs/1905.11096) 和 [NLP 统计检验指南](https://aclanthology.org/P18-1128.pdf)。具体 cluster 设计是针对本项目相关样本结构的选择。

**样本量应由可检测差异反推。** 120 个独立 test case 的单一二项比例，在简单随机假设和最不利 p≈0.5 时，粗略 95% 半宽约 9 个百分点；不能保证识别 1–2 点提升。pilot 用配对差值方差、family 相关性和所需最小效果做 power 模拟，再决定是否扩 test。300 是建设起点，不是统计可信的魔法数字。无错误的 30 个独立样本，其单侧 95% 错误率上界仍约 9.5%；若要在同一假设下支持错误率低于 1%，大约需要 299 个零错误独立样本。小切片零失败只能报告“本样本未观察到”。

### 6.4 可以发布的结论门槛

以下是拟议初值，需要在 pilot 后、看正式 test 结果前冻结：

| 门槛 | 标准 | 不满足时 |
| --- | --- | --- |
| 实验有效 | 输入/模型/代码可追溯，所有计划任务已结算，无标签泄漏与语料漂移，离线复算一致 | 标 invalid/incomplete，不发布冠军 |
| 标签可用 | gold span 全部校验；test 证据/rubric 经复核；各比较系统实际返回 top-10 判断齐全；争议有记录 | provisional，补标后全部系统同版重算 |
| 评分可信 | 关键确定性指标与参考实现一致；judge 有独立校准与覆盖率；关键争议人工裁定 | 语义分数标未验证，不自动用于发布决策 |
| 质量改善 | 例如 ΔnDCG 或 Δgrounded success 的配对 CI 下界 >0，点估计达到预设最小效果 | 证据不足；保留现状或扩大样本 |
| 成本改善 | 例如质量 CI 下界 >−1 个百分点，同时 warm p95 降低 ≥20% | 不称为同等质量的加速 |
| 关键场景 | exact 完整枚举、版本/否定/跨文档事实等已知回归不新增失败；切片变化与 CI 可见 | 定位后修复；不靠全局均分抵消 |

质量最小效果可先考虑 0.02 nDCG 或 3 个百分点 task success，但须根据真实成本价值与 power 调整，不承诺现有样本量能通过。以上门槛是实验决策协议，不是对生产可靠性的绝对保证。

## 7. 用评测定位优化方向

对每个失败记录：在哪个阶段丢失哪一个 critical facet、证据在哪个候选排名、后来是否被截断、实际可见文本是什么、最终错误是什么、相关错误与预算。允许多个原因同时存在，区分“观测事实”“待验证假设”“干预结果”。

| 观察到的症状 | 最小隔离实验 | 可指导的改动 |
| --- | --- | --- |
| gold evidence 不在索引文本中 | 对比原文→parsed body→chunk span | 文档解析、chunking、索引更新；先不换 embedding |
| exact 全找不齐且同篇占满 limit | 固定 corpus/pattern，按 occurrence、distinct source、分页深度对照 | match 完整性反馈、分页或 source 枚举契约 |
| 中文 query 差，英文等价 query 好 | 同 family zh/en 配对；翻译 BM25 单独计成本 | query 转换/跨语言 embedding，而非泛化地淘汰 lexical |
| candidate Recall 低 | K 扫描、固定 query 改写、dense/hybrid 对照 | 候选数、embedding、query 表达或融合 |
| 候选有证据，top-k 丢失 | 冻结候选重排；oracle 候选排序 | 重排输入长度、截断、排序或多样性选择 |
| 文档命中但 evidence span 未返回 | 对照 chunk/range/section 与 gold spans | chunk 边界、read 选择器、继续阅读 |
| evidence 已返回，生成仍错误 | gold context→同 generator；同 evidence→不同 synthesis | 输出提示、推理、引用和完整性约束 |
| 第一次搜索即足够，但重复至超限 | 当前停止策略与无提醒对照，检查实际新增 facet | 停止规则、预算、增量证据反馈 |
| source-ID / section 参数错误 | 冻结工具声明、请求和原始 selector，分离校验/执行状态 | 工具可用性、一致的身份表示与错误反馈 |
| 无证据仍给确定答案 | 删除关键证据/无答案对照，审查缺口行为 | evidence-aware 回答与澄清 |

优先研究的现有 case：`exact_001`、`qa_006`、`semantic_005`、`explore_002/004/006/007`。`exact_001` 的历史 occurrence-limit 机制证据较强；`qa_006` 应先分辨替代标签、候选缺失、排序、取段，不能先定性为 embedding 失效。停止问题已有修复，新的实验应检验修复后的收益和回归，而不是重新假设还处于旧状态。

对“充分证据后调用”指标重新命名为“达到标注阈值后的调用”。它不自动表示浪费：核实冲突、满足引用/最终输出要求可能仍有价值。只有在受控停止干预中，质量保持而成本下降，才能说该调用可省。

每轮从 dev 错误中选 1–2 个高影响假设，做单因素实验；validation 选候选，test 确认。先做 component screening，再测端到端，避免同时换模型、chunk、K、prompt 后无法归因。

## 8. 实施拆解与资源

### 8.1 代码与数据落点

保持现有 Runtime / RetrievalEngine 为执行主体，评测负责输入、观察、评分与报告。v2 不另写一个含隐藏策略的 Agent loop。

| 工作包 | 拟议落点 | 交付与验证 |
| --- | --- | --- |
| v2 contracts / dataset lint | 扩展 `evaluation/models.py`、`datasets.py`，必要时拆独立 evidence schema | 严格版本解析；unknown 字段、重复 ID、失效 span、split 泄漏、逻辑不可能全部拒绝；v1 原测试保留 |
| 标注资料 | `evaluation/data/v2/` 的 manifest、queries、qrels、evidence、rubrics、splits、dataset card | 先提交 dev/pilot；holdout labels 由独立评测位置管理；不把未复核合成标签标为 gold |
| 排名与证据评分 | 复用 `metrics.py` / `baselines.py`，新增按需模块 | 0/1 与分级 qrels、gain、K、tie、短列表、无答案、OR/AND 证据案例；与参考评分器对齐 |
| 运行与比较 | 扩展 `runs.py`，新增 pairwise statistics/report 模块 | 明确 attempted/executed/error、run-wide snapshot、随机顺序、预算和失败分母；回放不加载模型 |
| 最小可观测性 | 在 Runtime/provider/tool 边界附加 telemetry | 保留逐请求 usage、耗时、requested/executed/skipped/error 状态；不重构推理路径来维护第二套业务状态 |
| 输出语义评分 | `evaluation/judges.py` 或等价隔离 adapter | 版本化 rubric、自由文本/结构化回答适配、judge 校准、人工 review 导入与原始结果保留 |
| 外部 benchmark adapter | `evaluation/adapters/` | corpus/query/qrels ID 双向映射，官方单位与分数校验，小集 smoke + 全 corpus 正式运行 |
| 正式制品管理 | `evaluation/experiments/` 保存协议与制品 manifest，结果存长期 archive | 保存 hashes、检索地址和保留规则；clean environment 可验证/复算 |

以上路径是实施建议，除本次审计脚本和方案文档外尚未创建。新增依赖放 evaluation/dev extra，先确认项目 Python 3.13 的兼容性；不因外部评分库把 runtime 依赖无谓扩大。

### 8.2 分阶段交付

| 阶段 | 预计投入（非承诺） | 完成标准 |
| --- | --- | --- |
| P0：口径与当前基线 | 1–2 工程日 | 固定 metrics spec、比较矩阵、source/chunk 命名、当前 schema v2 隔离快照；四组固定 baseline 与当前默认 Agent 重测，旧/新分开 |
| P1：60-case pilot 与证据模型 | 2–4 工程日，另约 2–5 标注人日 | 完成 v2 parser/score、60 个新 dev 信息需求、标注指南、双审样本；实测 judgment 时间与 judge 校准可行性 |
| P2：预算、统计和输出质量 | 3–5 工程日 | 在线预算/telemetry、配对统计、gold-context 诊断、输出 rubric/judge、有效性 gate 可运行 |
| P3：core 与首轮受控实验 | 2–4 工程日，标注持续推进 | 扩至目标 300，冻结 split；按 power 决定 test 是否够；候选检索与停止各做一轮受控比较，给出继续/拒绝/证据不足结论 |
| P4：长期回归与外部有效性 | 首版闭环后按需 | 外部集合、scale/freshness suite、release 盲测轮换；每次优化有错误分类与效果证据 |

参考标注成本：若每题 pool 有 40 篇、每个 query-document 判断需 30–60 秒，300 题约为 **100–200 人时**，尚不含 query 编写、证据 span、rubric 和争议复核。复用已审阅文档可能降成本，长文/多跳可能增成本；必须用 pilot 修正。若只有一人兼职，先完成高质量 60-case pilot 与确定性诊断，不承诺 300-case 金标在一周内完成，也不把 provisional 包装成独立金标准。

计算成本同样先实测。历史平均每 trial 为 4B 14.25 秒、9B 21.84 秒、27B 66.86 秒；按旧分布粗估 300×3 次的串行耗时约 **3.6 / 5.5 / 16.7 小时**，不含 judge/新索引，且新语料和任务可能显著更慢。初期 component track 不调用 Agent；dev 筛选后只对少数候选做完整多 trial，27B 先跑预定义诊断子集。不能删失败样本来节省正式统计成本。

### 8.3 CI / 发布节奏

- PR：所有相关确定性测试、schema/标签 lint、历史评分 replay、最小本地 contract；无需每 PR 跑昂贵大模型全量。
- 计划执行的开发评测：冻结 dev、少量 trial、component 对照，记录每次调参；需要真实服务的检查独立标记。
- release candidate：独立 validation/test、预注册预算、完整错误保留、配对统计与关键人工核验，满足 gate 后发布结果。

当前 runner 的 completed 与任务全部成功是不同概念，CLI 可在任务失败时退出 0。实施时另加明确的 quality gate 返回码，不能拿“脚本执行完成”代替“评测通过”。本方案没有创建定时任务，也没有启动昂贵正式模型实验。

## 9. 每份正式报告的必需内容

报告首先回答“这次是否值得改变系统，以及证据有什么限制”，然后提供下表，而不是只输出成功率排行榜。

| 内容 | 要包含什么 |
| --- | --- |
| 实验问题 | 唯一主要假设、变化变量、预注册主指标和最小效果 |
| 有效性 | data/qrels/代码/模型/snapshot hash、完成矩阵、错误与缺失、judge 校准与判断覆盖 |
| 主结果 | baseline/candidate、独立 family 数、质量均值与配对 Δ、95% CI、成本、wins/ties/losses |
| 分组结果 | 任务/语言/长文/多跳/缺证据切片，不隐藏失败或小样本不确定性 |
| 归因 | candidate→rank→visible evidence→answer 各层变化；5–10 个代表失败及对应原始 evidence |
| 决策 | 采用、保留现状、继续取样或实验无效；说明支持结论的范围 |
| 下一步 | 优先级、可验证假设、下一次最小干预、预计收益/成本依据 |
| 复现 | 完整 run bundle、score replay 命令、metric spec、judge/rubric revision |

第一轮推荐落地顺序为：**P0 当前基线与评分契约 → 60-case pilot 的证据/缺口标注 → 预算与人工校准 → 检索候选和停止控制的受控实验 → 扩充盲测与外部基准。** 最终可信度来自可核验的数据来源、可复算的指标、受控比较和如实呈现的不确定性，而不是某一个评分框架或更多重复运行。

## 10. 本次交付与复算

本次新增本方案、只读审计脚本及其 JSON 输出，并在 evaluation README 加入入口。未修改被测算法、现有 v1 标签或历史结果。已有的 source/段落/引用指标都按实际能力区分，不声称 v2 已实现。

```sh
# 仓库根目录，输出到新的文件以便对比本次存档
.venv/bin/python evaluation/audits/20260910_review.py > /tmp/arkb-eval-audit-replay.json
.venv/bin/python -m pytest -q tests/evaluation -m 'not integration'
```

脚本要求本机保留三组正式 Agent 及固定 baseline 原始 artifacts；从缺少 gitignored 产物的新 clone 运行时会明确失败。JSON 记录被审计原始文件与当前 evaluator 的 hash。离线评分 replay 证明计算一致性，实时重测、独立人工 gold 与 v2 系统实施属于上述后续阶段。
