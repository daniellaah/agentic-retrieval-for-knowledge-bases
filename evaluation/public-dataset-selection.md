# ARKB 公开评估数据集选型

**2026-09-11 范围更新：当前仅维护英文检索。** 活跃公开集为 SciFact、Bright-Pro 两个技术域和 MuSiQue；中文专项已退出代码、测试与默认流程，数据集及已有成绩保留。下文是 P4 前的历史调研，其中文能力扩展建议不再作为当前待办。详见 [范围清理](../docs/english-scope-cleanup.md)。

调研日期：2026-09-10（America/Los_Angeles）。本文是 P4 前的选型建议，尚未下载完整数据、实现 adapter 或运行公开基准。规模与协议来自作者论文、仓库和数据卡；优先级、抽样规模与接入方案是结合 ARKB 的工程判断。

**结论：需要引入公开数据集。首轮选择 BEIR SciFact、Bright-Pro 技术域和 T2Retrieval；MuSiQue 作为多跳与缺证据专项。** 公开集提供外部问题、已有标注和更大的检索空间；ARKB 自己的人工 core 继续负责产品验收。两类结果分别报告。

**为什么现在需要**

当前 [P3 报告](p3-controlled-results.md) 的开发集只有 60 个问题、31 个待审需求族、58 篇文档，独立人工验收尚未完成。继续只在这些题上优化，无法区分“修好了几个熟悉的例子”和“改善了可迁移的能力”。公开集有三个直接用途：

| ARKB 的现有证据或缺口 | 公开集应补充什么 | 能支持的优化方向 |
| --- | --- | --- |
| P3 原始候选覆盖增加，最终返回覆盖不变；重排后来源排名与完整证据覆盖方向不一致 | 一个问题需要多个互补要点的标注 | 融合池大小、结果多样性、证据组合与重排选择 |
| 桥接文档被重排到返回范围之外；三个失败变体来自同一合成需求族 | 不同问题中的多跳依赖、支持段落与缺证据对照 | 子问题分解、桥接保留、回答充分性判断 |
| 现有中英文案例太少，语料形态和干扰项有限 | 独立中文查询与较大中文语料 | 中文分词、BM25/dense 互补、切块和 reranker 输入长度 |
| 已有指标参考实现，但还没有外部数据完整接入验证 | 小型、标准化 corpus / queries / qrels | ID 映射、官方检索单位、相关性阈值和评分一致性 |

公开标签可以直接用于其原生任务的外部评分，不必重做全量标注；接入转换及关键失败仍应抽查。它们不能自动验收 ARKB 的“精确找全所有出现位置、按文件名读取、实时笔记更新、无需检索、引用实际可见片段”等行为，也不能直接补成 300 个已审的项目需求族。

**选择哪些数据集**

| 选择 | 采用的具体版本和范围 | 推荐用途 | 主要边界与成本 |
| --- | --- | --- | --- |
| **BEIR SciFact：先接入** | BEIR `scifact` 的公开 test，300 个查询、5,183 篇摘要；保留全部语料 | 小型完整检索回归；nDCG@10、Recall@100；校验四种固定 baseline 的数据链路 | 科学论断检索与个人技术笔记存在领域差异，适合评分/回归检查，不能承担 ARKB 总体验收。[BEIR 数据表](https://github.com/beir-cellar/beir/wiki/Datasets-available) |
| **Bright-Pro：主要外部基准** | `yale-nlp/Bright-Pro`；先 `stackoverflow`：115 queries / 109,188 corpus units；再 `robotics`：101 / 63,920。两个域分别索引、分别评分 | 互补证据覆盖、困难语义检索、迭代搜索后的证据增量 | 英文技术/推理场景；标注对应官方分段后的检索单位。十万级文本需要先测本机索引时间与内存。[官方数据卡](https://huggingface.co/datasets/yale-nlp/Bright-Pro) |
| **T2Retrieval：中文检索基准** | `mteb/T2Retrieval` 当前 dev：22,812 queries / 118,605 passages；首轮固定抽取 500 queries，保留该版本全量语料 | zh→zh 的 nDCG@10、Recall@100；中文词法、语义、重排对照 | 首轮是明确命名的开发子集。该版本是加工后的 T2Ranking 子集，不等于原始 230 万段语料任务，也不能证明 zh→en 能力。[MTEB 数据卡](https://huggingface.co/datasets/mteb/T2Retrieval)、[原始 T2Ranking](https://github.com/THUIR/T2Ranking) |
| **MuSiQue-Ans / Full：专项诊断** | 官方 v1.0 dev；保留每题给定的 20 段上下文，从 2–4 hop 中分层选取约 100 个问题组，保留可回答/不可回答对应项 | 支持段落覆盖、answer/support F1、证据充分性与停止行为 | 属于给定候选上下文的任务；`Full` 指包含不可回答问题，不能解读为 full-corpus retrieval。隔离每题语料需要额外 runner 支持。[论文](https://aclanthology.org/2022.tacl-1.31/)、[官方数据与评分脚本](https://github.com/StonyBrookNLP/musique) |

表中的抽样数量是控制首次接入成本的提案，尚未做功效计算或抽样；不得把 500-query 子集分数写成完整 MTEB 成绩。静态检索可以在接入校验后扩到全部查询，Agent 则保留独立的预算与重复运行设计。

**为什么重点选 Bright-Pro**

Bright-Pro 的专家标注把信息需求拆成多个方面，并提供相应支持材料；论文同时研究静态检索和 Agent 搜索。与 ARKB 当前“找到几篇相关文档仍可能漏掉必要桥接/补充证据”的问题契合，因此比单纯增加一个 nDCG 榜单更有诊断价值。这是根据 P3 失败作出的适配判断，尚未证明 ARKB 在该集上的表现。[Bright-Pro 论文](https://aclanthology.org/2026.acl-long.1705/)

静态 track 同时报官方 α-nDCG、Aspect-Recall 和普通 nDCG/Recall，并记录 raw candidate → fusion pool → rerank → returned 的方面丢失。Agent track 首轮可采用作者固定子集在两个技术域的 25 + 25 个 query，比较相同预算下的单次检索与 ARKB。作者协议有固定搜索轮数和自主停止两种设置；若使用 ARKB 自己的模型、工具或预算，应报告为“ARKB on Bright-Pro”，不能称为复现论文的 Agent 结果。[作者评估仓库](https://github.com/yale-nlp/Bright-Pro)

这里的方面覆盖仍是公开检索标签代理指标：找到支持文档不等于相关片段已送入模型，更不等于答案已正确覆盖该方面。回答质量需独立评分；作者提供的 LLM judge 也需要在 ARKB 输出上抽样校准，不能直接当成人工结论。原始 BRIGHT 与 Bright-Pro 复用问题，不能把两个版本计为两批独立验证样本。

**MuSiQue 接入时必须保留的问题范围**

MuSiQue 通过连接单跳问题构造多跳依赖，Full 的不可回答项通过改变给定上下文使某一跳缺少答案。将不同题目的段落合并成一个知识库，可能重新引入缺失证据，破坏不可回答标签。因此第一版采用逐题隔离的候选语料，完整保留干扰段落与配对关系。它能诊断桥接和证据充分性，不能用于宣称十万篇知识库的检索效果。[论文的上下文构建与不可回答设计](https://aclanthology.org/2022.tacl-1.31.pdf)

运行时只提供问题与允许检索的正文，分解答案、支持段落标记和 answerability 留在评分侧。使用官方 answer/support F1；Full 另保留官方 group sufficiency 指标，同时按相同问题组聚合 ARKB 的结束状态和成本。确切配对规则、样本数量及数据 revision 在 adapter 阶段核验，不假设不同发布物数量完全相同。[官方评估说明](https://github.com/StonyBrookNLP/musique)

**有条件再引入的候选**

| 候选 | 何时值得引入 | 为什么不放在首轮必选范围 |
| --- | --- | --- |
| **MKQA 的 zh_cn→English 派生检索任务** | 明确测量中文提问检索英文笔记这一产品能力时 | 原始 MKQA 是 10,000 组、26 种语言对齐的问答，包含简体中文，但不提供一套穷尽的英文 passage qrels。需要另行固定英文语料与答案命中/证据标注规则；答案字符串出现不保证段落支持问题。[Apple 官方说明](https://github.com/apple/ml-mkqa) |
| **MIRACL-zh** | 需要更大规模中文检索验证，且已有索引资源预算时 | 中文 dev 393 queries，对应约 493 万 passages。它是各语言内部检索；同时跑 zh/en 仍不能替代 zh→en。首轮 T2Retrieval 更易控制成本。[官方仓库](https://github.com/project-miracl/miracl) |
| **MTRAG / MTRAG-UN 的 Cloud 技术文档域** | ARKB 接入用户会话历史，或明确建立保留历史的评估适配后 | Cloud 有 61,022 passages；MTRAG-UN 强调不可回答、欠明确及依赖历史的问题。很适合后续澄清/缺口诊断，但一轮请求内多次调用工具，与多轮用户对话是不同能力。[IBM 数据与协议](https://github.com/IBM/mt-rag-benchmark)、[人类对话格式](https://github.com/IBM/mt-rag-benchmark/blob/main/mtrag-human/README.md) |

跨语言是当前方案的明确剩余缺口：T2 和英文 Bright-Pro 都不能覆盖它。近期应先在 ARKB 人工 core 中建立同一需求的中英文配对查询、英语/中文证据和独立审阅；同一需求的翻译整组管理。需要外部可比结果时再增加冻结英文语料的 MKQA track。不能用自动翻译整个英文测试集后直接宣称取得中文官方成绩。

NFCorpus/FiQA 可以日后检验医学、金融领域迁移，但在当前技术知识库目标下边际价值较低。全量 BEIR/MTEB、MS MARCO、HotpotQA full-wiki 也不是首轮必需；先把上述少量集合测对，再决定是否需要更广的领域或规模覆盖。BEIR 中各任务语料大小与任务性质不同，不能用统一的集合数量代替覆盖分析。[BEIR 官方任务表](https://github.com/beir-cellar/beir)

**如何保证这些结果可信**

| 接入约束 | ARKB 的具体处理要求 |
| --- | --- |
| 冻结输入 | 保存数据与评分代码 revision、文件 hash、split、选中的 query IDs、语料数量、许可证和转换 manifest。本文中的在线卡片统计须在下载后复核；未下载就不声称已经冻结。 |
| 保留官方检索单位 | 官方 document/passage ID 映射为安全、无答案提示的平面 Markdown 文件名；原文、title 与 ID 映射可逆。不要把 Bright-Pro 的 query 相关路径编码直接暴露给 Agent。 |
| 检索排名可比 | ARKB 返回的 top-10 chunks 折叠后可能不足 10 篇官方文档。明确去重、聚合和补齐策略，评分聚合回官方单位；同时保留产品原生 chunk 指标，分别命名。 |
| 标签语义正确 | 当前 [v2 metric spec](metric-spec.json) 以 relevance≥2 为正例，外部二元 qrels 常以 1 为正例。为外部 track 独立配置阈值、gain、cutoff 和官方参考 scorer；不能原样套入 v2。 |
| 不因抽样降低检索难度 | SciFact、Bright-Pro 每域、T2 所选版本保留完整 corpus；可预先抽 query，不能依照该批 query 的 qrels 只留下正例与少量干扰项。MuSiQue 则忠实保留其逐题候选上下文，单独命名 track。 |
| gold 不进入运行路径 | 参考答案、方面内容、支持文档关系、answerability 和判定 rubric 仅供评分或专门标明的 oracle 对照；禁止进入普通检索索引、Agent 提示及线上停止决策。 |
| 未标注不等于已确认负例 | 官方分数按官方处理方式计算；诊断侧保留未判断状态、judged coverage 和新增候选抽查。不能为提高分数而事后把结果从官方排名中删掉。 |
| 文档相关性不冒充答案质量 | 公开 qrels/方面关联不能自动变成 v2 的精确字符 span。若需要“模型看到了完整证据”的判断，需额外定位并核验片段；未核验则仅报告文档/方面覆盖代理。 |
| 配对比较与暴露管理 | 每个数据集单独报告差值、区间、失败分布。翻译、MuSiQue 配对及重复 trial 不能增加独立样本量；公开数据按外部开发/回归资产管理，不默认视为当前模型从未见过的盲测集。 |
| 成本与质量一起看 | 固定模型、语料、候选、工具和证据预算；报真实错误、max_turns、token 与延迟。更早停止不能单独证明更好；外部 judge 未校准时把答案质量标为待验证。 |

外部数据 adapter 与原生 v2 core 保持各自标签语义，共享运行归档、轨迹和配对统计能力即可。发布结果采用三块并列证据：**ARKB 人工 core、外部公开任务、scale/freshness 等确定性产品回归**，不合成一个“ARKB 总分”。

**许可证与来源记录**

这是已核对的来源说明，接入时仍需随实际下载 revision 保存许可证文本和归属信息，避免把代码许可证当作全部语料许可证。

| 数据 | 当前官方说明 | 入库记录 |
| --- | --- | --- |
| SciFact | claims/evidence 为 CC BY 4.0；摘要为 S2ORC 的 ODC-By 1.0；代码 Apache 2.0。[上游 LICENSE](https://github.com/allenai/scifact/blob/master/LICENSE.md) | 分开记录标注与语料的许可；BEIR 格式转换不改变原始归属。 |
| Bright-Pro | 扩展发布为 MIT；底层 StackExchange 问题及 BRIGHT 语料保留原许可证。[数据卡许可说明](https://huggingface.co/datasets/yale-nlp/Bright-Pro#license) | 保留扩展和上游双层来源；不能笼统把底层网页全部记成 MIT。 |
| T2Retrieval / T2Ranking | 所选 MTEB 数据卡与 T2Ranking 上游声明 Apache 2.0。[数据卡](https://huggingface.co/datasets/mteb/T2Retrieval)、[作者仓库](https://github.com/THUIR/T2Ranking) | 记录加工版本及原始来源；禁止混用子集规模与全量任务成绩。 |
| MuSiQue | 数据 CC BY 4.0；作者还披露了与种子单跳数据的潜在训练/测试重叠。[发布说明](https://github.com/StonyBrookNLP/musique) | 保存种子来源与重叠提示，配对变体整体管理。 |

**建议的落地顺序和交付物**

1. **SciFact 校验链路。** 全量小语料接入；验证 ID/正文/标签转换，导出标准 run；与官方评分实现逐题比对，并检查空结果、二元 qrels、重复 chunk 等边界。只用于确定接入正确，不据此选产品冠军。
2. **Bright-Pro 技术域建立主要外部结果。** 先 Stack Overflow，再 Robotics；BM25、dense、hybrid、hybrid+rerank 使用同一语料。分别追踪候选、融合、重排、交付的方面覆盖，Agent 在预先固定的小样本和预算下单列运行。
3. **T2Retrieval 建立中文对照。** 固定 500-query 开发子集和完整加工版语料，核查中文分词、长度截断和 hybrid 增益；成本允许后扩全量 query，再产生完整任务结果。
4. **MuSiQue 专项验证下一轮假设。** 保留每题候选范围及对应问题组，检验桥接保留、缺证据时结束、回答与支持是否一致。使用领域泛化结果辅助决定策略，仍需回到 ARKB core 检查产品收益。

每一步交付数据卡/许可 manifest、固定协议、逐题结果、错误分类和可重算 run bundle。尚未实测十万级语料的本机索引成本，不承诺耗时或金额；先测代表性正文的吞吐、切块膨胀和磁盘/内存，再安排全量运行。公开集的调研和接入可以推进，但 [P3 的人工 core 与未见质量验收](implementation-progress.md) 仍独立保留为未完成条件。
