# P4：公开评估与长期回归

**P4 工程与本轮公开开发实验已完成（2026-09-11）。** 四套全语料 BM25 共 1,016 条、四套索引检索共 4,064 条，以及 Bright 50 题和 MuSiQue 200 个变体的 Agent 运行，均完成独立复算和归档验证。官方指标对照、规模/更新回归、离线 CI 和轮换保护已接入；两次未完成的模型尝试保留。P3 人工 core、250 份输出的独立人工审阅和真实发布盲测仍未完成，没有推广新产品配置。

本轮最明确的方向是中文词法切分、技术长查询的重排输入与同分排序，以及 Agent 的工具契约、结束策略和结构化输出。MuSiQue 严格成绩受格式不合规显著影响，正文中同时保留原分数与事后格式敏感性诊断。[完整结果矩阵](experiments/p4-external-20260910-summary.json)、[决策日志](experiments/p4-decision-ledger.json)、[最终制品清单](experiments/p4-artifacts-20260910-manifest.json)

## 已冻结的数据范围

| 集合 | 完整语料单位 | 本轮查询 | 口径 |
| --- | ---: | ---: | --- |
| BEIR SciFact | 5,183 | 300 | 官方公开 test |
| Bright-Pro Stack Overflow | 109,188 | 115 | 独立域；382 个方面 |
| Bright-Pro Robotics | 63,920 | 101 | 独立域；375 个方面 |
| MTEB T2Retrieval | 118,605 | 500 | dev 中预先按 seed=20260910 抽样，保留该加工版本完整 corpus |
| MuSiQue-Full | 每个 variant 保留原始 20 段 | 100 组 / 200 variants | 2/3/4-hop 分别 34/33/33 组，逐题隔离；不是全量官方 dev 成绩 |

下载确认 MuSiQue-Full dev 有 4,834 个样本、2,417 个完整配对。本轮为诊断目的提高了 3/4-hop 的比例，不能把本轮均值当成自然查询分布或完整官方任务均值。所有外部数据保持 public-development 身份，不写入项目的已审 core 配额。

原始数据的 revision、下载 URL、hash 和许可说明保存在 [下载清单](results/p4-inputs/downloads.json)；各规范化数据包保存全部原始 ID/title/text、选中 query IDs、qrels、方面和转换 manifest。原始标题和正文进入隔离 Markdown 目录；答案、方面内容、支持关系与 answerability 均留在评分侧。公开集选型理由见 [选型报告](public-dataset-selection.md)。

[原始输入对照](audits/20260910-p4-input-integrity.json) 核对了下载锁中的 32 个文件、296,896 个 corpus unit 的完整 ID/title/text、1,016 个选中 query 及其 qrels，以及 MuSiQue 的 200 个原始变体。没有发现重复原始 query ID、corpus ID 或被转换丢失的正文。这证明转换完整性，不证明作者标注绝无错误，也不能证明公开数据未进入模型训练。

从同一批锁定原始文件重新运行转换器，21 个规范化数据/manifest/选择文件均与首次输出逐字节相同。[转换复现](audits/20260910-p4-conversion-replay.json)

**公开标注也存在需要审阅的边界。** Stack Overflow 的 109,188 个 ID 对应 68,375 种不同的规范化 title/body，Robotics 的 63,920 个 ID 对应 42,382 种；分别有 9 和 4 个 query 出现“正例正文完全相同的另一 ID 未标为正例”。抽查包含 pandas/R/TeleBot 与 ROS/Gazebo 参考文档的重复片段。官方分数继续按原 ID/qrels 计算；这些情况作为标注别名和重复证据的独立诊断，后续审阅与去重对照另行版本化。SciFact/T2 未发现这种完整 title/body 重复。[重复文本审计](audits/20260910-p4-text-duplicates.json)

## 已完成的原生单位 BM25 基线

直接复用生产 `BM25Retriever`，每个公开 corpus unit 包装为一个完整 Note/Chunk；没有改分词或用 qrels 筛减 corpus。四套数据共 1,016 个查询，全部保留，运行错误 0。

| 集合 | nDCG@10 | Recall@100 | Weighted Aspect-Recall@10 | 空排名 |
| --- | ---: | ---: | ---: | ---: |
| SciFact | 0.6617 | 0.8859 | 不适用 | 0/300 |
| Bright-Pro Stack Overflow | 0.3444 | 0.6535 | 0.4357 | 0/115 |
| Bright-Pro Robotics | 0.2700 | 0.5902 | 0.3536 | 0/101 |
| T2Retrieval，500-query dev 子集 | 0.0290 | 0.0209 | 不适用 | 457/500 |

这些是不同任务上的开发诊断，不合并总分。后续索引 track 使用生产的 512-token 切块和 64-token overlap，不能把这张原生完整单位表与切块检索的差异全部归因于模型。

四套原生 BM25 另从完整 corpus 使用各自冻结实现重新检索，全部 1,016 个排名一致；这层检查直接重算检索过程。[全语料排名复现](audits/20260910-p4-native-retrieval-replay.json)

**已经可以确认的优化方向：中文词法切分。** 500 个查询中 469 个只有一个词项，464 个与所有已标注正例均无词项交集。当前 `nfc-casefold-word-v1` 使用连续 Unicode word token，不对中文句子做切分。这解释了大量空结果，支持下一轮独立的中文 tokenizer 对照；尚未实现或测量替代 tokenizer 的收益。逐 query 证据见 [分词审计](results/p4-regression/chinese-tokenization.json)。

## 索引与 Agent 实验协议

SciFact 索引 track 已完成 300×4=1,200 条记录，无运行错误，官方评分交叉校验最大误差 1.12×10⁻¹⁶。

| SciFact 配置 | nDCG@10 | Recall@10 | Recall@100 |
| --- | ---: | ---: | ---: |
| BM25，512-token chunks 聚合 | 0.6608 | 0.7838 | 0.8859 |
| Semantic | 0.6961 | 0.8382 | 0.9467 |
| Hybrid | 0.7216 | 0.8546 | 0.9633 |
| Hybrid + rerank20 | 0.7666 | 0.8859 | 0.9633 |

在该集和该配置下，重排相对 hybrid 的 nDCG@10 差值为 +0.0449，按 query 配对 bootstrap 的探索性 95% CI 为 [+0.0229, +0.0690]，62 胜 / 206 平 / 32 负。区间未做多重比较校正，公开 query 不等于已确认独立的 ARKB 需求族；该结果不能直接批准产品默认配置变更。[SciFact 完整结果](results/p4-scifact-indexed/summary.json)

Stack Overflow 完成 115×4=460 条记录，无运行错误；109,188 个源文档产生 246,224 个 chunks，其中 122,082 种不同 embedding 输入。完整索引准备耗时约 69.5 分钟，运行前后索引快照与重排权重校验通过。[独立复算](audits/20260910-p4-bright-stackoverflow-replay.json)

| Stack Overflow 配置 | nDCG@10 | Recall@100 | Alpha-nDCG@10 | Weighted Aspect-Recall@10 |
| --- | ---: | ---: | ---: | ---: |
| BM25，chunks 聚合 | 0.3360 | 0.6504 | 0.3499 | 0.4437 |
| Semantic | 0.3434 | 0.7603 | 0.3498 | 0.4417 |
| Hybrid | 0.4807 | 0.7934 | 0.4891 | 0.6110 |
| Hybrid + rerank20 | 0.3412 | 0.7934 | 0.3482 | 0.4875 |

**当前重排配置在技术长查询上存在明显退化。** 相对 hybrid，nDCG@10 差值 −0.1394，探索性 95% CI [−0.1855, −0.0935]，26 胜 / 17 平 / 72 负；方面召回差值 −0.1235，CI [−0.1750, −0.0720]。这与 SciFact 的正向结果方向相反，不能据单一短查询数据集推广当前 rerank 配置。[完整结果](results/p4-bright-stackoverflow-indexed/summary.json)

Robotics 也完成 101×4=404 条记录，无运行错误，完整索引 80,135 个 chunks、57,555 种不同 embedding 输入，准备耗时约 25.6 分钟。全部排名重组和 2,020 段重排正文通过独立核验。[复算](audits/20260910-p4-bright-robotics-replay.json)

| Robotics 配置 | nDCG@10 | Recall@100 | Alpha-nDCG@10 | Weighted Aspect-Recall@10 |
| --- | ---: | ---: | ---: | ---: |
| BM25，chunks 聚合 | 0.2480 | 0.5763 | 0.2644 | 0.3192 |
| Semantic | 0.3140 | 0.5643 | 0.3397 | 0.3828 |
| Hybrid | 0.3904 | 0.6492 | 0.4158 | 0.4855 |
| Hybrid + rerank20 | 0.2919 | 0.6492 | 0.3206 | 0.4373 |

Robotics 的重排差值为 −0.0986，探索性 95% CI [−0.1394, −0.0598]，27 胜 / 11 平 / 63 负；方面覆盖差值 −0.0482。两个技术域均未支持推广当前重排配置。[完整结果](results/p4-bright-robotics-indexed/summary.json)

T2 完成 500×4=2,000 条记录，无运行错误；完整 118,605 文档产生 212,659 个 chunks、212,165 种不同 embedding 输入，索引准备耗时约 186.6 分钟。排名重组、10,000 段重排正文及参考指标均已核验，最大数值差为 3.33×10⁻¹⁶。[独立复算](audits/20260910-p4-t2-replay.json)

| T2，固定 500-query dev 子集 | nDCG@10 | Recall@10 | Recall@100 | 空排名 |
| --- | ---: | ---: | ---: | ---: |
| BM25，chunks 聚合 | 0.0284 | 0.0197 | 0.0209 | 457/500 |
| Semantic | 0.8346 | 0.8190 | 0.9334 | 0/500 |
| Hybrid | 0.8272 | 0.8165 | 0.9344 | 0/500 |
| Hybrid + rerank20 | 0.8249 | 0.8171 | 0.9344 | 0/500 |

**T2 支持修正中文词法路径，但未支持当前重排带来明确增益。** 重排的 nDCG@10 差值为 −0.0023，探索性 95% CI [−0.0161, +0.0123]，124 胜 / 215 平 / 161 负；重排阶段平均额外消耗约 4.57 秒。Semantic 的本轮均值高于 hybrid，但这里没有声称它在未见数据上稳定占优。两路候选并集的已知正例召回与 semantic 单路相同，均为 0.9599；当前 BM25 没有向并集增加已标注正例。该结论仅针对本轮公开开发子集及当前 tokenizer。[完整结果](results/p4-t2-indexed/summary.json)、[分阶段来源诊断](results/p4-regression/t2-rerank-truncation/stage-attribution.json)

T2 的 10,000 个重排输入中，3,896 个截断，涉及 495/500 题，没有完全丢失正文的输入。同分规则对照发现 12 题含同分组、3 个完整排名变化，但全部汇总指标不变；不能把 Bright 上的同分问题直接推广成所有任务的主要瓶颈。[输入截断](results/p4-regression/t2-rerank-truncation/summary.json)、[同分对照](audits/20260910-p4-t2-tie-policy-summary.json)

索引 track 复用 `Runtime.index`、BM25、semantic、生产 RRF 和固定 Qwen reranker。每路取得 500 个 chunks；按排名首次出现折叠到原始 corpus ID，最多保留 100 个单位。重排处理 hybrid 最前 20 个不同 source 的首个 chunk，再接回原第 21–100 位；因此 Recall@100 在这个重排对照中按设计不变。nDCG@10 和方面覆盖才用于观察前部排序变化。该配置是明确记录的 benchmark 配置，产品默认值保持不变。

每个样本保存原始候选 ID/分数/字符范围、重排输入正文与分数、最终排名和逐项指标。延迟按实际执行阶段记录，共享的检索结果不伪造四次独立端到端延迟；本机耗时是资源规划信息，不用于宣称生产延迟优势。

SciFact 的 6,000 个实际重排输入中，1,237 个（20.6%）发生右侧截断，涉及 272/300 个查询；没有完全丢失正文的候选。512-token 总上限扣除固定模板的 48 tokens 后，查询、标题和正文共用 464 tokens。这是实际输入丢失量，尚不能推导改变长度的因果收益；后续应固定候选池比较输入长度/选段策略。[截断审计](results/p4-regression/scifact-rerank-truncation/summary.json)

Stack Overflow 的 2,300 个实际重排输入中，1,582 个被截断，600 个完全未保留正文；30/115 个 query 的全部 20 个候选均无正文。该 30 题的平均 nDCG 变化为 −0.2210，其余 85 题也下降 −0.1106，因此截断与退化同时存在，但不能单凭分层证明全部损失由截断造成。Robotics 的 2,020 个输入中，1,140 个被截断，520 个未保留正文，对应 26/101 个 query。[Stack Overflow 截断](results/p4-regression/bright-stackoverflow-rerank-truncation/summary.json)、[Robotics 截断](results/p4-regression/bright-robotics-rerank-truncation/summary.json)

Stack Overflow 两路完整候选并集的已知正例召回为 0.9409，hybrid 前 100 为 0.7934，重排池前 20 为 0.6259，hybrid 前 10 为 0.5252，重排后前 10 为 0.4160。49 题在融合前 100 时丢失至少一个并集正例，50 题的前 10 召回在重排后下降；这些类别可重叠，不能相加当成互斥错误率。候选数量减少本来就会降低召回，需要在固定候选池上验证融合和排序改动。[逐题阶段分析](results/p4-regression/bright-stackoverflow-rerank-truncation/stage-attribution.json)

**事后离线对照进一步定位到同分排序规则。** 上述 30 题各自的 20 个有效输入 token 序列完全相同，保存的 20 个模型分数也完全相同；当前 `Reranker` 用文档身份打破同分，因而覆盖了 hybrid 顺序。全域有 67 题存在同分组；仅对已保存的分数改成同分保留原候选顺序，46 题的排名发生变化，nDCG@10 从 0.3412 提高到 0.3989（+0.0576），方面覆盖从 0.4875 提高到 0.5326。它仍低于 hybrid，无法解释或消除全部退化。这个对照没有重新调用模型，属于看过结果后的开发诊断，未替换主表分数、未改变产品规则，也未经过未见数据验证。[同分规则复算](audits/20260910-p4-stackoverflow-tie-policy-summary.json)

Robotics 的同类对照发现 26 个全体输入/分数相同的 query，nDCG@10 从 0.2919 回升到 0.3490（+0.0571），方面覆盖从 0.4373 回升到 0.4812；仍未达到 hybrid 的排序分数。SciFact 没有候选分数相同的查询，全部排名保持不变。[Robotics 同分对照](audits/20260910-p4-robotics-tie-policy-summary.json)、[SciFact 对照](audits/20260910-p4-scifact-tie-policy-summary.json)

Bright-Pro Agent 采用作者固定子集中两个技术域各 25 题；MuSiQue 使用上述 100 个配对组。复用当前 4B Agent、8 轮上限、12 次工具/10 次 query/6 次 read、4,000 evidence tokens 和 120 秒协作式截止。每个 case 一个 trial，用于首轮外部诊断，不足以声明随机稳定性。MuSiQue 追加固定 JSON 输出要求，明确答案、可回答性和支持文件；它是 ARKB 工具检索协议，不是论文原始 reader 的复现。

**Agent 协议修订 v2。** 首次 Stack Overflow 尝试在第一个 query 的 exact match 中运行超过 17 分钟，尚无完成行。中断 traceback 确认它正在逐文档启动 `rg`；该未完成尝试完整保留，不能报作 25 题完成或筛掉后当作成功。v2 在评估层对每次 `run_agent` 增加 180 秒 POSIX 硬超时，产品 120 秒协作预算保持原样；超时按执行错误保留部分轨迹和实际耗时。硬超时不涵盖事先索引准备，不能保证取消服务端已接收的推理，部分原生调用也可能延迟信号处理。两个 Bright 域和 MuSiQue 均采用这组预算与超时设置；MuSiQue 的后续 v3 隔离修订见下文。题目、选择集合、提示词及生产检索逻辑保持不变。[原尝试及修订记录](audits/20260910-p4-agent-v1-interruption.json)

后续 v2.1 instrumentation 补记 MuSiQue 每题上下文建索引成本，并使计时异常避开 HTTP 库的网络异常转换。180 秒限制和实验内容保持不变；已完成的 Stack Overflow v2 只有工具超时，没有模型请求错误，已有分数未重分类。[修订日志](experiments/p4-decision-ledger.json)

**MuSiQue 隔离修订 v3。** v2 完成第一个变体后，把第二个目录交给同一个 vault 数据库，生产的 source-scope 保护正确拒绝，批次因此失败；只有 1 个完成变体、0 个完整配对。v3 为每个变体新建独立 SQLite 数据库，不跨变体共享 embedding 缓存，生产保护保持原样。两个合成上下文用真实 Ollama/SQLite/Qdrant 的 6 项隔离检查通过后，重新运行全部 200 个变体。原始第一题存在一次额外曝光，旧结果保留，不在旧、新输出之间挑选较好值；题目、source 身份、提示词、模型、预算及生产源码均未改变。[失败记录](audits/20260910-p4-musique-v2-incomplete.json)、[实际隔离检查](audits/20260910-p4-musique-context-isolation.json)

不含评测题目的本机配置探针返回 OK，4B 的实际加载 `context_length` 为 262,144；它与重排器的 512-token 上限分别记录。正式 Agent 运行会逐 attempt 保留加载模型信息，并在前后核对模型定义。四套检索 query 和 MuSiQue 的全部 4,000 个段落输入均通过 embedding 的 8,192-token 预检查。[配置探针](audits/20260910-p4-chat-context-probe.json)、[输入长度检查](audits/20260910-p4-token-preflight.json)

MuSiQue 无效输出/执行失败使用 null answerability，不能自动获得“不可回答”的正确判断分。回答 F1、支持 F1、配对充分性由独立实现与官方脚本交叉验证。Bright-Pro 的文档/方面命中仅作为检索代理，回答语义质量仍等待独立审阅，不自动采用未校准的 judge 分数。

Stack Overflow Agent v2 的 25 题已全部尝试：17 个 final、5 个 evidence-budget 停止、2 个 max_turns、1 个 exact 工具硬超时。离线复算核对了 79 次模型请求、62 个工具事件及 217 段原文，全部结果包含轨迹。[轨迹复算](audits/20260910-p4-bright-stackoverflow-agent-replay.json)

| Stack Overflow Agent，固定 25 题 | 工具返回 | 加入对话 | 提交模型请求 |
| --- | ---: | ---: | ---: |
| 平均不同文档数 | 5.24 | 4.96 | 4.96 |
| 已知正例文档召回 | 0.2943 | 0.2877 | 0.2877 |
| 加权方面覆盖代理 | 0.3703 | 0.3537 | 0.3537 |

6 个 final 没有提交任何正文证据。上述 68% final 比例只是运行完成比例；提交正文也不证明回答正确或支持完整。这里按该 25 题报告全部工具次序中的不同文档，并非固定 top-10 排名，也不能与 115 题检索均值直接作同预算优劣比较。人工输出质量审阅为 0/25。[审阅包与诊断](reviews/p4-bright-stackoverflow-20260910/README.md)、[汇总](reviews/p4-bright-stackoverflow-20260910/summary.json)

该 25 题的 Agent 阶段累计 636.2 秒，p50 17.9 秒、p95 37.0 秒，最长 180.1 秒；另有约 20.3 秒共享 engine 准备，既有 corpus 索引成本在前文单列。39 次 search 中 30 次显式 hybrid、9 次使用默认 semantic。22 次 document-ID read 平均 2.46 秒，源码当前通过枚举文件路径解析 ID，值得另做读取规模剖析，同时保留 live edit/delete/rename 契约。上述耗时为本机描述性观察，不是受控生产 SLA。[完整成本](audits/20260910-p4-bright-stackoverflow-agent-costs.json)

Robotics Agent 的 25 题也已全部尝试：8 个 final、8 个 evidence-budget 停止、4 个 max_turns、5 个 error。核验了 133 次模型请求、124 个工具事件、480 段原文，没有缺失错误轨迹。提交证据的平均不同文档数为 10.84，已知正例召回 0.2535，加权方面覆盖代理 0.3385；没有无提交正文的 final。32% 的 final 比例同样不是回答准确率。[轨迹复算](audits/20260910-p4-bright-robotics-agent-replay.json)、[审阅包](reviews/p4-bright-robotics-20260910/README.md)

Robotics 的 5 个错误中，1 个是模型请求的 180 秒硬超时；4 个是读取参数错误：两次对 55/104 字符正文请求读取到第 1,000 字符，一次将不同搜索结果的 document ID 与 section ID 混用，一次 section ID 格式错误。工具均正确拒绝，当前 Agent 则因此终止。这支持另行验证更清晰的读取范围接口、成对绑定的证据引用和受预算约束的错误恢复，而不是把这些失败全部归因于检索模型。[参数绑定审计](audits/20260910-p4-robotics-tool-errors.json)

Robotics Agent 阶段累计 695.1 秒，p50 22.6 秒、p95 33.0 秒，最长 180.0 秒；共享准备约 6.4 秒。一次请求超时导致 133 次请求中只有 132 次有 provider usage，完整 token 总量因此保留 null，不能把未知消耗当成 0。[成本记录](audits/20260910-p4-bright-robotics-agent-costs.json)

## MuSiQue 配对结果与输出格式诊断

v3 的完整 100 组 / 200 个变体全部保留：49 个 final、82 个 max_turns、62 个 evidence-budget 停止、7 个工具错误；没有模型或工具硬超时。1,424 次模型请求、1,398 个工具事件和 5,252 段原文全部通过复算。每个变体保留原始 20 段并独立索引。[轨迹核验](audits/20260910-p4-musique-replay.json)

| 严格协议指标 | 分母 | 本轮结果 | 探索性配对 bootstrap 95% CI |
| --- | --- | ---: | --- |
| Answer F1 | 100 个可回答变体 | 0.0724 | [0.0267, 0.1224] |
| Answer EM | 100 个可回答变体 | 0.0600 | [0.0200, 0.1100] |
| Support F1 | 100 个可回答变体 | 0.0733 | [0.0300, 0.1233] |
| Group answer sufficiency F1 | 100 个完整配对 | 0.0000 | [0.0000, 0.0000] |
| Group support sufficiency F1 | 100 个完整配对 | 0.0000 | [0.0000, 0.0000] |
| Answerability accuracy，附加诊断 | 全部 200 个变体 | 0.0400 | [0.0150, 0.0650] |

前五项与锁定的官方评分脚本在官方三位小数精度上一致；最后一项为附加诊断。配对充分性要求同一问题的可回答/不可回答变体均判断正确，再计可回答变体的答案或支持 F1。按完整配对重抽样 2,000 次，不独立抽两个变体。零分样本的 bootstrap 区间退化为零，不代表总体风险被精确证明为零。[官方对照](audits/20260910-p4-musique-official.json)、[区间与逐题诊断](reviews/p4-musique-20260910/summary.json)

| 预选 hop 分层 | 配对数 | final / 变体数 | 严格有效输出 | Answer F1 | Support F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2-hop | 34 | 31/68 | 8/68 | 0.2129 | 0.2157 |
| 3-hop | 33 | 14/66 | 0/66 | 0.0000 | 0.0000 |
| 4-hop | 33 | 4/66 | 0/66 | 0.0000 | 0.0000 |

**这些严格分数同时受到执行未结束和输出格式的影响，不能直接解释成多跳推理准确率。** 49 个 final 中只有 8 个通过预先固定的 JSON/来源校验；另外 41 个在 JSON 解析时失败。100 个可回答变体为 8 个 true、92 个 undefined；100 个不可回答变体全部为 undefined。undefined 不获得正确拒答分。75 个可回答变体提交过所有已知支持来源的正文片段，但严格 Answer F1 仍未满分；这是来源级代理加输出协议的观察，不证明所有必要段落完整可见，也不证明模型理解了证据。

为定位格式影响，另对**相同的 200 份原始输出**进行事后诊断：保留严格有效结果及全部非 final 失败，仅从严格无效的 final 中提取唯一的 JSON 代码块，再执行相同字段与 source 校验；不改答案、不补字段、不依据 gold 挑选。35 份恢复为可解析，6 份没有唯一合规代码块，严格有效总数从 8 增至 43。诊断 Answer F1 为 0.2140、EM 0.1800、Support F1 0.2562、配对 answer/support sufficiency 为 0.0682/0.0800，answerability accuracy 为 0.1950，均仍保留原分母。诊断预测也通过官方评分器复核。**这不是新模型的提升，也不替换主表**；它支持下一轮预先固定结构化输出机制和解析规则，把格式合规率与内容分数分别报告。[完整提取决策与分数](audits/p4-musique-format-20260910/summary.json)、[官方复核](audits/20260910-p4-musique-format-official.json)

7 个工具错误均发生在 read：5 个 document ID 格式错误（包括把 `.md` 文件名当成 ID），1 个存在的 ID 与抄错的 source 不匹配，1 个同时传 section 和字符范围。这与 Robotics 的参数绑定问题方向一致。当前没有重新尝试这些失败来挑选输出。[实际参数审计](audits/20260910-p4-musique-tool-errors.json)

Agent 阶段累计约 82.3 分钟，平均 24.7 秒/变体，p50 23.3 秒、p95 40.9 秒；另有 200 次独立上下文索引合计约 203.3 秒。1,424 次请求均有原生 usage，prompt/eval counters 合计 10,600,974 / 278,384；这是累计 provider 计数，不是独立输入量或账单。成本与包含失败的完成比例分别保留。人工审阅仍为 0/200。[成本](audits/20260910-p4-agent-costs.json)、[审阅包](reviews/p4-musique-20260910/README.md)

## 可定位的失败样例

下表是看过开发结果后选出的诊断例子，用于回到原始记录检查，不用于估计失败发生率。所有同类样本的数量和分母以前文完整矩阵为准；来源命中变化不等于回答语义判断。

| 数据与原 query ID | 实际观察 | 原始证据入口 |
| --- | --- | --- |
| Stack Overflow 静态检索，`0` | 两路并集命中 4/5 个已知正例，hybrid100 留下 2/5，重排前十只剩 1/5；涉及融合与排序两个阶段 | [逐阶段来源 ID](results/p4-regression/bright-stackoverflow-rerank-truncation/stage-attribution.json) |
| Stack Overflow 静态检索，`10` | 20 个重排输入均无正文；方面覆盖从 0.7778 降到 0.5556 | [输入 token 记录](results/p4-regression/bright-stackoverflow-rerank-truncation/rows.jsonl)、[原候选及正文](results/p4-bright-stackoverflow-indexed/candidates.jsonl) |
| Robotics 静态检索，`0` | 重排前十命中 `robotics-0/extraction_3.txt`，重排后未命中任何已知正例；此题没有完全丢掉正文的重排输入 | [逐阶段来源 ID](results/p4-regression/bright-robotics-rerank-truncation/stage-attribution.json) |
| Stack Overflow Agent，`7` | exact 工具执行触发 180 秒评估硬超时，保留错误与部分轨迹 | [完整运行行](results/p4-bright-stackoverflow-agent-v2/rows.jsonl) |
| Robotics Agent，`17` | 对 55 字符正文请求 `[0, 1000)`；范围校验拒绝后 Agent 终止 | [失败参数与实际正文长度](audits/20260910-p4-robotics-tool-errors.json) |
| Robotics Agent，`72` | 把已返回的不同搜索结果中的 document ID 和 section ID 拼到一次 read；目标文档并不拥有该 section | [先前结果与引用绑定](audits/20260910-p4-robotics-tool-errors.json) |
| MuSiQue，`2hop__242753_658778`，case 2/3 | final 在 JSON 代码块外附带解释；严格解析失败，唯一代码块提取诊断保留其中的答案/拒答与来源字段 | [提取决策](audits/p4-musique-format-20260910/decisions.json)、[原始输出审阅包](reviews/p4-musique-20260910/review-packets.json) |
| MuSiQue，`4hop1__696484_49925_13759_736921`，case 168 | read 同时传 section 与字符范围，且 end=2000 超出 1832 字符正文；接口正确拒绝后运行终止 | [失败参数](audits/20260910-p4-musique-tool-errors.json) |

## 评分、规模与更新验证

- 普通 nDCG/Recall 使用外部 qrels 的 `>0` 正例与线性 gain；全部完成排名与 `pytrec-eval-terrier==0.5.10` 的最大误差不超过 3.34×10⁻¹⁶。没有把外部 `1` 错送到 ARKB v2 的 `≥2` 正例规则。
- Bright-Pro 的两个方面指标完成 3,456 次官方纯函数对照，包含完整实际排名矩阵、空排名、gold 正序/逆序，最大误差 2.23×10⁻¹⁶；两个域没有发现跨 query 的文档→方面冲突。[交叉校验](results/p4-regression/aspect-reference.json)
- MuSiQue 的 12 组/24 个人工构造评分夹具包含正确、部分正确、错误、无效 abstention；五个指标在官方三位小数精度上一致。它们是评分器测试，不计入真实 benchmark 结果。[评分夹具校验](results/p4-regression/musique-reference.json)
- 更新契约 12/12 通过：冻结 BM25 保留旧文本；live read/match 看见新文本；revision 不同可识别；删除/重命名后旧 ID 拒绝读取；重建后仅返回新状态。该探针不代表异步向量发布时延。[更新验证](results/p4-regression/freshness.json)
- 另以真实本机模型、SQLite 与 Qdrant 完成 12/12 发布检查：t0/t1 两次索引、live 读取、删除/重命名、旧捕获 engine 保持旧快照、新 engine 读取新快照；两个远端向量快照均与 SQLite 一致。这是受控状态验证，不是异步更新 SLA。[实际发布验证](results/p4-freshness-live/results.json)
- 从真实 Bright-Pro 文本预先选择嵌套的 1k/10k 文档，用同一批 25 个查询测 BM25 操作成本，没有计算裁剪 corpus 后的质量分数。本机索引构建约 57/565 ms，query p50 约 15/112 ms，p95 约 18/130 ms；为非隔离负载下的描述性观察。[规模探针](results/p4-regression/scale.json)

## 下一轮优化与验收方式

以下是依据本轮开发证据排出的实验顺序，具体替代方案尚未测量。每项先固定输入、预算和主要指标，再运行完整配对比较；当前公开集用于诊断，最终采用仍需独立 ARKB 数据。

| 顺序 | 改动候选与依据 | 下一轮必须固定和检查的内容 |
| --- | --- | --- |
| 1 | 中文 tokenizer：T2 有 457/500 个 BM25 空排名，词法路径未向候选并集补充已标注正例 | 对照当前规则、字符 n-gram、中文分词；保持完整语料、查询和 BM25 参数，检查 nDCG@10、Recall@100、空排名率、英文/混合查询回归与索引成本 |
| 2 | 重排输入分配和同分规则：两个 Bright 技术域明显退化，56 题的候选全部丢失正文，同分覆盖原 hybrid 顺序 | 固定候选池，分别对照更长输入、显式 query/body 配额、正文无法保留时回退及同分稳定排序；检查方面覆盖、零正文率与额外推理成本，避免一次同时改多个因素后无法归因 |
| 3 | 工具执行与错误恢复：exact 逐文档子进程耗时失控，read 范围和引用绑定错误使 Agent 直接终止 | 单独验证批量匹配、已知 ID 读取定位、可取消的调用截止、绑定 document/section/range 的引用和受预算约束的恢复；必须保持原文范围、排序、revision 及 live edit/delete/rename 契约 |
| 4 | Agent 证据使用、结束与输出协议：MuSiQue 151/200 无 final，41/49 个 final 严格解析失败；Bright 也有预算停止及无正文 final | 先固定结构化输出机制与解析规则，将合规率和内容分数分开；再分别对照按需读取、证据打包及明确的证据依赖，保持检索模式和总预算。同时保留无需检索任务，人工检查正确性、主张支持和适当拒答，不能只优化 final 比例 |
| 5 | 完全重复正文与标注别名：Bright 原始 ID 多于不同正文，部分正例存在未标注的同文 ID | 保留官方 ID/qrels 主轨，另建经审阅的别名敏感性与证据去重对照；保留来源身份，不自动合并不同版本或文件身份任务 |

每项实验应在运行前登记主要收益指标、允许的质量退化、成本上限、样本量与停止规则；门槛由项目需求和独立 pilot 的方差确定，不能看过结果后反推。全部错误和超时进入分母，报告配对差值、区间、胜平负和失败切片。公开集上筛选出的方案再进入未见的已审 ARKB validation；只有预先定义的质量/成本门槛通过，才消费独立 release holdout。原始结果、事后诊断和后续正式对照分别版本化。[决策日志](experiments/p4-decision-ledger.json)

## 长期回归和发布边界

[确定性 CI 工作流](../.github/workflows/evaluation.yml) 已增加 PR/push/manual 的离线测试和更新契约入口，不下载完整公开 corpus、不启动模型服务。工作流尚未在远端运行；本机最近一次全仓验证为 1,059 passed，63 个真实服务 integration 测试按标记排除。

CI 另从仓库中已封存的 SciFact 制品加载完整 5,183 文档与 300 查询，用**当前代码**重新计算 BM25 排名及指标并对照冻结基线。本机正常环境和最小 evaluation 依赖环境均为 300/300 一致。后续改动如果改变排名，会输出逐 query 差异并失败；有意改变算法时需审阅差异、显式登记新基线。这是公开开发集的确定性回归检查，不是发布质量认证。[当前代码回归](audits/20260910-p4-current-bm25-regression.json)、[最小环境回归](audits/20260910-p4-minimal-ci-bm25-regression.json)

CI 使用固定 commit 的已发布 [checkout v7.0.1](https://github.com/actions/checkout/releases/tag/v7.0.1) 和 [setup-uv v10.1.0](https://github.com/astral-sh/setup-uv/releases/tag/v10.1.0)，Python 3.13 与 uv 0.11.14；Action 版本按官方发布记录核对。

本机隔离环境仅安装 `--extra evaluation`，最新版全套检查为 1,058 passed / 1 skipped / 63 integration deselected；工作环境为 1,059 passed / 63 deselected。跳过的是可选 Torch 测试，两次测试期间源码 hash 保持一致，原始日志已保留。此前更新契约也为 12/12 通过。这验证最小依赖路径，仍不等于 Linux 远端 CI 已运行。[最新版测试记录](audits/20260910-p4-current-tests-v2.json)、[隔离环境资料](audits/20260910-p4-ci-local-environment.json)

[轮换模块](../src/arkb/evaluation/regression.py) 在评测前消费一批独立盲测身份，以带文件锁和 hash chain 的账本记录 dataset、需求组、规范化查询 hash 和 release ID；失败尝试也不能把同一批数据重新标成未见。公开开发集、未审数据、空集合、重复身份与损坏账本会被拒绝。实际使用需由独立保管者把账本置于调参 checkout 之外；软件不能认证审阅人的独立性或物理盲法。

当前 core 未通过人工验收，因此**没有预留/消费任何真实 release holdout，没有执行真实盲测轮换，也没有据外部结果推广新产品配置**。这项人工条件独立保留，不能由公开集分数替代。

SciFact 已形成 [34.8 MiB 归档](experiments/artifacts/p4-scifact-20260910.tar.gz) 并从解包数据使用冻结源码复算：300 条原生 BM25 与 1,200 条四组检索结果一致；后者另从候选记录重组全部排名，核对 6,000 段正文与完整源文档。[归档验证](audits/20260910-p4-scifact-archive-validation.json)。归档含完整规范化 corpus、原始候选、冻结代码和许可出处，省略 SQLite/Qdrant 缓存及模型权重；离线复算分数不需模型，重新生成模型输出需要重建索引。

Stack Overflow 的 [45.4 MiB 归档](experiments/artifacts/p4-bright-stackoverflow-20260910.tar.gz) 也已通过解包复算：115 个原生 BM25 排名从完整语料重新检索，460 条四组结果重组和评分一致，25 条 Agent 的请求、工具和证据来源核对通过。[归档验证](audits/20260910-p4-bright-stackoverflow-archive-validation.json)

Robotics 的[完整归档](experiments/artifacts/p4-bright-robotics-20260910.tar.gz) 同样通过：101 个原生排名重算、404 条四组结果重组，以及 25 条 Agent 的请求/工具/原文核验均一致。[归档验证](audits/20260910-p4-bright-robotics-archive-validation.json)

T2 的完整归档为 136,707,615 字节，按三片保留（每片最多 64 MiB），[分片清单](experiments/artifacts/p4-t2-20260910.multipart.json) 同时绑定逐片和整包 SHA-256。已从分片重组并复算 500 个原生排名、2,000 条索引结果和 10,000 段正文；完整内容与原包一致。仓库忽略可重组的单个大包，保留分片和清单。[重组验证](audits/20260910-p4-t2-multipart-validation.json)

MuSiQue 的[完整归档](experiments/artifacts/p4-musique-20260910.tar.gz) 包含 200 个上下文、原始输出、gold/预测、冻结代码及审阅包；解包后 200 个变体的评分与全部请求、工具和原文一致。[归档验证](audits/20260910-p4-musique-archive-validation.json)

辅助回归、真实更新探针与两次未完成的模型尝试封存在[回归证据 v2 归档](experiments/artifacts/p4-regression-evidence-20260910-v2.tar.gz)：435 个文件校验通过，12 项确定性更新检查从解包源码重跑一致。首次辅助包漏带已被 fixture 校验表绑定的 SQLite WAL/SHM 文件，复验正确失败；原失败包保留，v2 修正规则后重新封存，实验分数没有改变。[v2 验证](audits/20260910-p4-regression-archive-validation-v2.json)、[首次失败记录](audits/20260910-p4-regression-archive-v1-failure.json)

13 次终态尝试（8 次检索、3 次完整 Agent、2 次未完成 Agent）与当前代码的 32 个生产 Python 模块逐文件相同，P4 仅增加评估与归档设施；P0–P3 已有改动保持原样。冻结版本中的评估 instrumentation 差异另在协议修订中记录。[生产源码一致性](audits/20260910-p4-production-source-consistency-v2.json)

首个 SciFact runner 没有记录向量/重排权重在运行前的字节 hash。已完成运行后的 SQLite/Qdrant 全快照一致性验证并记录权重 hash，但不能补称运行前后已做该检查；后续索引 runner 增加了前后核对。[补充完整性审计](results/p4-regression/scifact-post-run-integrity.json)

## 执行入口

```sh
uv sync --locked --extra evaluation --extra rerank
.venv/bin/python evaluation/experiments/fetch_p4.py --output evaluation/results/p4-inputs
.venv/bin/python evaluation/experiments/prepare_p4.py \
  --inputs evaluation/results/p4-inputs --output evaluation/results/p4-data-NEW
.venv/bin/python evaluation/experiments/run_p4.py \
  --dataset evaluation/results/p4-data/scifact \
  --output evaluation/results/p4-scifact-NEW --track indexed
.venv/bin/python evaluation/experiments/run_p4.py \
  --dataset evaluation/results/p4-data/scifact \
  --output evaluation/results/p4-scifact-indexed --track indexed --replay
.venv/bin/python evaluation/experiments/p4_regression.py probe \
  --output evaluation/results/p4-regression-NEW \
  --scale-dataset evaluation/results/p4-data/bright-stackoverflow
.venv/bin/python evaluation/experiments/run_p4_agents.py --track bright \
  --dataset evaluation/results/p4-data/bright-stackoverflow \
  --index-run evaluation/results/p4-bright-stackoverflow-indexed \
  --output evaluation/results/p4-bright-stackoverflow-agent-NEW
.venv/bin/python evaluation/experiments/run_p4_agents.py --track musique \
  --selected evaluation/results/p4-data/musique-selected.json \
  --output evaluation/results/p4-musique-agent-NEW
.venv/bin/python evaluation/audits/verify_p4_archive.py \
  evaluation/experiments/artifacts/p4-scifact-20260910.tar.gz \
  --output evaluation/results/p4-scifact-archive-replay-NEW.json
.venv/bin/python evaluation/experiments/check_p4_deterministic.py \
  --output evaluation/results/p4-current-code-regression-NEW.json
.venv/bin/python evaluation/experiments/p4_archive_parts.py replay \
  evaluation/experiments/artifacts/p4-t2-20260910.multipart.json \
  --output evaluation/results/p4-t2-parts-replay-NEW.json
```

新实验使用新输出目录，保留已有制品。兼容的已完成或失败索引可用 `--reuse-index <prior/index.sqlite>` 复制缓存到新实验，原尝试不被覆盖。本轮完整矩阵、失败诊断、审阅包和可复算制品均已保存；尚未配置远端长期存储或执行远端 CI。下一步按决策日志开展受控改进，同时推进独立人工 core、输出校准和未见 validation/test。
