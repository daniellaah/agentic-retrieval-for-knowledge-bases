# Evaluation 优化实施进度

**当前维护范围（2026-09-11）：英文检索。** 中文专项实现、测试用例、数据生成器和默认评估分支已退出；已有数据集与 P4 历史成绩保留。以下阶段数量均为原批次结果；本次清理及最新验证见 [范围清理](../docs/english-scope-cleanup.md)。

更新：2026-09-11。按 [优化方案](eval-optimization-plan.md) 推进，当前 **P0 完成，P1/P2 工程实现与诊断完成，P3 工程与首轮受控开发实验完成，P4 工程及本轮公开开发实验完成**。P1 人工验收、P2 独立质量校准、P3 的 300-family core 与未见 validation/test，以及真实 release holdout 仍未完成。

| 阶段 | 当前状态 | 已交付 / 剩余条件 |
| --- | --- | --- |
| P0 当前基线与口径 | 完成 | 4 种固定 baseline + 默认 4B × 3 trials；280 条记录，冻结代码/语料/模型、完整快照校验、离线 replay、仓库制品归档 |
| P1 证据模型与 pilot | 工程版完成；验收未完成 | v2 schema/lint/评分、60 queries / 31 families、58 docs、198 条真实诊断；尚需真实来源、至少 29 个独立需求族、人工复核与标注耗时 |
| P2 预算、统计、输出质量 | 工程首版与实测完成；质量验收未完成 | 可选 observer、在线次数/证据预算、usage、输出绑定、gold-context/judge、校准统计和 gate；36 Agent + 8 synthesis + 4 judge，独立标签与校准仍缺失，硬实时截止/完整 embedding usage 未实现 |
| P3 core 与受控实验 | 工程和开发实验完成；core/质量验收未完成 | 184 条候选诊断 + 48 次停止消融，完整归档/replay/决策日志；core 入库/整组切分/冻结检查及 power 分析。仅 31 个待审需求族，尚缺 269 个新候选族与全部独立审阅；validation/test 尝试为 0 |
| P4 外部有效性与长期回归 | 工程与公开实验完成；人工/发布验收待完成 | [P4 报告](p4-external-results.md)：1,016 条全语料 BM25、4,064 条索引检索、Bright 50 题与 MuSiQue 200 个变体，官方对照、完整归档/replay、scale/freshness、离线 CI 和轮换保护均已完成；250 份输出待审，真实 release holdout 未消费，远端 CI 未运行 |

## 实测结论入口

- [P0 当前基线报告](p0-current-baseline-results.md)：Agent v1 93/120 成功（77.50%），6 次运行错误计入失败；逐题变化揭示总分掩盖的改善与退化。
- [P1 provisional pilot 报告](p1-pilot-results.md)：198 条记录已结算并可复算；case 与 family 聚合产生不同方向，标签覆盖不足，不选冠军。
- [P2 预算与输出报告](p2-budget-results.md)：36 次试验中 23 次正常结束、11 次预算停止、2 次耗尽轮数；输出正确性保持待审，发布检查预期退出 2。
- [P3 受控实验报告](p3-controlled-results.md)：候选 20→40 没有改善最终草稿证据覆盖；固定候选重排收益不确定；停止提醒开关各 14/24 正常结束。定位融合截断、桥接排名丢失、缺口/歧义未结束，保留默认配置。
- [P4 公开实验报告](p4-external-results.md)：定位中文词法空排名、技术长查询的重排截断/同分退化和 Agent 工具/预算问题。MuSiQue 严格 Answer F1=0.0724，事后唯一 JSON 块提取诊断=0.2140，两者分别报告；不能将输出格式影响解释成纯推理能力。后续受控实验与采用条件见决策日志。
- [数据说明](data/v2/pilot/README.md) 与 [人工复核包](reviews/pilot-20260910/README.md)：所有草稿保持 provisional，人工判断 0/3,248，无已验证的人工一致率或标注耗时。

## 工程接口

| 文件 | 职责 |
| --- | --- |
| `src/arkb/evaluation/v2.py` | 严格 manifest/hash/字段/revision/span/family 校验；显式 cutoff 排名；实际可见正文的 OR/AND evidence coverage |
| `src/arkb/evaluation/pilot.py` | 复用生产 Runtime；冻结输入、隔离索引、保留原始观察与错误；前后 snapshot/model/source 校验和评分 replay |
| `src/arkb/evaluation/comparison.py` | 完整配对集合；trial→case→family 聚合；配对 bootstrap、CI、wins/ties/losses；拒绝缺失结果 |
| `src/arkb/evaluation/reviews.py` | 空判断保持未知；实际标注进度/耗时；独立 reviewer ID 的共同样本混淆矩阵与 quadratic weighted κ |
| `src/arkb/evaluation/controlled.py` | 冻结候选前缀、固定池重排、分阶段 facet 丢失；交叉试验顺序及模型已见证据阈值诊断 |
| `src/arkb/evaluation/core.py`、`power.py` | provenance/exposure/独立审阅/pool 审计绑定、整组 split 与冻结；假设场景下的配对二元功效分析 |
| `evaluation/metric-spec.json` | v1/v2 口径与分母，不把来源覆盖或引用 ID 正确写成答案正确 |
| `evaluation/experiments/` | 预注册协议、草稿生成/盲评导出、运行、汇总、完整归档与离线验证 |

P0/P1 保留当时的被测 Agent、检索排序与知识库逻辑；P2 随后增加 opt-in Agent observer 与预算路径，未启用时的请求/轨迹兼容。P3 增加默认 true 的 `search_stall_reminder` Python 参数用于受控消融，默认请求行为保持一致。v1 标签和历史结果未重写，旧制品用各自冻结源码复算。初次 P0 启动的快照读取错误已修复，失败尝试仍保留。

## 验证证据

- 全仓确定性测试：**1059 passed，63 个真实服务 integration 测试按标记排除**。实际模型实验和真实服务探针单列记录，不计入确定性测试数。
- [排名参考实现交叉验证](audits/20260910-v2-reference.json)：504 组输入、1,946 个数值比较、70 个空分母对照，最大绝对差 3.33×10⁻¹⁶。
- 参考为独立环境中的 `ir-measures==0.4.3`、`pytrec-eval-terrier==0.5.10`；没有增加产品 runtime 依赖。明确转换 exponential gain / binary qrels，先截断到 K 再请求不带 cutoff 的 RR，避免不支持的 provider 参数改变口径。
- P0 与 pilot 均从压缩归档解包后，使用冻结源码离线重算；修改临时解包数据后，哈希校验会拒绝。见 `audits/20260910-*-archive-validation.json`。
- P2 归档还包含实际 reference tokenizer，可重算交付/返回/去重证据计数及请求 JSON 参考计数，校对原生 usage 汇总。最终评分器再次完成 [参考实现对照](audits/20260910-p2-final-reference.json)。
- P3 [解包验证](audits/20260910-p3-archive-validation.json) 重算 184 个候选结果和 48 个 Agent 输出，核对提醒插入、完整请求轨迹、实际交付证据和预算计数；篡改被拒绝，质量发布检查预期退出 2。
- P4 全量原生 BM25 排名重算 1,016/1,016 一致，四组检索的排名重组与普通指标参考误差不超过 3.34×10⁻¹⁶；Bright 3,456 次官方方面评分对照一致，MuSiQue 严格与格式诊断预测均通过官方评分器。当前代码的 SciFact 全语料 BM25 回归在工作环境与最小 evaluation 环境均为 300/300 一致。
- P4 [最新确定性检查](audits/20260910-p4-current-tests-v2.json)：工作环境 1,059 passed；最小 evaluation 依赖环境 1,058 passed / 1 skipped；均排除 63 个 integration 测试。真实服务更新检查另为 12/12。远端 CI 未运行。
- 归档保存完整输出、冻结源码、锁文件、语料、标签与哈希；P0–P3 包含 SQLite，P4 主公开实验省略可重建索引缓存和模型权重，保留所有离线复算输入。T2 使用三片归档，全部内容重组校验通过。[最终制品清单](experiments/p4-artifacts-20260910-manifest.json) 绑定各包；尚未配置远端长期制品存储。

## 继续执行所需输入

P1 的剩余工作需要**可使用的真实笔记/查询来源，以及独立人工审阅人**。已准备可直接填写的复核文件，不需要重新确认是否继续实施。先完成小批次计时和分歧检查，再确定全量标注资源；没有人工结果时不能声称完成 gold、双审或 judge 校准。

P2 已补运行错误阶段、Agent chat usage、预算、完整正文证据适配和输出评审接口，另有 [44 个待审输出包](reviews/p2-budget-20260910/README.md)。P3 受控开发实验已完成，并导出 [12 个唯一待审包（对应 48 次运行）](reviews/p3-controlled-20260910/README.md) 与 [core 入库登记](data/v2/core-intake/README.md)。质量结论仍需 gold、校准和未见数据；当前未把模型 judge 的判断自动升级为人工标签。

P4 另导出 Bright 两域各 25 份及 MuSiQue 200 份独立输出审阅包，当前人工完成数为 0。公开实验不补占 ARKB core 配额、不冒充未见 test，也没有消费真实 release holdout。下一轮可依据已定位的失败做受控优化；正式采用仍需上述独立数据与人工条件。

## 常用命令

```sh
# 确定性验证
.venv/bin/python -m pytest -q -m 'not integration'
.venv/bin/python -m arkb.evaluation.v2 evaluation/data/v2/pilot \
  --notes-dir evaluation/data/v2/pilot/corpus --allow-provisional

# 仅在本地 Ollama/Qdrant 可用时，使用新的输出目录重测
.venv/bin/python evaluation/experiments/run_p0.py \
  --output evaluation/results/p0-NEW-ID --qdrant-url http://127.0.0.1:32776
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m arkb.evaluation.pilot \
  --output evaluation/results/pilot-NEW-ID --qdrant-url http://127.0.0.1:32776 --allow-provisional
```

服务地址是本次本机地址，其他环境应替换。所有 runner 拒绝覆盖已有目录；未来实时重测不承诺轨迹逐字相同。
