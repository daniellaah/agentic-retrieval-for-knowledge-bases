# P1 provisional pilot：2026-09-10

**证据评分链路已实测跑通，但数据尚未达到 P1 金标准验收要求。** 60 条查询来自 31 个信息需求族，所有问题、证据与 qrels 均为未独立复核的草稿；本轮不发布模型优劣结论。

## 范围与有效性

按 [冻结协议](experiments/pilot-protocol.json)，在 [pilot 语料](data/v2/pilot/README.md) 的独立新索引上执行 198 条记录：46 个问题 × 4 种检索方法，6 个显式 match、4 个显式 read，以及 4 个 no-retrieval 不适用项。只把 query 或显式工具夹具参数交给生产 Runtime，标签在得到观察后才参与评分。

194 条执行成功，4 条不适用，0 个运行错误。语料、代码、模型和完整向量快照的前后检查通过；194 条执行结果的指标均可从原始观察重算。这个有效性检查证明实验记录和计算一致，不证明标注正确。

保留 [机器摘要](experiments/p1-pilot-20260910-summary.json)、约 2.05 MB 的 [完整压缩制品](experiments/artifacts/p1-pilot-20260910.tar.gz) 及 [哈希 manifest](experiments/artifacts/p1-pilot-20260910.tar.manifest.json)。数据来源、待复核问题、语料局限见 [dataset card](data/v2/pilot/README.md)。

## 诊断结果

Evidence coverage 使用实际返回的正文、revision、字符区间与证据 OR/AND 组合；source 名称命中不直接得分。46 个检索问题中，43 个有定义好的证据要求，属于 26 个 family；3 个缺证据/歧义问题没有可用正例分母，保留 null。

| 方法 | 按查询均分，43 queries | 按 family 均分，26 families | 草稿 Assessed@10，46 queries |
| --- | ---: | ---: | ---: |
| BM25 | 8.14% | 10.58% | 3.70% |
| Semantic | 87.21% | 93.27% | 13.70% |
| Hybrid | 84.88% | 91.99% | 13.70% |
| Hybrid + Rerank | 81.40% | 92.31% | 13.04% |

这些是 **provisional evidence coverage**，不是 grounded task success。Assessed@10 表示 top-10 位置中已有草稿 qrel 的比例，包含不足 10 个 unique source 的空位；不是人工判断覆盖率。真正独立人工复核覆盖率目前为 0%。qrels 稀疏且未复核，因此本报告不把 nDCG/Recall 排名当作选择模型的依据。

6 个显式 match 的返回文件集合均符合草稿夹具，4 个显式 read 的证据覆盖均为 1。它们的输入参数由夹具直接指定，不能推论 Agent 会选对工具、参数或答案。文件名 match 的 2 条观察没有字符坐标，当前严格 span scorer 将其列为不计证据的观察；原始全文仍保存，exact 文件集合判断不受影响。将“完整正文已返回”安全映射为范围是后续 v2 Agent 适配需要覆盖的场景。

答案正确性、事实完整性、引用支撑、缺证据时的克制回答均未在本轮运行，相关字段保持 null。部分可答题即使覆盖了已有证据，也不能因此判定最终回答正确。

## 统计口径为何影响方向

按 family 聚合：先平均同一 case 的 trials，再平均 family 内的 cases，最后各 family 等权。此次每个方法只有一次 trial，以下 95% 区间使用 10,000 次配对 family bootstrap、seed `20260910`，均为探索性诊断。

| 变化 | family 均值差 | 95% CI | 胜 / 平 / 负（family） |
| --- | ---: | ---: | ---: |
| Semantic → Hybrid | −1.28 个百分点 | [−3.85, 0.00] | 0 / 25 / 1 |
| Hybrid → Hybrid + Rerank | +0.32 个百分点 | [−4.81, +5.13] | 2 / 23 / 1 |

两项比较均不支持确定改善结论；区间也不覆盖标注错误、语料来源偏差或生产分布不确定性。

Rerank 的 case 均分下降，而 family 均分略升：`pilot_040/041/042` 是同一多跳模板的三个实例，各从 coverage 1 降为 0；`pilot_024` 从 0.5 升为 1，`pilot_055` 从 0 升为 1。前者不能算三份独立的模板退化证据，后两者属于其他需求族。这是本轮保留 family 身份的直接价值。

## 可验证的下一步

1. **先审多跳桥接标签。** `pilot_040/041/042` 在 rerank 后仍有政策段落，但服务到现行政策的桥接 span 未返回。先确认该桥接是否为任务必需，再冻结同一候选列表比较排序前后；本轮尚未运行冻结候选实验，不能直接断言根因是 reranker。
2. **复核跨度与替代证据。** 部分草稿使用整段而非最小事实 span；检查是否错误惩罚了已足够回答的片段，以及是否遗漏替代来源。
3. **建立独立标签覆盖。** 按 [人工复核包](reviews/pilot-20260910/README.md) 先计时审阅 50 个文档组合和 10 个问题，再安排完整相关性、证据、rubric 与第二审阅。未知 qrels 不补成 0。
4. **补充真实需求与语料。** 当前距 60 个独立需求族至少还差 29 个。补充后重新按 family/主题检查近重复与 split，保留这批已见数据为 dev。
5. **完成 P1 验收后再进入 P2。** 用人工结果校准输出 judge、落实预算/usage/错误阶段记录、gold-context 诊断及有效性 gate；统计基础模块已准备，不代表这些项目已经完成。

## 离线复算

```sh
mkdir -p /tmp/arkb-pilot-replay
tar -xzf evaluation/experiments/artifacts/p1-pilot-20260910.tar.gz -C /tmp/arkb-pilot-replay
.venv/bin/python evaluation/experiments/verify_bundle.py \
  /tmp/arkb-pilot-replay/p1-pilot-20260910

# 查看实际人工进度，不会填补缺失标签
.venv/bin/python -m arkb.evaluation.reviews \
  --dataset evaluation/data/v2/pilot --review evaluation/reviews/pilot-20260910
```
