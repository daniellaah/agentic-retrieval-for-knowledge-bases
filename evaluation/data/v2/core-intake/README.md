# P3 core 数据入库与冻结

当前是入库登记和冻结工具，**尚无 300 个独立需求的 core gold**。现有 pilot 为 60 queries / 31 个待审需求族；人工确认独立性为 0。还需至少 269 个新的候选需求族，并审阅全部 300 个族；语料 58 篇，按方案 core 起步 300 篇还差 242 篇。数量不能替代质量或数据代表性。

## 已登记的数据

- `pilot-registry.json` 绑定 pilot manifest 的规范 JSON 指纹；每个 family 一条记录，保留作者、来源、使用许可、独立性审阅和 pool 审阅状态。
- `core-policy.json` 保存 120 dev / 60 validation / 120 test 的目标、固定切分 seed，以及不可清空的开发暴露清单。任务配比是原方案提案；当前每族主任务暂取第一条 query，不能据此声称某能力已有足够独立样本。
- 全部已用 pilot 保守地归入一个 `known-pilot-development` 泄漏组，锁在 dev。关联题、翻译、同主题模板不能拆到不同 split。新需求需要人工判断 leakage group；字符串校验不能自动发现所有语义泄漏。
- `readiness-20260910.json` 是当前实际缺口，不是已完成证明。已有 31 个 dev 族，目标仍缺 dev 89 / validation 60 / test 120。

## 新数据如何进入

1. 在新的入库目录准备 v2 `queries.jsonl`、`evidence.jsonl`、`qrels.jsonl`、`corpus/`、`manifest.json`。复用 pilot 必须原样保留 family/exposure；新增真实查询应先确认使用许可。记录真实来源，模型编写不得标为用户日志。
2. 更新 registry，使每个 family 恰有一条登记，重新绑定该入库 manifest 指纹。为每族指定主要任务、作者、source_reference、使用许可、保守 leakage group 和是否用于开发。已用于 prompt 调试、错误分析或 judge 调参的题都属于已暴露数据。
3. 作者以外的审阅者检查“独立信息需求”与组边界，在 `independence_review` 填姓名标识、理由与 reviewed。案例、所有 span 和所有 qrels 也必须在 v2 中完成 reviewed；程序拒绝空 reviewer 和草稿。
4. 按不同检索系统建立 pool；确认 pooled 文档判断已完成，并抽查 pool 外文档。每族保存 `pool-audits/<sha256>.json`，registry 的 `pool_review` 引用实际文件哈希。查询需由至少两种方法建 pool；除穷举全语料外，必须有 pool 外检查。已存在的 [pilot 全文档待审包](../../../reviews/pilot-20260910/README.md) 可继续使用，空判断不自动转成负例。
5. 运行 readiness；只有数量、许可、独立审阅、标签、pool 审计及完整分组切分都通过，才可 `--freeze-to`。现有目录不覆盖。切分按组大小与固定哈希顺序填充剩余配额；若无法填满，保留未分配组并停止冻结，不能拆开关联组凑数。
6. 由独立数据维护者保存未暴露 validation/test 标签及操作映射。此仓库工具没有权限隔离能力；写了 test 不代表开发者从未看过。validation 选择须登记尝试；test 失败细节一旦用于开发，就加入永久暴露清单，下一轮换未见数据。

`pool-audits` 记录格式：

```json
{
  "schema_version": "arkb-pool-audit-v1",
  "family_id": "新需求族 ID",
  "dataset_fingerprint": "当前入库 manifest 的 fingerprint",
  "reviewer": "独立审阅者 ID",
  "queries": {
    "query-id": {
      "pool_methods": ["bm25@20", "semantic@20", "hybrid@20", "rerank@20"],
      "pooled_sources": ["已完成人工判断的文档.md"],
      "outside_pool_sources": ["池外抽查且已判断的文档.md"]
    }
  },
  "rationale": "描述检索池、独立发现的证据、缺失与争议如何处理。"
}
```

`no_retrieval` 可用空列表并说明理由。所有列出的文档必须存在于语料及该 query 的 reviewed qrels；程序还会校验 family、query 集合、数据指纹和 reviewer 绑定。文件及 reviewer 字段可核验一致性，但无法认证真实身份或审阅质量。

## 命令

```sh
.venv/bin/python evaluation/experiments/prepare_core.py \
  --dataset evaluation/data/v2/pilot \
  --registry evaluation/data/v2/core-intake/pilot-registry.json \
  --policy evaluation/data/v2/core-intake/core-policy.json \
  --report /tmp/core-readiness-NEW.json

# 人工审阅、新数据和数量配额完成后，对新的入库目录运行：
.venv/bin/python evaluation/experiments/prepare_core.py \
  --dataset PATH_TO_REVIEWED_INTAKE --registry PATH_TO_REGISTRY \
  --policy evaluation/data/v2/core-intake/core-policy.json \
  --report /tmp/core-freeze-NEW.json --freeze-to evaluation/data/v2/core-NEW
```

冻结产物保留原始审阅登记、源 manifest 指纹及实际分组分配。数据冻结通过不代表模型质量通过。

## 样本量

[功效敏感性分析](../../../audits/20260910-p3-power.json) 用独立 family 的二元配对结果，按假设的改善和 discordance（两系统一成一败的比例）进行 10,000 次模拟，再用双侧精确条件二项检验计数正向检出率。120 个测试族只是起点；3–5 点提升在多种合理假设下检出率很低。

当前没有独立人工 grounded-success 配对结果，不能用草稿证据覆盖率代替质量差异，正式样本量保持未选择。待人工 pilot 给出 discordance 后重做预期差异与预算分析；这份 superiority 分析也不代替停止策略的 noninferiority 验证。
