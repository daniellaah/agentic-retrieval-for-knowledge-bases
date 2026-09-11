# v2 pilot：待复核的开发数据

状态：**provisional，禁止作为发布金标准或独立盲测集**。这是 P1 的工程与标注草稿，不表示 P1 人工验收已完成。

## 数据范围

| 资产 | 当前数量 | 含义 |
| --- | ---: | --- |
| 查询 | 60 | 新编写的开发查询，全部已被开发过程看到 |
| 信息需求族 | 31 | 同事实的检索、找资料、问答共用 family；六个多跳模板实例也共用一个 family |
| 文档 | 58 | 现有 40 篇 example notes + 18 篇明确标为虚构的 service/policy/archive 夹具 |
| 证据 span | 48 | 精确正文范围、revision、quote、SHA-256；当前均未独立复核 |
| query-document 标签 | 78 | 仅草拟相关/负例标签；其余组合为未判断 |
| split | dev 60 | 无 validation/test，不可宣称 60 个独立信息需求 |

任务分布见 [validation.json](validation.json)：12 semantic、12 exploratory、12 QA、6 multi-hop、6 exact、4 read、4 evidence gap、4 no retrieval。

来源全部是 `assistant_authored` 查询及现有示例/合成语料，不是生产用户日志。中文问题主要搜索英文笔记。缺少真实中文语料、长文、自然多跳、大规模语料及真实请求频率；任何均分都不能解释为生产成功率。现有 gold 段落有时长于最小证据，需人工检查边界，防止把上下文未全覆盖误判为事实缺失。

原计划的 60 个独立需求尚未达到。需要补至少 29 个经过独立性检查的新需求族，不能靠翻译、换实体或重写同一问题补足。

## 文件与契约

- `manifest.json` 冻结三份 JSONL 哈希、语料正文哈希、文档 revision 和来源。
- `queries.jsonl` 保存信息需求、family、任务、answerability、rubric 和 facet。
- `evidence.jsonl` 保存正文 span。坐标基于 `Note.content`：去除首个 H1、外层 whitespace strip 后的 Python Unicode 字符位置，右端不包含。
- `qrels.jsonl` 保存 0–3 级标签；Recall 正例阈值为 ≥2。未出现的组合不是已验证的负例。
- `review.md` 是含草稿标签的证据核对材料；盲评文档时先不要阅读它。
- `corpus/` 是唯一送入检索的目录；查询、答案、qrels、review 不入索引。

一个 facet 的 `alternatives` 是 OR，每个 alternative 数组内的 spans 是 AND。实际返回的文本、正文范围和 revision 全部匹配才计覆盖；来源文件名命中不能替代证据覆盖。回答正确性与 grounded success 暂为 null。

`exact_pattern` / `read_source` 仅用于显式工具契约诊断；检索模型只能看到 query。显式工具结果不是 Agent 选择工具的成功率。零匹配 exact 和无答案任务不使用普通正例 Recall 分母。

## 复核与版本管理

使用 [人工复核流程](../../../reviews/pilot-20260910/README.md)。完成独立审阅前保留所有 `provisional` 和空 `reviewers`。不可只批量替换状态为 reviewed。

修订标签/文本时建立新的数据目录及 manifest，保留原版、修改理由和审阅记录；所有比较系统在同一标签版本重新评分。`build_pilot.py` 是初始草稿生成器，拒绝覆盖目录，不能用于重置已开始的人工审阅。

```sh
.venv/bin/python -m arkb.evaluation.v2 evaluation/data/v2/pilot \
  --notes-dir evaluation/data/v2/pilot/corpus --allow-provisional
```

不传 `--allow-provisional` 会拒绝执行，这是预期行为。Schema 校验只证明数据自洽，不能证明标签语义正确。
