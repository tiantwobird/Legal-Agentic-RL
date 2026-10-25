# tasks（600 条）

劳动法 RL 任务集：**train 300 / dev 100 / test 150 / badcase 50**。
由 150 条早期人工题 + 450 条新增题组成，合并时按 `task_id` 去重，当前无重复题。

## 文件

| 文件 | 行数 |
| --- | ---: |
| `tasks_all_reviewed.jsonl` | 600 |
| `tasks_train_labeled.jsonl` | 300 |
| `tasks_train_prompts_only.jsonl` | 300 |
| `tasks_dev_labeled.jsonl` | 100 |
| `tasks_dev_prompts_only.jsonl` | 100 |
| `tasks_test_labeled.jsonl` | 150 |
| `tasks_test_prompts_only.jsonl` | 150 |
| `tasks_badcase_labeled.jsonl` | 50 |
| `tasks_badcase_prompts_only.jsonl` | 50 |

- `*_labeled.jsonl`：完整标注（`gold_article_ids`、`gold_issues`、`expected_answer_points`、`missing_facts`、`source_evidence`、`review` 等）。
- `*_prompts_only.jsonl`：每行恰好 4 键 `task_id, prompt, jurisdiction, applicable_date`，用于推理/评测输入。
- `tasks_all_reviewed.jsonl`：全部 600 条。
- `audit_summary.json`：数量、题型、难度、法条覆盖等统计。
- `MANIFEST.json`：各文件 SHA256；test 与 badcase 已冻结（`frozen: true`）。
- `generation_report.json`：来源归档 SHA256、数量、覆盖、相似度与校验摘要。

## 使用

- 当前就是仓库生效的 `data/tasks/`，训练/评测默认读取，不需要额外参数。
- badcase 只用于评测（`--split badcase`），不得混入训练。
- test/badcase 已冻结；修改后 `MANIFEST.json` 校验会失败。

## 来源

- 输入归档：`tasks_source.zip`
- 归档 SHA256：`cb6b7ac86e84e89f40f6df6415636bd11dcf3a0e15a2a4db2b20c8b19fc784c7`
- 语料：`data/labor_subset.jsonl`（351 条 / 5 部法）

## 限制

- 仅覆盖 5 部法律/司法解释（语料 351 条），无历史版本。
- 统一适用日期 2024-06-01。
- 内容由助手生成并经程序化锚点校验与自检；**未经过独立律师复核**
  （`independent_human_legal_expert_review = pending`），不构成法律意见。
- 35 条 test 仍为非结构化 prompt；少数早期题存在要点超长或引文截取偏差，未改动。
