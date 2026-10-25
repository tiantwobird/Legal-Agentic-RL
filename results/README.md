# results/

评测、分析、采集的输出目录。

## 会生成什么

| 命令 | 产出 |
| --- | --- |
| `python -m legalagent.eval.score_rollout` | `score_dev.json` / `score_test.json` |
| `python -m legalagent.eval.eval_agentic` | `eval_agentic_<split>.json` |
| `python -m legalagent.analysis.bad_case_analyzer` | `bad_case_report.md` + `bad_cases.json` |
| `python -m legalagent.tools.baseline_toolcalling` | `baseline_report.json` |
| `python -m legalagent.analysis.collect_rollouts` | `rollout_groups_<policy>.jsonl` |
| `python -m legalagent.analysis.bad_case_sampler` | `curriculum_audit.json` |
| `python -m legalagent.data.create_review_sheet` | `reviews/manual_expert_review_sheet.csv` |

`.gitignore` 里写的是 `results/*` + `!results/README.md`，所以这些文件默认不会进提交。
想发布自己的结果，用 `git add -f` 指定文件，或把值得留的结论写进 `docs/`。
