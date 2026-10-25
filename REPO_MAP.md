# 仓库地图

按「数据 → 工具 → 奖励 → 训练 → 评测 → 分析」六层组织，每个文件标注角色。
所有命令在仓库根目录执行；路径统一由 `legalagent/paths.py` 给出。

## 一、目录

| 路径 | 角色 |  |
| --- | --- | --- |
| `legalagent/` | 全部可导入代码（六层包） |  |
| `tests/` | 50 条单测 + fixtures |  |
| `data/` | 语料、任务集（`tasks/` 600 题） |  |
| `runs/` | 训练输出目录 |  |
| `results/` | 评测输出目录 |  |
| `docs/` | 设计文档 |  |
| `scripts/` | Windows 一键脚本（UTF-8 BOM） |  |
| `chinalaw-cli/` | 上游法规库 CLI，**不内置** | ⛔️ 自行 clone |

## 二、分层清单

### L0 数据 —— `legalagent/data/`

| 文件 | 作用 |
| --- | --- |
| `export_subset.py` | 调上游 `chinalaw get` 导出 5 部法 / 351 条 → `data/labor_subset.jsonl` + `manifest.json`（可选步骤） |
| `build_db.py` | 校验 JSONL（字段 / 日期 / 哈希 / 重复 / 版本区间）→ `data/legal_statutes.sqlite3`（原子替换） |
| `validate_tasks.py` | 校验当前 600 题：引用、哈希、日期、拆分隔离、gold 泄漏、badcase 隔离 |
| `create_review_sheet.py` | 生成专家复核 CSV → `results/reviews/` |
| `curriculum.py` | 把课程文件（prompt-only）与训练标签按 `task_id` 连接 |

### L1 工具 —— `legalagent/tools/`

| 文件 | 作用 |
| --- | --- |
| `legal_tools.py` | `search_statute` / `get_statute`：版本感知、只读 SQLite、日期越界显式报错 |
| `tools_schema.py` | 工具 JSON Schema + `dispatch()` 唯一分发入口 |
| `agent_tools.py` | **训练专用精简版** `search_compact` / `get_compact`（返回 JSON 字符串） |
| `rollout_demo.py` | 单题多轮 rollout：生成 → 解析 → 执行 → 回填 |
| `baseline_toolcalling.py` | 开放问题基线（6 题） |
| `demo.py` | 不需要模型的最小示例 |

> 为什么有两套工具：完整版返回太长，会被 TRL 判定超预算并整体回滚，
> 模型看不到结果、reward 只看到悬空的 tool_call。训练必须用精简版，评测与人工查条用完整版。

### L2 奖励 —— `legalagent/reward.py`

六分量确定性奖励 + 幻觉 / 过度引用 / 零引用惩罚，附 TRL 适配器工厂。

### L3 训练 —— `legalagent/train/`

| 文件 | 作用 |
| --- | --- |
| `train_grpo.py` | 单轮 grounded GRPO（预注入候选，LoRA r=16；`--adapter` 续训） |
| `train_grpo_agentic.py` | 多轮 agentic GRPO（模型自调工具，TRL `tools=`；`--adapter` 续训） |
| `probe_tool_flow.py` | 探针：验证 reward 能否看到工具轨迹 |
| `audit_tool_mask.py` | 审计 TRL 的 tool token mask（工具返回是否被正确排除出 loss） |

### L4 评测 —— `legalagent/eval/`

| 文件 | 作用 |
| --- | --- |
| `score_rollout.py` | 任务集打分（`--grounded` / `--adapter`；`--split train/dev/test/badcase`） |
| `eval_agentic.py` | 多轮 agentic 评测（与训练同工具、同提示；支持 `badcase`） |

### L5 分析 —— `legalagent/analysis/`

| 文件 | 作用 |
| --- | --- |
| `bad_case_analyzer.py` | 评测结果归为 6 类失败 → `results/bad_case_report.md` |
| `collect_rollouts.py` | 用当前策略在 train 题上采 G 条 → `results/rollout_groups_<policy>.jsonl` |
| `bad_case_sampler.py` | 按组内奖励方差 / 成功率 / 错误类别算权重，产出课程集 |

### 基础设施

| 文件 | 作用 |
| --- | --- |
| `legalagent/paths.py` | `ROOT` / `DATA` / `TASKS` / `RUNS` / `RESULTS` / `DOCS` |
| `legalagent/model_io.py` | 统一的模型加载与「是否允许下载」开关（`--download` / `LEGALAGENT_ALLOW_DOWNLOAD`） |

## 三、产物与输出约定

| 位置 | 内容 |  |
| --- | --- | --- |
| `data/labor_subset.jsonl` | 351 条法条语料（含哈希与来源 URL） |  |
| `data/tasks/*.jsonl` | 任务集 600 题（300/100/150/50），含 prompts-only 与 badcase |
| `data/legal_statutes.sqlite3` | 只读快照，`build_db` 生成 |  |
| `runs/<name>/` | adapter、`trainer_state.json`、`config.json`、`reward_stats.json` |  |
| `results/` | 评测 / 分析 / 采集输出 |  |

`.gitignore` 忽略全部 `*.safetensors` / `*.pt` / `*.pth` / `*.bin`，
所以仓库不会意外变大；要发布自己的 adapter 用 `git add -f`。

## 四、环境

- Python ≥ 3.10；核心层零第三方依赖
- 训练：`pip install -e ".[train]"`，或按 `requirements.txt` 装匹配 CUDA 的 torch
- 默认离线，只有加 `--download` 或设 `LEGALAGENT_ALLOW_DOWNLOAD=1` 才会联网

