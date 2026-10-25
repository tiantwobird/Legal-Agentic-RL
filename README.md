# LegalAgent

# 数据集与研究代码仅供研究与教学使用；不构成法律意见，不得用于替代执业律师。

面向中国劳动法的**版本感知**检索 Agent，用 GRPO 训练。

模型不允许凭记忆作答：它只能通过两个确定性工具查条文，奖励函数逐条核对
「引用是否真的来自工具返回」「条文是否与 gold 一致」「要点是否覆盖」。
仓库从法条导出、任务构建、奖励、训练、评测一路可跑，50 条单测不需要 GPU。


## 快速开始

前置：Python ≥ 3.10。跑训练需要一块 CUDA 显卡。

```bash
git clone https://github.com/tiantwobird/legalagent.git
cd legalagent

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .

# 语料随仓库分发；SQLite 快照由它确定性生成（不进 git）
python -m legalagent.data.build_db

# 50 条单测，纯 CPU，几秒钟
python -m unittest discover -s tests -t .
```

想直接试训练，再装训练依赖并跑一步冒烟：

```bash
pip install -e ".[train]"          # 或按 requirements.txt 的说明单独装匹配 CUDA 的 torch
python -m legalagent.train.train_grpo --max-steps 1 --limit 4 --download --out runs/smoke
```

`--download` 允许联网拉 `Qwen/Qwen3-1.7B`。不加这个参数时一切都在离线模式，
只读本地 Hugging Face 缓存 —— 这是为了避免训练脚本在后台偷偷下载模型。




更小的卡：把 `--num-generations 2` 和 `per_device_train_batch_size` 调小即可，
但组内样本少于 2 时 GRPO 没有优势信号。

## 目录

```text
legalagent/                 包：全部可导入代码
├── paths.py                仓库布局常量（唯一的路径来源）
├── model_io.py             模型加载与「是否允许下载」的唯一开关
├── data/                   导出 → SQLite → 任务集 → 校验
├── tools/                  两个工具 + rollout + 开放问题基线
├── reward.py               六分量确定性奖励
├── train/                  单轮 GRPO、多轮 agentic GRPO、探针、掩码审计
├── eval/                   打分与 agentic 评测
└── analysis/               失败分类、rollout 采集、课程采样

tests/                      50 条单测（+ fixtures）
data/                       labor_subset.jsonl、manifest、tasks/（600 题，含 badcase）
runs/                       训练输出（仓库内为空）
results/                    评测输出（仓库内为空）
docs/                       设计文档
scripts/                    Windows 一键脚本
```

## 数据流

```text
chinalaw 本地库
   └─ export_subset ──► data/labor_subset.jsonl（351 条 / 5 部法）
                           └─ build_db ──► data/legal_statutes.sqlite3（只读、版本感知）
                                                data/tasks/（600 题：300/100/150/50，含 badcase）
                                                └─ validate_tasks（引用 / 哈希 / 拆分 / 泄漏）
```

`data/labor_subset.jsonl` **已随仓库分发**，所以 `export_subset` 是可选的：
只有在你想换法条来源或更新版本时才需要它（它依赖上游 `chinalaw-cli`，见下文）。

## 训练是怎么设计的

- **工具层**：`search_statute` / `get_statute`，只读 SQLite，按 `applicable_date` 选版本；
  日期超出快照范围时显式返回「无法判断」，绝不拿新法条顶替。
- **奖励层**：六分量（`tool_use` / `grounding` / `gold` F1 / `answer_points` /
  `insufficiency` / 惩罚项），全部确定性，没有 LLM 判官。
- **训练层**：单轮（预注入候选）与多轮（模型自调工具，TRL `tools=`）两种 GRPO。
- **分析层**：把评测结果归到 6 类失败，并可按失败类别重采样训练题（课程采样）。

细节：[`docs/tasks-reward-train-eval.md`](docs/tasks-reward-train-eval.md)、
[`docs/tools.md`](docs/tools.md)、[`docs/bad-case-sampling.md`](docs/bad-case-sampling.md)。
完整文件清单见 [`REPO_MAP.md`](REPO_MAP.md)。

## 命令速查

所有命令在仓库根目录执行。默认路径由 `legalagent/paths.py` 统一给出。

| 阶段 | 命令 |
| --- | --- |
| 建库 | `python -m legalagent.data.build_db` |
| 校验任务集 | `python -m legalagent.data.validate_tasks` |
| 单测 | `python -m unittest discover -s tests -t .` |
| 工具 demo（不用模型） | `python -m legalagent.tools.demo` |
| 单轮训练 | `python -m legalagent.train.train_grpo --max-steps 30 --out runs/grpo_single` |
| 单轮评测 | `python -m legalagent.eval.score_rollout --split dev --grounded --adapter runs/grpo_single` |
| 多轮训练 | `python -m legalagent.train.train_grpo_agentic --max-steps 30 --out runs/grpo_agentic` |
| 多轮评测 | `python -m legalagent.eval.eval_agentic --split dev --adapter runs/grpo_agentic` |
| badcase 评测 | `python -m legalagent.eval.eval_agentic --split badcase --adapter runs/grpo_agentic` |
| 掩码审计 | `python -m legalagent.train.audit_tool_mask` |
| 失败分析 | `python -m legalagent.analysis.bad_case_analyzer` |
| 采集 → 采样 → 训练 | `.\scripts\run_curriculum.ps1`（Windows） |
| 数据 → 测试 → 训练 → 评测 | `.\scripts\run_all.ps1`（Windows） |

Linux / macOS 没有 `.ps1` 的等价脚本，直接按上表逐条执行即可；`run_all.ps1` 的每个
`Invoke-Step` 都会先打印等价的 `python -m ...` 命令。


先用当前策略在 300 道 train 题上每题采 G 条，统计组内奖励方差、成功率和错误类别，
再决定下一段训练多练哪些题。题目多时可先用 `--limit` 或分层子集控制成本，
未被采到的题会被采样器赋中性权重。

```text
collect_rollouts ──► rollout_groups.jsonl ──► bad_case_sampler ──► curriculum_train.jsonl
                                                                        │
                                        train_grpo_agentic --curriculum ┘
```

```bash
python -m legalagent.analysis.collect_rollouts --adapter runs/grpo_agentic --groups 4 --download
python -m legalagent.analysis.bad_case_sampler \
  --labels data/tasks/tasks_train_labeled.jsonl \
  --prompts data/tasks/tasks_train_prompts_only.jsonl \
  --groups results/rollout_groups_grpo_agentic.jsonl \
  --output data/tasks/curriculum_train.jsonl
python -m legalagent.train.train_grpo_agentic --curriculum data/tasks/curriculum_train.jsonl \
  --adapter runs/grpo_agentic --max-steps 10
```

两处 `--adapter` 必须指向**同一个** adapter：课程权重是按它的行为算的，若训练从基座
重新建 LoRA，策略就和课程对不上了。只有第一轮（还没有 adapter、基座采集）可以把
两处 `--adapter` 都去掉（groups 文件相应变成 `rollout_groups_base.jsonl`）。


## 常见问题

**模型下载不了 / 报「不在本地缓存」。**
加 `--download`，或设 `LEGALAGENT_ALLOW_DOWNLOAD=1`。默认是离线模式。

**显存不够。**
把 `--num-generations` 降到 2、`--max-completion-length` 降到 256；
多轮版本还可以把 `--max-rounds` 降到 2。

**`data/legal_statutes.sqlite3` 不存在。**
跑 `python -m legalagent.data.build_db`。它不进 git 是因为 `build_db` 会写入构建时间，
每次重建都会产生 diff。

**想重新导出语料。**
`export_subset` 依赖上游 CLI，仓库不内置：

```bash
git clone https://github.com/nh59yytyd5-dev/chinalaw-cli.git chinalaw-cli
# 该仓库需要它自己的 venv 与本地法规库（~/.chinalaw/chinalaw.db）
python -m legalagent.data.export_subset
```

不重新导出也能完整跑训练：`data/labor_subset.jsonl` 已经在仓库里。



## 数据来源

法条文本经上游 [`chinalaw-cli`](https://github.com/nh59yytyd5-dev/chinalaw-cli) 从公开政府法规库导出，
每条记录带 `source_url`、`source_hash`、`source_checked_at` 与 `text_sha256`，
可用 `manifest.json` 复核。本仓库**不包含任何模型权重**。



## 许可证

代码与随仓库分发的数据集以 **MIT** 发布，全文见 [LICENSE](LICENSE)。

- 法条正文来自公开政府法规库，属于官方文件；仓库保留 `source_url` 与 `text_sha256`，可逐条复核；
- 模型权重不随仓库分发。使用 `Qwen/Qwen3-1.7B` 时请遵守其自身许可证；
- 上游 `chinalaw-cli` 不包含在本仓库内，按其自身许可证使用。

如果你在研究中使用本项目，引用方式是仓库 URL + 版本号：

```bibtex
@software{legalagent2026,
  title  = {LegalAgent: version-aware labour-law retrieval agent trained with GRPO},
  author = {tiantwobird},
  year   = {2026},
  url    = {https://github.com/tiantwobird/legalagent}
}
```
