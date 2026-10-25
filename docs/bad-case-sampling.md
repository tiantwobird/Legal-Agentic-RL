# LegalAgent-RL — Bad-Case-aware Task Sampler

这是 **阶段之间的任务重采样器**，不是 GRPO Trainer，也不是离线经验回放模块。


## 输入

`tasks_train_labeled.jsonl`：供采样统计使用，绝不能传进 agent prompt。
`tasks_train_prompts_only.jsonl`：仅可模型可见内容。
`rollout_groups.jsonl`：由**当前同一个 policy_version**对训练题每题采样 G 次后汇总，字段示例：

```json
{
  "task_id": "LABOR_011",
  "policy_version": "ckpt_step_100",
  "rewards": [0.0, 0.4, 0.6, 1.0],
  "successes": [0, 0, 1, 1],
  "error_types": [
    ["retrieval_error"],
    ["citation_error"],
    [],
    []
  ]
}
```

`successes` 应由事先冻结的可靠任务成功判据得出，并不要求 reward=1；`error_types` 为每条失败轨迹的已核实错误标签，可多选。
`rewards` 是该轨迹的连续 GRPO 奖励；如果一组答案都整体错误但各有部分进展，只要 reward 不同，仍可存在 GRPO 学习信号。
不要把历史轨迹直接重复使用于下一轮策略更新；**只用旧的错误摘要来决定选哪些题，新轨迹一定重新从当前模型采样**。

## 运行

```bash
python -m unittest discover -s tests -t . -p "test_sampler.py"

python -m legalagent.analysis.bad_case_sampler \
  --labels data/tasks/tasks_train_labeled.jsonl \
  --prompts data/tasks/tasks_train_prompts_only.jsonl \
  --groups tests/fixtures/example_rollout_groups_SYNTHETIC.jsonl \
  --count 35 --seed 42 --explore 0.30 \
  --output results/curriculum_prompts_only.jsonl \
  --audit results/curriculum_audit.json
```

## 核心方法

对任务 i，G 条同策略轨迹的奖励标准差为 s_i，完整任务成功比例为 a_i，观察到的错误类别的全局频率为 e_i：

- `signal_i = min(1, s_i / 0.20)`
- `raw_i = 1 + 1.5 signal_i + 0.5 (1 - a_i) + 0.7 e_i`
- 若 `s_i≈0`，`raw_i *= 0.35`；若 `a_i=1`，`raw_i *= 0.55`。
- `w_i = clip(raw_i, 0.4, 3.0)`，最后 `P(i)=0.30/N+0.70*w_i/sum_j w_j`。

系数是 **启发式初值**，只保证可解释、保留均匀探索、对有组内奖励差异的困难题倾斜；没有经过效果验证。不要当成论文方法已证实。

若奖励全为 0，不能直接把该问题一味提高采样权重，因为 GRPO 同组奖励一致、无组内相对优势。应先调查是否 reward 过稀疏、工具出错、标签不对或需要少量 SFT。

## 接入 TRL 的建议

- 首先用固定 train/validation 数据，跑 Qwen3-1.7B Agent baseline，记录完整工具调用日志和每题 G=4 条轨迹。
- 从**当前 checkpoint**的 train rollout 整理 `rollout_groups.jsonl`，再运行上述脚本生成 `curriculum_prompts_only.jsonl`。
- 将输出 JSONL 作为**下一短训练阶段**的任务输入；TRL 的每题 G 个 completion 仍在那个阶段由**当时的当前策略**重新生成。不要让模型看到标注文件。
- 每隔一小段训练步骤重新评测/计算权重。若采用 GRPOTrainer，最简单的 MVP 是按训练阶段重新初始化 trainer/dataset 并从最新 adapter checkpoint 继续训练；注意保持优化器/调度器的连续性和相同训练预算，避免误解释实验结果。也可实现自定义可更新 sampler，但不要在 Trainer 中途无控制地替换 dataloader。
- `curriculum_prompts_only.jsonl` 允许重复 task_id（有放回采样）；每个实例仍由训练器在本策略下产生新的 G 条轨迹。同个训练 batch 中要按工具框架的批处理要求确保不会错误混组。


## 合规与质量约束

- 文件自带的 gold 标签在上传的数据中尚未经过独立法律专家确认。研究实验可以试运行，不等于权威法律意见。
- 不应通过误报、泄漏标准答案、缩短答案来刷 reward；重大引用错误应受硬性约束和复核。



等价的三条命令：

| 步骤 | 命令 | 产出 |
| --- | --- | --- |
| 1 采集 | `python -m legalagent.analysis.collect_rollouts --adapter runs/grpo_agentic --groups 4` | `results/rollout_groups_<version>.jsonl` |
| 2 采样 | `python -m legalagent.analysis.bad_case_sampler --labels ... --prompts ... --groups ...` | `data/tasks/curriculum_train.jsonl` |
| 3 训练 | `python -m legalagent.train.train_grpo_agentic --curriculum data/tasks/curriculum_train.jsonl --adapter runs/<同一 policy_version> --max-steps 10` | `runs/<name>_curr/` |

第 3 步的 `--adapter` 必须与第 1 步采集用的 adapter 相同（第一轮基座采集时省略）。
训练脚本会把 `PeftModel` 交给 TRL 并传 `peft_config=None`：TRL 会冻结一份
`ref` adapter 作 KL 参照，使本阶段的正则项对准「本阶段起点」，而不是基座模型。
续训记录写在 `runs/<name>_curr/config.json` 的 `lineage` 字段里。

### 错误类别是怎么来的

采样器只认 4 个错误类别，分析器有 6 个失败类，映射写在 `collect_rollouts.ERROR_MAP`：

| 分析器失败类 | 采样器错误类别 | 含义 |
| --- | --- | --- |
| `no_tool` | `tool_planning_error` | 该调工具没调 |
| `no_recall` | `retrieval_error` | 调了但返回 0 条候选 |
| `hallucinated` | `citation_error` | 编造 article_id |
| `wrong_article` | `citation_error` | 引了真实但非本题 gold 的条文 |
| `missing_points` | `reasoning_error` | 要点缺失 |
| `insufficient_handling` | `reasoning_error` | 信息不足时硬答 |

成功判据取分析器的 `ok`（调了工具、召回到候选、无幻觉、命中 gold、要点达标），**不是 reward=1**。
这是保守定义：reward 可能在「答对一半」时给 0.6，那种轨迹算失败，但分数照样进 `rewards` ——
采样器正是靠组内 reward 方差识别「有部分进展、值得多练」的题。

