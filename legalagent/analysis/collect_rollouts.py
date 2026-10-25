"""Collect per-task rollout groups for the bad-case-aware task sampler.

The sampler (`analysis.bad_case_sampler.py`) needs, for every *train* task, G
rollouts from **one** policy version with their rewards, binary successes and
verified error labels. Nothing in this repository produced that file before, so
this script is the missing producer:

    current policy ──G rollouts/task──► score_rollout ──► classify ──► rollout_groups.jsonl

It runs the same compact tools, the same system prompt and the same tool loop as
`train.train_grpo_agentic`, so the groups describe the policy that is actually
about to be updated. Nothing from the gold labels enters the prompt.

    # 用当前 agentic adapter 采一遍 train 题
    python -m legalagent.analysis.collect_rollouts --adapter runs/grpo_agentic --groups 4

    # 用基座模型采（还没有 adapter 时）
    python -m legalagent.analysis.collect_rollouts --groups 4 --policy-version base

然后拿输出喂给 sampler：

    python -m legalagent.analysis.bad_case_sampler \
        --labels data/tasks/tasks_train_labeled.jsonl \
        --prompts data/tasks/tasks_train_prompts_only.jsonl \
        --groups results/rollout_groups_grpo_agentic.jsonl \
        --output data/tasks/curriculum_train.jsonl \
        --audit results/curriculum_audit.json

不要拿 dev/test 的错题来训练或重采样；`--split` 只接受 `train`。

成本：G 条 × N 题 × (最多 max_rounds 轮) 次生成。16 GB 单卡、Qwen3-1.7B 下
单题单条约 40–60 秒，所以 35 题 × G=4 大约 1.5–2.5 小时。先用 `--limit 3` 试通。
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from ..model_io import check_adapter_dir, load_adapter_model, load_model, load_tokenizer, resolve_device
from .bad_case_analyzer import classify
from .bad_case_sampler import read_jsonl, save_jsonl
from ..eval.eval_agentic import run_task
from ..eval.score_rollout import load_split
from ..paths import RESULTS
from ..reward import RewardConfig, score_rollout

# 分析器有 6 个失败类，采样器只认 4 个错误类别。映射是设计选择，不是等价关系：
#   tool_planning_error ← 该调工具没调
#   retrieval_error     ← 调了但没召回候选
#   citation_error      ← 编造 ID 或引了非 gold 的真实条文（两种都污染引用质量）
#   reasoning_error     ← 要点缺失 / 信息不足时硬答
ERROR_MAP = {
    "no_tool": "tool_planning_error",
    "no_recall": "retrieval_error",
    "hallucinated": "citation_error",
    "wrong_article": "citation_error",
    "missing_points": "reasoning_error",
    "insufficient_handling": "reasoning_error",
}

# 成功判据：沿用 bad_case_analyzer 的 "ok"（调了工具、召回到候选、无幻觉、
# 命中 gold、要点达标、信息不足题有交代）。它是一个保守的「完整成功」，
# 与 reward 高低不完全等价 —— 采样器本来就用 rewards 的方差看部分进展。
SUCCESS_CLASS = "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="train", choices=["train"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--groups", type=int, default=4,
                        help="每题采样条数 G，至少 2（组内要有奖励方差）")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter", default="", help="LoRA adapter 目录；留空用基座模型")
    parser.add_argument("--download", action="store_true",
                        help="允许联网下载模型（离线环境下不要加）")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-version", default="",
                        help="同一次采集必须一致；默认取 adapter 目录名，否则 base")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    if args.groups < 2:
        raise SystemExit("--groups 至少为 2：一组的奖励若完全相同，采样器会把它降权")

    import torch

    version = args.policy_version or (Path(args.adapter).name if args.adapter else "base")
    tasks = load_split(args.split, args.limit or None)
    print(f"[collect] split={args.split} tasks={len(tasks)} G={args.groups} "
          f"policy_version={version}")

    tokenizer = load_tokenizer(args.model, download=args.download)
    model = load_model(args.model, download=args.download, dtype="auto").eval()
    if args.adapter:
        check_adapter_dir(args.adapter, args.model)
        model = load_adapter_model(model, args.adapter, trainable=False).eval()
        print(f"[adapter] {args.adapter}")
    model.to(resolve_device())

    cfg = RewardConfig()
    rows: list[dict] = []
    class_counts: Counter[str] = Counter()

    for task in tasks:
        rewards: list[float] = []
        successes: list[int] = []
        errors: list[list[str]] = []
        for i in range(args.groups):
            torch.manual_seed(args.seed + i)
            rollout = run_task(model, tokenizer, task, args.max_new_tokens,
                               args.max_rounds, sample=True, temperature=args.temperature)
            breakdown = score_rollout(task, rollout, cfg)
            cls, _why = classify(task, {
                "components": breakdown.components,
                "details": breakdown.details,
                "tool_ids": rollout["tool_ids"],
            })
            ok = cls == SUCCESS_CLASS
            rewards.append(round(breakdown.total, 4))
            successes.append(1 if ok else 0)
            errors.append([] if ok else [ERROR_MAP.get(cls, "reasoning_error")])
            class_counts[cls] += 1

        rows.append({
            "task_id": task["task_id"],
            "policy_version": version,
            "rewards": rewards,
            "successes": successes,
            "error_types": errors,
        })
        spread = max(rewards) - min(rewards)
        print(f"  {task['task_id']:<10} success={sum(successes)}/{args.groups} "
              f"reward=[{min(rewards):.2f},{max(rewards):.2f}] spread={spread:.2f}")

    out = Path(args.out) if args.out else RESULTS / f"rollout_groups_{version}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(out, rows)

    usable = sum(1 for row in rows if len(set(row["rewards"])) > 1)
    print(f"\n[written] {out}  ({len(rows)} tasks)")
    print(f"[signal] {usable}/{len(rows)} 题组内有奖励差异（其余会被采样器降权）")
    print(f"[classes] {dict(sorted(class_counts.items()))}")
    if usable == 0:
        print("[warn] 所有组奖励都相同：采样器退化为均匀采样。先查 reward 是否过稀疏、"
              "tool 是否报错、temperature 是否太低。")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
