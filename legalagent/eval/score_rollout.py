"""Score a model against the pilot tasks with the deterministic reward.

    python -m legalagent.eval.score_rollout --split dev              # 10 dev tasks
    python -m legalagent.eval.score_rollout --split test --limit 5   # frozen test split
    python -m legalagent.eval.score_rollout --split dev --few-shot --sample

The model is loaded once; each task runs the usual generate → tool → generate
loop, then `reward.score_rollout` grades it. Nothing from the gold labels ever
enters the prompt: `tasks_*_prompts_only.jsonl` is what the model sees.

This is the same number you will compare before/after GRPO, so keep the config
and the split fixed when reporting.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from ..paths import RESULTS, TASKS
from ..reward import RewardConfig, score_rollout
from ..tools.rollout_demo import SYSTEM_PROMPT, build_few_shot, converse, load_model


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_split(split: str, limit: int | None) -> list[dict]:
    """Model sees prompts_only; labels come from the labelled file, joined by task_id."""
    labeled = {t["task_id"]: t for t in read_jsonl(TASKS / f"tasks_{split}_labeled.jsonl")}
    prompts = read_jsonl(TASKS / f"tasks_{split}_prompts_only.jsonl")
    tasks = []
    for item in prompts:
        task = labeled.get(item["task_id"])
        if task is None:
            raise KeyError(f"{item['task_id']} missing from tasks_{split}_labeled.jsonl")
        tasks.append({**task, "prompt": item["prompt"]})
    return tasks[:limit] if limit else tasks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test", "badcase"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--few-shot", action="store_true")
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--grounded", action="store_true",
                        help="与 train_grpo.py 同构：先注入候选条文，模型只负责作答（只检验训练效果）")
    parser.add_argument("--adapter", default="", help="LoRA adapter 目录；留空则评基座模型")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    tasks = load_split(args.split, args.limit or None)
    model, tokenizer = load_model(args.model, not args.download, args.device)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"[adapter] {args.adapter}")

    if args.grounded:
        from ..train.train_grpo import SYSTEM as TRAIN_SYSTEM, build_rows

        grounded_rows = build_rows(tasks, None)
        by_task = {row["task_id"]: row for row in grounded_rows}

    gen_args = SimpleNamespace(
        max_new_tokens=args.max_new_tokens,
        max_rounds=args.max_rounds,
        enable_thinking=args.enable_thinking,
        sample=args.sample,
        temperature=args.temperature,
        seed=args.seed,
        print_chars=400,
    )

    cfg = RewardConfig(w_tool_use=0.0, w_grounding=0.15, w_gold=0.35, w_points=0.40) if args.grounded else RewardConfig()
    rows: list[dict] = []
    print(f"[model] {args.model} on {args.device} | split={args.split} tasks={len(tasks)} | grounded={args.grounded}\n")

    for task in tasks:
        if args.grounded:
            row = by_task[task["task_id"]]
            messages = [
                {"role": "system", "content": TRAIN_SYSTEM},
                {"role": "user", "content": row["user"]},
            ]
            candidates = list(row["candidate_ids"])
            gen_args.max_rounds = 1
        else:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            if args.few_shot:
                messages.extend(build_few_shot())
            messages.append({
                "role": "user",
                "content": f"{task['prompt']}\n\n适用日期：{task['applicable_date']}",
            })
            candidates = None

        result = converse(model, tokenizer, messages, gen_args, verbose=False)
        if candidates is not None:
            result = {
                **result,
                "tool_ids": candidates,
                "tool_calls": [{"name": "search_statute", "ok": True}],
            }
        scored = score_rollout(task, result, cfg)
        rows.append({
            "task_id": task["task_id"],
            "task_type": task["task_type"],
            "total": scored.total,
            "components": scored.components,
            "details": scored.details,
            "answer": result["answer"],
            "tool_ids": result["tool_ids"],
            "rounds": result["rounds"],
        })
        comp = " ".join(f"{k[:4]}={v:.2f}" for k, v in scored.components.items())
        print(f"{task['task_id']:<10} {task['task_type']:<24} total={scored.total:+.3f}  {comp}")

    n = len(rows)
    mean_total = sum(r["total"] for r in rows) / n if n else 0.0
    comp_names = sorted({k for r in rows for k in r["components"]})
    comp_means = {k: round(sum(r["components"].get(k, 0.0) for r in rows) / n, 4) for k in comp_names}
    tool_rate = sum(1 for r in rows if r["components"].get("tool_use")) / n if n else 0.0
    grounded = [r for r in rows if r["details"]["cited"]]
    grounded_rate = (
        sum(len(r["details"]["grounded"]) / len(r["details"]["cited"]) for r in grounded) / len(grounded)
        if grounded else 0.0
    )
    halluc = sum(len(r["details"]["hallucinated"]) for r in rows)

    summary = {
        "model": args.model,
        "split": args.split,
        "few_shot": args.few_shot,
        "grounded": args.grounded,
        "adapter": args.adapter or None,
        "sample": args.sample,
        "tasks": n,
        "mean_total": round(mean_total, 4),
        "component_means": comp_means,
        "tool_call_rate": round(tool_rate, 4),
        "grounded_citation_rate": round(grounded_rate, 4),
        "hallucinated_ids": halluc,
    }
    print("\n=== summary ===")
    print(f"mean_total            {summary['mean_total']:+.4f}")
    for key, value in comp_means.items():
        print(f"  {key:<18} {value:.4f}")
    print(f"tool_call_rate        {summary['tool_call_rate']:.3f}")
    print(f"grounded_citation     {summary['grounded_citation_rate']:.3f}")
    print(f"hallucinated ids      {halluc}")

    out_path = Path(args.out) if args.out else RESULTS / f"score_{args.split}.json"
    out_path.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[report] {out_path}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
