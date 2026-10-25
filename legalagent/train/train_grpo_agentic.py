"""Agentic GRPO: the model calls the tools itself, reward reads the real trace.

Differences from `train_grpo.py` (v0, single-turn grounded):
- no pre-retrieved candidates are injected; the model must call search/get itself
- tools are the compact wrappers (full payloads get rolled back by TRL)
- the reward parses the TRL message trace: tool messages -> article ids,
  final assistant message -> answer

    python -m legalagent.train.train_grpo_agentic --max-steps 1 --limit 2   # plumbing
    python -m legalagent.train.train_grpo_agentic --max-steps 30            # real run
    python -m legalagent.train.train_grpo_agentic --curriculum data/tasks/curriculum_train.jsonl \
        --adapter runs/grpo_agentic --out runs/grpo_agentic_curr             # next curriculum stage
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..data.curriculum import load_curriculum
from ..model_io import check_adapter_dir, load_adapter_model, load_model, load_tokenizer
from ..paths import RUNS, TASKS
from ..reward import RewardConfig, score_rollout
from ..tools.agent_tools import get_compact, search_compact

SYSTEM = (
    "你是中国劳动法检索助手。工作纪律：\n"
    "1. 回答任何法律问题前，必须先调用 search_compact 检索条文；对关键条文再用 get_compact 读全文。\n"
    "2. 只依据工具返回的条文作答，引用必须来自工具返回的 article_id，不得编造。\n"
    "3. 回答给出结论 + 必要条件 + 条文 ID，控制在 200 字以内。\n"
    "4. 事实不足时明确指出缺什么，不要硬下结论。"
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_trace(conv) -> tuple[str, list[dict], list[str]]:
    """Return (final answer, tool calls, article ids seen in tool results)."""
    answer = ""
    calls: list[dict] = []
    ids: list[str] = []
    for message in conv if isinstance(conv, list) else []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            for tc in message.get("tool_calls") or []:
                fn = tc.get("function") or {}
                calls.append({"name": fn.get("name"), "arguments": fn.get("arguments") or {}, "ok": True})
            if (message.get("content") or "").strip():
                answer = message["content"]
        elif role == "tool":
            payload = message.get("content")
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:
                    payload = None
            if isinstance(payload, dict):
                if payload.get("ok") and isinstance(payload.get("results"), list):
                    ids.extend(r["article_id"] for r in payload["results"] if r.get("article_id"))
                elif payload.get("article_id"):
                    ids.append(payload["article_id"])
                elif payload.get("ok") is False and calls:
                    calls[-1]["ok"] = False
    return answer, calls, sorted(set(ids))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--max-completion-length", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--out", default=str(RUNS / "grpo_agentic"))
    parser.add_argument("--curriculum", default="",
                        help="bad_case_sampler 产出的 prompt-only JSONL；给了就只训这些题（可重复）")
    parser.add_argument("--adapter", default="",
                        help="上一阶段的 LoRA adapter 目录；给了就从它续训，而不是从基座重建 LoRA")
    parser.add_argument("--download", action="store_true",
                        help="允许联网下载模型（离线环境下不要加）")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    adapter_cfg = None
    if args.adapter:
        adapter_cfg = check_adapter_dir(args.adapter, args.model)
        if Path(args.out).resolve() == Path(args.adapter).resolve():
            raise SystemExit(
                "--out 不能与 --adapter 相同：会原地覆盖上一阶段 adapter，训练记录无法回溯。"
            )

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    if args.curriculum:
        tasks = load_curriculum(args.curriculum, limit=args.limit or None)
    else:
        tasks = read_jsonl(TASKS / "tasks_train_labeled.jsonl")
        if args.limit:
            tasks = tasks[: args.limit]
    by_id = {t["task_id"]: t for t in tasks}

    tokenizer = load_tokenizer(args.model, download=args.download)
    rows = [{
        "prompt": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{t['prompt']}\n\n适用日期：{t['applicable_date']}"},
        ],
        "task_id": t["task_id"],
    } for t in tasks]
    dataset = Dataset.from_list(rows)
    print(f"[data] {len(dataset)} agentic prompts")

    model = load_model(args.model, download=args.download, dtype=torch.bfloat16)
    if args.adapter:
        model = load_adapter_model(model, args.adapter)
        print(f"[adapter] 从 {args.adapter} 续训（底座 {adapter_cfg.get('base_model_name_or_path', args.model)}）")

    cfg = RewardConfig()
    stats: list[dict] = []

    def reward_agentic(completions, task_id=None, **kwargs):
        scores = []
        for conv, tid in zip(completions, task_id):
            answer, calls, ids = parse_trace(conv)
            task = by_id[tid]
            breakdown = score_rollout(task, {"answer": answer, "tool_calls": calls, "tool_ids": ids}, cfg)
            scores.append(breakdown.total)
            stats.append({
                "task_id": tid,
                "total": round(breakdown.total, 4),
                **{k: round(v, 4) for k, v in breakdown.components.items()},
                "tool_ids": ids[:6],
                "cited": breakdown.details["cited"],
                "answer_chars": len(answer),
            })
        return scores

    # Continuing a saved adapter: pass the PeftModel itself and no peft_config
    # (TRL rejects both together). A fresh run keeps the original recipe.
    lora = None
    if not args.adapter:
        lora = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
    grpo_args = GRPOConfig(
        output_dir=args.out,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_tool_calling_iterations=2,
        learning_rate=args.lr,
        beta=0.04,
        temperature=0.9,
        logging_steps=1,
        save_steps=max(args.max_steps, 1),
        max_steps=args.max_steps,
        report_to="none",
        bf16=True,
        gradient_checkpointing=True,
        use_vllm=False,
        optim="adamw_torch",
        chat_template_kwargs={"enable_thinking": False},
    )

    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        grpo_dict = grpo_args.to_dict()
    except Exception:  # noqa: BLE001 - 记录配置失败不应该中断训练
        grpo_dict = {}
    (run_dir / "config.json").write_text(
        json.dumps({
            "cli": vars(args),
            "grpo": grpo_dict,
            "lineage": {
                "base_model": args.model,
                "continued_from": str(Path(args.adapter).resolve()) if args.adapter else None,
                "adapter_base_model": (adapter_cfg or {}).get("base_model_name_or_path"),
            },
        }, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    trainer = GRPOTrainer(
        model=model, args=grpo_args, processing_class=tokenizer,
        reward_funcs=[reward_agentic], train_dataset=dataset,
        tools=[search_compact, get_compact], peft_config=lora,
    )
    trainer.train()
    # `save_steps` only writes checkpoint-<n>/adapter_*; the run root must also
    # be a loadable adapter, because the next curriculum round uses it as
    # --adapter (and eval/collect point at runs/<name>).
    trainer.save_model(args.out)

    if stats:
        n = len(stats)
        mean_total = sum(s["total"] for s in stats) / n
        print(f"\n[reward] samples={n} mean_total={mean_total:.4f}")
        for key in ("tool_use", "grounding", "gold", "answer_points"):
            values = [s.get(key, 0.0) for s in stats]
            print(f"  {key:<13} mean={sum(values)/n:.4f}")
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "reward_stats.json").write_text(
            json.dumps({"mean_total": mean_total, "samples": stats}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(f"[done] adapter under {args.out}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
