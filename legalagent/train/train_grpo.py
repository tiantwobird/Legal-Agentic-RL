"""GRPO smoke test: does the policy actually learn from the legal reward?

Scope note: this v0 script trains a **single-turn grounded** task (retrieved
articles are injected into the prompt) because that keeps the token accounting
exact. Multi-turn tool calling needs TRL's `tools=`/`rollout_func` and masked
environment tokens; that is `train_grpo_agentic.py`, not this file.

    python -m legalagent.train.train_grpo --max-steps 1 --limit 4   # plumbing check
    python -m legalagent.train.train_grpo --max-steps 30            # real run

Everything stays on one 16 GB GPU: Qwen3-1.7B in bf16 + LoRA r=16, 4 generations
per prompt, 192 completion tokens.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..data.curriculum import load_curriculum
from ..model_io import check_adapter_dir, load_adapter_model, load_model, load_tokenizer
from ..paths import RUNS, TASKS
from ..reward import RewardConfig, score_rollout
from ..tools.legal_tools import search_statute

SYSTEM = (
    "你是中国劳动法检索助手。下面会给出本地法规库检索到的候选条文（含 article_id）。"
    "请只依据这些条文回答问题，并给出结论、必要条件与条文ID；不得引用候选之外的条文，不得编造。"
    "回答控制在 150 字以内，先写结论，最后一行写引用。"
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_rows(tasks: list[dict], limit: int | None) -> list[dict]:
    if limit:
        tasks = tasks[:limit]
    rows = []
    for task in tasks:
        hits = search_statute(task["prompt"], task["applicable_date"], limit=3)
        articles = hits.get("results", [])
        block = "\n\n".join(
            f"[{a['article_id']}]《{a['law_title']}》{a['article_number_display']}\n{a['snippet']}"
            for a in articles
        )
        rows.append({
            "task_id": task["task_id"],
            "candidate_ids": [a["article_id"] for a in articles],
            "user": f"{task['prompt']}\n\n适用日期：{task['applicable_date']}\n\n【本地检索到的候选条文】\n{block}",
        })
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=320)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--out", default=str(RUNS / "grpo_single"))
    parser.add_argument("--curriculum", default="",
                        help="bad_case_sampler 产出的 prompt-only JSONL；给了就只训这些题（可重复）")
    parser.add_argument("--adapter", default="",
                        help="上一阶段的 LoRA adapter 目录；给了就从它续训，而不是从基座重建 LoRA")
    parser.add_argument("--download", action="store_true")
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
        rows = build_rows(tasks, None)
    else:
        tasks = read_jsonl(TASKS / "tasks_train_labeled.jsonl")
        rows = build_rows(tasks, args.limit or None)
    by_id = {t["task_id"]: t for t in tasks}

    tokenizer = load_tokenizer(args.model, download=args.download)
    prompts = []
    for row in rows:
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": row["user"]}]
        prompts.append(tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False))

    dataset = Dataset.from_list([
        {"prompt": prompt, "task_id": row["task_id"], "candidate_ids": row["candidate_ids"]}
        for prompt, row in zip(prompts, rows)
    ])
    print(f"[data] {len(dataset)} prompts, avg chars {sum(map(len, prompts)) // len(prompts)}")

    model = load_model(args.model, download=args.download, dtype=torch.bfloat16)
    if args.adapter:
        model = load_adapter_model(model, args.adapter)
        print(f"[adapter] 从 {args.adapter} 续训（底座 {adapter_cfg.get('base_model_name_or_path', args.model)}）")

    # tool_use is pre-executed in this variant, so its weight is zeroed out.
    cfg = RewardConfig(w_tool_use=0.0, w_grounding=0.15, w_gold=0.35, w_points=0.40)
    stats: list[dict] = []

    def reward_legal(completions, task_id=None, candidate_ids=None, **kwargs):
        scores = []
        for text, tid, ids in zip(completions, task_id, candidate_ids):
            task = by_id[tid]
            rollout = {
                "answer": text,
                "tool_ids": list(ids),
                "tool_calls": [{"name": "search_statute", "ok": True}],
            }
            breakdown = score_rollout(task, rollout, cfg)
            scores.append(breakdown.total)
            stats.append({"task_id": tid, "total": breakdown.total, **breakdown.components})
        return scores

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
        mask_truncated_completions=True,
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
        log_completions=False,
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
        model=model,
        args=grpo_args,
        processing_class=tokenizer,
        reward_funcs=[reward_legal],
        train_dataset=dataset,
        peft_config=lora,
    )
    trainer.train()
    # `save_steps` only writes checkpoint-<n>/adapter_*; the run root must also
    # be a loadable adapter, because the next curriculum round uses it as
    # --adapter (and eval/collect point at runs/<name>).
    trainer.save_model(args.out)

    n = len(stats)
    if n:
        mean_total = sum(s["total"] for s in stats) / n
        print(f"\n[reward] samples={n} mean_total={mean_total:.4f}")
        for key in ("gold", "grounding", "answer_points"):
            values = [s.get(key, 0.0) for s in stats]
            print(f"  {key:<14} mean={sum(values)/n:.4f}")
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "reward_stats.json").write_text(
            json.dumps({"mean_total": mean_total, "samples": stats}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(f"[done] adapter saved under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
