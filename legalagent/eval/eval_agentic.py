"""Evaluate a model on the agentic task with the same tools used in training.

The trained adapter only knows the compact tool names (`search_compact` /
`get_compact`) and the agentic system prompt, so measuring it with the old
`search_statute` tool set would be unfair and would mostly measure tool-name
confusion. This evaluator mirrors training: same prompt, same tools, same loop.

    python -m legalagent.eval.eval_agentic --split dev --out results/eval_agentic_before.json
    python -m legalagent.eval.eval_agentic --split dev --adapter runs/grpo_agentic --out results/eval_agentic_after.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..model_io import check_adapter_dir, load_adapter_model, load_model, load_tokenizer
from ..paths import RESULTS, TASKS
from ..reward import RewardConfig, score_rollout
from ..tools.agent_tools import get_compact, search_compact
from ..tools.rollout_demo import parse_tool_calls
from ..train.train_grpo_agentic import SYSTEM, parse_trace

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "search_compact",
        "description": "检索劳动法条文，返回条文 ID、条号和简短摘要。",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "中文检索词"},
            "applicable_date": {"type": "string", "description": "YYYY-MM-DD"},
            "limit": {"type": "integer", "description": "1-5，默认 2", "default": 2},
        }, "required": ["query", "applicable_date"], "additionalProperties": False}}},
    {"type": "function", "function": {
        "name": "get_compact",
        "description": "读取指定条文的正文，用于核对引用。",
        "parameters": {"type": "object", "properties": {
            "article_id": {"type": "string", "description": "形如 labor_contract_law:39"},
            "applicable_date": {"type": "string", "description": "YYYY-MM-DD"},
        }, "required": ["article_id", "applicable_date"], "additionalProperties": False}}},
]

FUNCS = {"search_compact": search_compact, "get_compact": get_compact}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_task(model, tokenizer, task, max_new_tokens: int, max_rounds: int,
             sample: bool = False, temperature: float = 0.9) -> dict:
    """Run one task end to end.

    `sample=False` (evaluation) is greedy. `sample=True` (curriculum collection)
    draws with temperature, because the bad-case sampler needs *within-task*
    reward variance — greedy decoding gives G identical rollouts and zero signal.
    """
    import torch

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{task['prompt']}\n\n适用日期：{task['applicable_date']}"},
    ]
    transcript = []
    for _ in range(max_rounds):
        text = tokenizer.apply_chat_template(
            messages, tools=TOOL_SCHEMAS, add_generation_prompt=True,
            tokenize=False, enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        gen_kwargs = {"max_new_tokens": max_new_tokens,
                      "pad_token_id": tokenizer.eos_token_id,
                      "do_sample": sample}
        if sample:
            gen_kwargs.update(temperature=temperature, top_p=0.95)
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        reply = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        messages.append({"role": "assistant", "content": reply})
        entry = {"role": "assistant", "content": reply, "tool_calls": []}
        transcript.append(entry)

        calls = [c for c in _parse(reply)]
        if not calls:
            break
        for call in calls:
            fn = FUNCS.get(call["name"])
            payload = fn(**(call["arguments"] or {})) if fn else json.dumps({"ok": False, "error": "unknown_tool"})
            messages.append({"role": "tool", "content": payload, "name": call["name"]})
            transcript.append({"role": "tool", "content": payload, "name": call["name"]})
            entry["tool_calls"].append({"name": call["name"], "ok": True, "arguments": call["arguments"]})

    answer, calls, ids = parse_trace(transcript)
    return {"answer": answer, "tool_calls": calls, "tool_ids": ids}


def _parse(text: str) -> list[dict]:
    return parse_tool_calls(text)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test", "badcase"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--download", action="store_true",
                        help="允许联网下载模型（离线环境下不要加）")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model, download=args.download)
    model = load_model(args.model, download=args.download, dtype="auto").eval()
    if args.adapter:
        check_adapter_dir(args.adapter, args.model)
        model = load_adapter_model(model, args.adapter, trainable=False).eval()
        print(f"[adapter] {args.adapter}")
    model.to("cuda" if _cuda() else "cpu")

    tasks = read_jsonl(TASKS / f"tasks_{args.split}_labeled.jsonl")
    if args.limit:
        tasks = tasks[: args.limit]

    cfg = RewardConfig()
    rows = []
    print(f"[eval] split={args.split} tasks={len(tasks)} adapter={bool(args.adapter)}\n")
    for task in tasks:
        rollout = run_task(model, tokenizer, task, args.max_new_tokens, args.max_rounds)
        breakdown = score_rollout(task, rollout, cfg)
        rows.append({
            "task_id": task["task_id"], "task_type": task["task_type"],
            "total": round(breakdown.total, 4), "components": breakdown.components,
            "details": breakdown.details, "answer": rollout["answer"],
            "tool_ids": rollout["tool_ids"], "tool_calls": rollout["tool_calls"],
        })
        comp = " ".join(f"{k[:4]}={v:.2f}" for k, v in breakdown.components.items())
        print(f"{task['task_id']:<10} total={breakdown.total:+.3f}  {comp}")

    n = len(rows)
    mean_total = sum(r["total"] for r in rows) / n if n else 0.0
    comp_names = sorted({k for r in rows for k in r["components"]})
    summary = {
        "model": args.model, "adapter": args.adapter or None, "split": args.split, "tasks": n,
        "mean_total": round(mean_total, 4),
        "component_means": {k: round(sum(r["components"].get(k, 0.0) for r in rows) / n, 4) for k in comp_names},
        "tool_call_rate": round(sum(1 for r in rows if r["components"].get("tool_use")) / n, 4) if n else 0.0,
    }
    print("\n=== summary ===")
    print(f"mean_total {summary['mean_total']:+.4f}")
    for k, v in summary["component_means"].items():
        print(f"  {k:<13} {v:.4f}")
    out = Path(args.out) if args.out else RESULTS / f"eval_agentic_{args.split}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[report] {out}")
    return 0


def _cuda() -> bool:
    import torch

    return torch.cuda.is_available()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
