"""Stage-1 wiring probe: does TRL hand the tool trace back to the reward?

Runs ONE GRPO step with `tools=[search_compact, get_compact]` and prints what the
reward function actually receives. Three questions:

1. do the model's tool calls reach the reward? (visibility)
2. do the tool *responses* reach it, or do they get rolled back for length?
3. does the final answer survive as the last assistant message?

    python -m legalagent.train.probe_tool_flow
"""

from __future__ import annotations

import argparse
import json

from datasets import Dataset

from ..model_io import load_model, load_tokenizer
from ..paths import RUNS, TASKS
from ..tools.legal_tools import get_statute, search_statute

SYSTEM = "你是劳动法检索助手。先调用工具检索法条，再依据条文作答，并在结尾给出条文 ID。"


def search_compact(query: str, applicable_date: str, limit: int = 2) -> dict:
    """检索条文（训练用精简版）：只返回 ID、条号和 80 字摘要，减少上下文占用。

    Args:
        query: 中文检索词。
        applicable_date: 适用日期 YYYY-MM-DD。
        limit: 返回条数，1-5。
    """
    result = search_statute(query, applicable_date, min(int(limit or 2), 5))
    if not result.get("ok"):
        return result
    return {
        "ok": True,
        "results": [
            {"article_id": r["article_id"], "display": r["article_number_display"],
             "snippet": r["snippet"][:80]}
            for r in result.get("results", [])
        ],
    }


def get_compact(article_id: str, applicable_date: str) -> dict:
    """读取条文完整正文（训练用精简版，去掉元数据字段）。

    Args:
        article_id: 形如 labor_contract_law:39 的条文 ID。
        applicable_date: 适用日期 YYYY-MM-DD。
    """
    result = get_statute(article_id, applicable_date)
    if not result.get("ok"):
        return result
    article = result["article"]
    return {"ok": True, "article_id": article["article_id"],
            "text": article["text"][:400], "temporal_status": result["temporal_status"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--download", action="store_true",
                        help="允许联网下载模型（离线环境下不要加）")
    args = parser.parse_args()

    import torch
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    tasks = [json.loads(line) for line in (TASKS / "tasks_train_labeled.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()][:2]
    tokenizer = load_tokenizer(args.model, download=args.download)

    prompts = [[
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{task['prompt']}\n\n适用日期：{task['applicable_date']}"},
    ] for task in tasks]

    dataset = Dataset.from_list([{"prompt": p, "task_id": t["task_id"]} for p, t in zip(prompts, tasks)])
    print(f"[data] {len(dataset)} conversational prompts")

    model = load_model(args.model, download=args.download, dtype=torch.bfloat16)

    seen: list[dict] = []

    def reward_probe(completions, task_id=None, **kwargs):
        for conv in completions:
            if isinstance(conv, list):
                roles = [m.get("role") for m in conv]
                calls = [tc.get("function", {}).get("name")
                         for m in conv if isinstance(m, dict)
                         for tc in (m.get("tool_calls") or [])]
                last = conv[-1].get("content") or ""
            else:
                roles, calls, last = ["<str>"], [], conv
            seen.append({"roles": roles, "calls": calls, "last_chars": len(last),
                         "first": conv if isinstance(conv, str) else conv})
        return [1.0 for _ in completions]

    lora = LoraConfig(r=8, lora_alpha=16, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    args = GRPOConfig(
        output_dir=str(RUNS / "probe_tool_flow"),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_generations=2,
        max_completion_length=1024,
        max_tool_calling_iterations=2,
        max_steps=1,
        logging_steps=1,
        report_to="none",
        bf16=True,
        use_vllm=False,
        temperature=0.9,
        chat_template_kwargs={"enable_thinking": False},
    )

    trainer = GRPOTrainer(
        model=model, args=args, processing_class=tokenizer,
        reward_funcs=[reward_probe], train_dataset=dataset,
        tools=[search_compact, get_compact], peft_config=lora,
    )
    trainer.train()

    print("\n=== what the reward saw ===")
    for i, item in enumerate(seen[:4]):
        print(f"\n--- completion {i}: roles={item['roles']} calls={item['calls']} last_chars={item['last_chars']}")
        print(json.dumps(item["first"], ensure_ascii=False)[:700])
    has_call = any(item["calls"] for item in seen)
    has_tool_msg = any("tool" in item["roles"] for item in seen)
    print(f"\n[verdict] tool_calls={has_call} | tool_response_msgs={has_tool_msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
