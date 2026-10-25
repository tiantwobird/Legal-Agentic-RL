"""Baseline: how often does the untrained model actually use the tools?

This is the "before" measurement for the RL roadmap (step 3/4): load one model,
run a fixed query set through the same loop as `rollout_demo.py`, and score

- tool_call_rate      : did the model emit at least one <tool_call>?
- grounded_citation_rate : are the cited article_ids actually in tool output?
- hallucinated_ids    : cited ids the tools never returned (must be 0 after training)

The `expect` field is for reporting only; it is never put into the prompt.

    python -m legalagent.tools.baseline_toolcalling --model Qwen/Qwen3-1.7B --few-shot
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from ..paths import RESULTS
from .rollout_demo import (
    SYSTEM_PROMPT,
    build_few_shot,
    compact,
    generate,
    load_model,
    parse_tool_calls,
)
from .tools_schema import as_json, dispatch

CITE_RE = re.compile(r"([a-z][a-z0-9_]*)\s*:\s*(\d+)")

QUERIES: list[dict[str, str]] = [
    {"q": "员工严重违反公司规章制度被辞退，公司需要支付赔偿金吗？", "date": "2024-01-01", "expect": "labor_contract_law:39|labor_contract_law:87"},
    {"q": "公司没签书面劳动合同，员工能主张双倍工资吗？", "date": "2024-01-01", "expect": "labor_contract_law:82"},
    {"q": "劳动仲裁申请时效是多久？", "date": "2024-01-01", "expect": "labor_dispute_arbitration_law:27"},
    {"q": "劳动合同到期不续签，经济补偿怎么算？", "date": "2024-01-01", "expect": "labor_contract_law:46|labor_contract_law:47"},
    {"q": "公司安排周末加班，加班费按什么标准支付？", "date": "2024-01-01", "expect": "labor_law:44"},
    {"q": "试用期内公司可以随便辞退员工吗？", "date": "2024-01-01", "expect": "labor_contract_law:21|labor_contract_law:39"},
]


def one_round(model, tokenizer, messages, args) -> tuple[str, list[dict]]:
    reply = generate(
        model, tokenizer, messages, args.max_new_tokens, args.enable_thinking,
        sample=args.sample, temperature=args.temperature, seed=args.seed,
    )
    messages.append({"role": "assistant", "content": reply})
    tool_rows = []
    for call in parse_tool_calls(reply):
        result = dispatch(call["name"], call["arguments"])
        payload = compact(call["name"], result)
        messages.append({"role": "tool", "content": as_json(payload), "name": call["name"]})
        ids = []
        if call["name"] == "search_statute":
            ids = [x["article_id"] for x in payload.get("results", [])]
        elif call["name"] == "get_statute":
            ids = [payload.get("article", {}).get("article_id")]
        tool_rows.append({"name": call["name"], "ok": result.get("ok"), "ids": [i for i in ids if i]})
    return reply, tool_rows


def main() -> int:
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--out", default=str(RESULTS / "baseline_report.json"))
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, not args.download, args.device)
    print(f"[model] {args.model} on {args.device}\n")

    rows = []
    for item in QUERIES:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if args.few_shot:
            messages.extend(build_few_shot())
        messages.append({"role": "user", "content": f"案情与问题：{item['q']}\n\n适用日期：{item['date']}"})

        all_ids: list[str] = []
        used_tool = False
        final = ""
        for _ in range(args.max_rounds):
            reply, tool_rows = one_round(model, tokenizer, messages, args)
            final = reply
            if not tool_rows:
                break
            used_tool = True
            for row in tool_rows:
                all_ids.extend(row["ids"])

        cited = [f"{law}:{num}" for law, num in CITE_RE.findall(final)]
        grounded = [c for c in cited if c in all_ids]
        hallucinated = [c for c in cited if c not in all_ids]
        row = {
            "query": item["q"],
            "date": item["date"],
            "tool_called": used_tool,
            "tool_ids": sorted(set(all_ids)),
            "cited_ids": cited,
            "grounded_citations": grounded,
            "hallucinated_citations": hallucinated,
            "answer": final.strip(),
        }
        rows.append(row)
        flag = "TOOL" if used_tool else "no-tool"
        print(f"[{flag:7}] grounded={len(grounded)}/{len(cited)} halluc={len(hallucinated)}  {item['q'][:28]}")
        if cited:
            print(f"          cited: {cited}")

    total = len(rows)
    with_tool = sum(1 for r in rows if r["tool_called"])
    cited_total = sum(len(r["cited_ids"]) for r in rows)
    grounded_total = sum(len(r["grounded_citations"]) for r in rows)
    halluc_total = sum(len(r["hallucinated_citations"]) for r in rows)
    report = {
        "model": args.model,
        "few_shot": args.few_shot,
        "sample": args.sample,
        "queries": total,
        "tool_call_rate": round(with_tool / total, 3),
        "cited_total": cited_total,
        "grounded_total": grounded_total,
        "grounded_citation_rate": round(grounded_total / cited_total, 3) if cited_total else 0.0,
        "hallucinated_total": halluc_total,
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== summary ===")
    print(f"tool_call_rate        {with_tool}/{total}  ({report['tool_call_rate']})")
    print(f"grounded_citation_rate {grounded_total}/{cited_total}  ({report['grounded_citation_rate']})")
    print(f"hallucinated ids      {halluc_total}")
    print(f"[report] {args.out}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
