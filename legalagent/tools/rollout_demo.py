"""Minimal Qwen3 tool-calling rollout against the local legal statute tools.

This is the "does the loop actually work" harness, not a trainer: it loads one
model, gives it two tools, runs up to N tool rounds, and prints the trace.

    python rollout_demo.py --self-test
    python rollout_demo.py --query "员工严重违反规章制度被辞退，公司要赔钱吗？" --date 2024-01-01

Design notes:
- `applicable_date` is supplied by the caller, never taken from the system clock,
  so a 2024 case is not silently judged against today's law.
- Tool results are compacted before they go back into the context: the model needs
  IDs, snippets, URLs and temporal status, not the store's internal bookkeeping.
- A tool error is fed back as data (ok=false), which is what lets the model recover
  or say "本地数据不足" instead of inventing a citation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from .tools_schema import TOOLS, as_json, dispatch

SYSTEM_PROMPT = """你是中国劳动法检索助手。回答前必须按以下纪律工作：
1. 涉及任何法条，先调用 search_statute 检索，再用 get_statute 读取完整正文；不得凭记忆引用法条。
2. 每次工具调用都必须传题目给出的 applicable_date，绝不使用今天日期。
3. 只引用工具实际返回过的 article_id 和原文；回答末尾用列表写出引用：<law_id:article_number>《法规名》第N条。
4. 如果工具返回 ok=false（例如 version_unavailable_for_date 表示本地没有那个日期的版本），
   如实说明“本地数据不足以判断该日期的适用文本”，不得改用其它版本替代。
5. 工具没检索到，不等于法律上没有该规则；措辞要区分这两种情况。"""

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Extract JSON tool calls; tolerate trailing prose and truncated JSON.

    Small models often stop mid-object when they hit the token limit, so a
    missing closing brace is repaired instead of dropping the call silently.
    """
    calls: list[dict[str, Any]] = []
    for raw in TOOL_CALL_RE.findall(text):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = _repair_json(raw)
            if payload is None:
                continue
        if isinstance(payload, dict) and isinstance(payload.get("name"), str):
            calls.append({"name": payload["name"], "arguments": payload.get("arguments") or {}})
    return calls


def _repair_json(raw: str) -> dict | None:
    """Close unbalanced braces / quotes, then retry json.loads."""
    depth = 0
    in_string = False
    escaped = False
    for char in raw:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        else:
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
    candidate = raw
    if in_string:
        candidate += '"'
    if depth > 0:
        candidate += "}" * depth
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def compact(name: str, result: dict[str, Any]) -> dict[str, Any]:
    """Trim a tool result to what the model needs, keeping every caveat."""
    if not result.get("ok"):
        return result
    if name == "search_statute":
        return {
            "ok": True,
            "applicable_date": result.get("applicable_date"),
            "results": [
                {
                    "article_id": item["article_id"],
                    "law_title": item["law_title"],
                    "article_number_display": item["article_number_display"],
                    "snippet": item["snippet"],
                    "effective_from": item["effective_from"],
                    "temporal_status": item["temporal_status"],
                    "source_url": item["source_url"],
                }
                for item in result.get("results", [])
            ],
            "warnings": result.get("warnings", []),
        }
    if name == "get_statute":
        article = result.get("article", {})
        return {
            "ok": True,
            "applicable_date": result.get("applicable_date"),
            "temporal_status": result.get("temporal_status"),
            "notes": result.get("notes", []),
            "article": {
                "article_id": article.get("article_id"),
                "law_title": article.get("law_title"),
                "article_number_display": article.get("article_number_display"),
                "part": article.get("part"),
                "text": article.get("text"),
                "effective_from": article.get("effective_from"),
                "effective_to": article.get("effective_to"),
                "source_url": article.get("source_url"),
                "text_sha256": article.get("text_sha256"),
            },
        }
    return result


FEW_SHOT: list[dict[str, Any]] = []  # kept for backwards import compatibility


def scripted_call(query: str, applicable_date: str) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """A real tool round with a synthetic assistant message, for plumbing tests."""
    arguments = {"query": query, "applicable_date": applicable_date, "limit": 3}
    content = f"<tool_call>\n{json.dumps({'name': 'search_statute', 'arguments': arguments}, ensure_ascii=False)}\n</tool_call>"
    result = dispatch("search_statute", arguments)
    payload = compact("search_statute", result)
    return arguments, content, payload


def load_model(model_id: str, local_files_only: bool, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        local_files_only=local_files_only,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    model.to(device).eval()
    return model, tokenizer


def build_few_shot() -> list[dict[str, Any]]:
    """One real, fully executed example round; nothing here is fabricated."""
    arguments, content, payload = scripted_call("违法解除劳动合同 赔偿金", "2024-01-01")
    hits = payload.get("results", [])
    if hits:
        top = hits[0]
        answer = (
            f"依据《{top['law_title']}》{top['article_number_display']}：{top['snippet'][:80]}… "
            f"引用：{top['article_id']}。"
        )
    else:
        answer = "本地库未检索到相关条文，不能据此断言法律上没有该规则。"
    return [
        {"role": "user", "content": "示例：员工在 2024-01-01 被违法辞退，能主张什么？"},
        {"role": "assistant", "content": content},
        {"role": "tool", "content": as_json(payload), "name": "search_statute"},
        {"role": "assistant", "content": answer},
    ]


def generate(model, tokenizer, messages: list[dict], max_new_tokens: int, enable_thinking: bool,
             sample: bool = False, temperature: float = 0.7, top_p: float = 0.9, seed: int = 0) -> str:
    import torch

    prompt = tokenizer.apply_chat_template(
        messages,
        tools=TOOLS,
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=enable_thinking,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens, "pad_token_id": tokenizer.eos_token_id}
    if sample:
        torch.manual_seed(seed)
        kwargs.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=50)
    else:
        kwargs.update(do_sample=False)
    with torch.no_grad():
        output = model.generate(**inputs, **kwargs)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def _ids_from_payload(name: str, payload: dict[str, Any]) -> list[str]:
    if name == "search_statute":
        return [item["article_id"] for item in payload.get("results", []) if item.get("article_id")]
    if name == "get_statute":
        article_id = payload.get("article", {}).get("article_id")
        return [article_id] if article_id else []
    return []


def converse(model, tokenizer, messages: list[dict[str, Any]], args, verbose: bool = True) -> dict[str, Any]:
    """Run the generate → tool → generate loop; returns answer plus every tool id seen."""
    trace: list[dict[str, Any]] = []
    tool_ids: list[str] = []
    answer = ""

    for round_index in range(1, args.max_rounds + 1):
        reply = generate(
            model, tokenizer, messages, args.max_new_tokens, args.enable_thinking,
            sample=args.sample, temperature=args.temperature, seed=args.seed + round_index,
        )
        messages.append({"role": "assistant", "content": reply})
        answer = reply
        if verbose:
            print(f"\n=== round {round_index}: assistant ===")
            print(reply.strip()[: args.print_chars])

        calls = parse_tool_calls(reply)
        if not calls:
            trace.append({"round": round_index, "assistant": reply, "tool_calls": []})
            if verbose:
                print("\n[stop] no tool call -> final answer")
            break

        round_calls = []
        for call in calls:
            result = dispatch(call["name"], call["arguments"])
            payload = compact(call["name"], result)
            ids = _ids_from_payload(call["name"], payload) if result.get("ok") else []
            tool_ids.extend(ids)
            messages.append({"role": "tool", "content": as_json(payload), "name": call["name"]})
            round_calls.append(
                {"name": call["name"], "arguments": call["arguments"], "ok": bool(result.get("ok")), "ids": ids}
            )
            if verbose:
                print(f"\n--- tool: {call['name']} ok={result.get('ok')} args={json.dumps(call['arguments'], ensure_ascii=False)}")
                print(as_json(payload)[: args.print_chars])
        trace.append({"round": round_index, "assistant": reply, "tool_calls": round_calls})

    return {
        "messages": messages,
        "trace": trace,
        "answer": answer,
        "tool_ids": sorted(set(tool_ids)),
        "rounds": len(trace),
    }


def run(query: str, applicable_date: str, args) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    if args.few_shot:
        messages.extend(build_few_shot())
    messages.append({"role": "user", "content": f"案情与问题：{query}\n\n适用日期：{applicable_date}"})

    if args.scripted:
        arguments, content, payload = scripted_call(query, applicable_date)
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "tool", "content": as_json(payload), "name": "search_statute"})
        print("[scripted] injected one search_statute call to test the plumbing")

    model, tokenizer = load_model(args.model, not args.download, args.device)
    print(f"[model] {args.model} on {args.device}", flush=True)

    result = converse(model, tokenizer, messages, args, verbose=True)

    if args.trace_out:
        Path(args.trace_out).write_text(json.dumps(result["trace"], ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[trace] {args.trace_out}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--query", default="员工严重违反公司规章制度被辞退，公司需要支付赔偿金吗？")
    parser.add_argument("--date", default="2024-01-01", help="applicable_date，必须由题目给出")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--enable-thinking", action="store_true", help="开启 Qwen3 thinking（更慢）")
    parser.add_argument("--few-shot", action="store_true", help="在上下文里放一个真实执行的示例轮次")
    parser.add_argument("--scripted", action="store_true", help="注入一次真实工具调用，只测工具链路与回填")
    parser.add_argument("--sample", action="store_true", help="采样解码（默认贪心）")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--download", action="store_true", help="允许联网下载模型（默认只用本地缓存）")
    parser.add_argument("--print-chars", type=int, default=1200)
    parser.add_argument("--trace-out", default="")
    parser.add_argument("--self-test", action="store_true", help="不加载模型，只测工具链路")
    args = parser.parse_args()

    if args.self_test:
        result = dispatch("search_statute", {"query": "违法解除劳动合同 赔偿金", "applicable_date": args.date, "limit": 3})
        print(f"search ok={result.get('ok')} results={len(result.get('results', []))}")
        if result.get("results"):
            top = result["results"][0]
            print("  top:", top["article_id"], top["temporal_status"])
            detail = dispatch("get_statute", {"article_id": top["article_id"], "applicable_date": args.date})
            print(f"get ok={detail.get('ok')} chars={len(detail.get('article', {}).get('text', ''))}")
        return 0

    run(args.query, args.date, args)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
