"""Tool schemas + dispatcher for the legal agent.

The schemas are the contract shown to the model (Qwen3 chat template renders them
as JSON inside <tools>...</tools>); `dispatch` is the only place tool names map to
real Python functions, so a hallucinated tool name fails loudly instead of being
silently ignored.

Run `python -m legalagent.tools.tools_schema` for a dependency-free self-test.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from .legal_tools import get_statute, search_statute

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_statute",
            "description": (
                "在本地劳动法条库中检索与问题相关的条文。返回候选条文的稳定 ID、"
                "摘要、版本与来源链接。必须传入案件适用的日期，用于判断该日期下有效的版本。"
                "这只是候选，定论前必须再用 get_statute 读取该条的完整正文。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "中文法律检索词，用争议焦点与关键事实描述，例如“严重违反规章制度 解除劳动合同”。",
                    },
                    "applicable_date": {
                        "type": "string",
                        "description": "案件事实对应的适用日期，格式 YYYY-MM-DD。不要使用今天日期，必须由题目给出。",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "返回条数，1-20，默认 5。",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 5,
                    },
                },
                "required": ["query", "applicable_date"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_statute",
            "description": (
                "按条文 ID 读取完整正文、文本哈希、来源链接与版本元信息。"
                "article_id 必须来自 search_statute 的结果，不要自行编造。"
                "回答中的引用必须使用这里返回的原文。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "article_id": {
                        "type": "string",
                        "description": "形如 labor_contract_law:39 的稳定条文 ID（law_id:article_number）。",
                    },
                    "applicable_date": {
                        "type": "string",
                        "description": "与检索时相同的适用日期，格式 YYYY-MM-DD。",
                    },
                },
                "required": ["article_id", "applicable_date"],
                "additionalProperties": False,
            },
        },
    },
]

_FUNCTIONS: dict[str, Callable[..., dict]] = {
    "search_statute": search_statute,
    "get_statute": get_statute,
}


def dispatch(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute one tool call and always return a JSON-serialisable dict."""
    if name not in _FUNCTIONS:
        return {"ok": False, "error": "unknown_tool", "message": f"未知工具 {name!r}，可用：{sorted(_FUNCTIONS)}"}
    if not isinstance(arguments, dict):
        return {"ok": False, "error": "bad_arguments", "message": "arguments 必须是 JSON 对象"}
    try:
        return _FUNCTIONS[name](**arguments)
    except TypeError as exc:
        return {"ok": False, "error": "bad_arguments", "message": str(exc)}
    except Exception as exc:  # tool bugs must not crash the rollout
        return {"ok": False, "error": type(exc).__name__, "message": str(exc)}


def as_json(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


if __name__ == "__main__":
    demo_calls = [
        ("search_statute", {"query": "严重违反规章制度 解除劳动合同", "applicable_date": "2024-01-01", "limit": 3}),
        ("get_statute", {"article_id": "labor_contract_law:39", "applicable_date": "2024-01-01"}),
        ("search_statute", {"query": "x"}),  # missing required arg
        ("no_such_tool", {}),
    ]
    for name, args in demo_calls:
        result = dispatch(name, args)
        print(f"[{name}] ok={result.get('ok')} error={result.get('error')}")
        if result.get("ok") and "results" in result:
            for hit in result["results"]:
                print("   ", hit["article_id"], round(hit["score"], 3), hit["temporal_status"])
        elif result.get("ok"):
            print("   ", result["article"]["article_id"] if "article" in result else "", result.get("temporal_status"))
        print("   ", as_json(result)[:160])
