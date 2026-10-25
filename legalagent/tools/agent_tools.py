"""Compact tool wrappers for RL training.

Why compact: TRL feeds the tool result back into the context and regenerates. A
full `search_statute` payload (3 articles x 220-char snippets + coverage block)
eats the whole token budget, TRL rolls it back, and the model never gets to see
a result — the reward then only sees a dangling tool call.

These wrappers keep the same semantics (same tool name family, same arguments)
but return only what the model needs: IDs, article numbers, a short snippet.
Results are JSON strings, not dicts, because TRL stringifies dicts with Python
`repr` (single quotes), which is not parseable JSON.
"""

from __future__ import annotations

import json

from .legal_tools import get_statute, search_statute


def search_compact(query: str, applicable_date: str, limit: int = 2) -> str:
    """检索劳动法条文，返回条文 ID、条号和简短摘要。

    Args:
        query: 中文检索词，例如"违法解除劳动合同 赔偿金"。
        applicable_date: 适用日期，格式 YYYY-MM-DD，必须由题目给出。
        limit: 返回条数，1 到 5，默认 2。

    Returns:
        JSON 字符串，包含 results 列表（article_id / display / snippet）。
    """
    try:
        count = max(1, min(int(limit or 2), 5))
    except (TypeError, ValueError):
        count = 2
    result = search_statute(query, applicable_date, count)
    if not result.get("ok"):
        return json.dumps(result, ensure_ascii=False)
    payload = {
        "ok": True,
        "results": [
            {
                "article_id": item["article_id"],
                "display": item["article_number_display"],
                "snippet": item["snippet"][:80],
            }
            for item in result.get("results", [])
        ],
    }
    return json.dumps(payload, ensure_ascii=False)


def get_compact(article_id: str, applicable_date: str) -> str:
    """读取指定条文的正文，用于引用前核对原文。

    Args:
        article_id: 形如 labor_contract_law:39 的条文 ID。
        applicable_date: 适用日期，格式 YYYY-MM-DD。

    Returns:
        JSON 字符串，包含 article_id / text（截断到 400 字）/ temporal_status。
    """
    result = get_statute(article_id, applicable_date)
    if not result.get("ok"):
        return json.dumps(result, ensure_ascii=False)
    article = result["article"]
    return json.dumps(
        {
            "ok": True,
            "article_id": article["article_id"],
            "text": article["text"][:400],
            "temporal_status": result.get("temporal_status"),
        },
        ensure_ascii=False,
    )
