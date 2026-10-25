"""Demonstrate the two tools without model/API/GPU.

    python -m legalagent.tools.demo
"""
import json

from .legal_tools import get_statute, search_statute


def show(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    print("== 1. search_statute ==")
    found = search_statute("严重违反规章制度 解除劳动合同", "2024-01-01", limit=3)
    show(found)
    if found["results"]:
        print("\n== 2. get_statute (take returned article_id) ==")
        show(get_statute(found["results"][0]["article_id"], "2024-01-01"))
    print("\n== Historical version unavailable ==")
    show(get_statute("labor_contract_law:39", "2010-01-01"))
