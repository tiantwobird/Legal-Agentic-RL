"""Turn eval artifacts into a bad-case report with actionable causes.

Inputs (all optional, use what exists), under results/:
  score_dev.json                    agentic dev run (10 tasks)
  baseline_report.json              6 open-ended baseline queries
  eval_grounded_before/after.json   grounded single-turn before/after

Output:
  bad_case_report.md   human-readable, one section per failure class
  bad_cases.json       machine-readable, for the next training round

    python -m legalagent.analysis.bad_case_analyzer
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ..paths import RESULTS, TASKS

CLASS_RULES = [
    ("no_tool", "没调用工具，凭记忆作答"),
    ("no_recall", "调了工具但没有拿到候选条文"),
    ("hallucinated", "引用了工具从未返回的 article_id"),
    ("wrong_article", "引用的条文真实存在，但不是本题的 gold"),
    ("missing_points", "答到了方向，但漏掉关键要点"),
    ("insufficient_handling", "信息不足的题没有指出缺失事实"),
]


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def classify(task: dict | None, row: dict) -> tuple[str, str]:
    """Return (primary_class, one-line evidence)."""
    comp = row.get("components", {})
    details = row.get("details", {})
    cited = details.get("cited") or row.get("cited_ids") or []
    grounded = details.get("grounded") or row.get("grounded_citations") or []
    halluc = details.get("hallucinated") or row.get("hallucinated_citations") or []
    tool_ids = row.get("tool_ids") or []
    called = bool(comp.get("tool_use")) or bool(row.get("tool_called"))

    if not called:
        return "no_tool", f"未调用工具，却引用了 {cited or '（无）'}"
    if not tool_ids:
        return "no_recall", "工具调用成功但返回 0 条候选"
    if halluc:
        return "hallucinated", f"编造 {halluc}"
    gold = (task or {}).get("gold_article_ids") or []
    hit = [c for c in grounded if c in gold]
    if gold and not hit:
        return "wrong_article", f"引用了 {grounded}，gold 是 {gold}"
    if (task or {}).get("task_type") == "insufficient_information" and comp.get("insufficiency", 1) < 0.5:
        return "insufficient_handling", "没有点名缺失事实或没有给出不确定性说明"
    if comp.get("answer_points", 1) < 0.4:
        return "missing_points", f"points={comp.get('answer_points', 0):.2f}"
    return "ok", ""


def analyze() -> dict:
    tasks = {}
    dev_path = TASKS / "tasks_dev_labeled.jsonl"
    if dev_path.exists():
        for line in dev_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                task = json.loads(line)
                tasks[task["task_id"]] = task

    results = {"agentic": [], "open_ended": [], "grounded": {}}

    score_dev = load(RESULTS / "score_dev.json")
    if score_dev:
        for row in score_dev["rows"]:
            task = tasks.get(row["task_id"])
            cls, why = classify(task, row)
            results["agentic"].append({
                "task_id": row["task_id"],
                "task_type": row.get("task_type"),
                "total": round(row["total"], 3),
                "class": cls,
                "evidence": why,
                "cited": row.get("details", {}).get("cited", []),
                "gold": (task or {}).get("gold_article_ids", []),
                "answer": row.get("answer", "")[:200],
            })

    baseline = load(RESULTS / "baseline_report.json")
    if baseline:
        for row in baseline["rows"]:
            cls, why = classify(None, {
                "tool_ids": row.get("tool_ids") or [],
                "components": {"tool_use": 1.0 if row.get("tool_called") else 0.0},
                "details": {
                    "cited": row.get("cited_ids") or [],
                    "grounded": row.get("grounded_citations") or [],
                    "hallucinated": row.get("hallucinated_citations") or [],
                },
            })
            results["open_ended"].append({"query": row["query"], "class": cls, "evidence": why})

    for name in ("before", "after"):
        data = load(RESULTS / f"eval_grounded_{name}.json")
        if data:
            results["grounded"][name] = {
                "mean_total": data["summary"]["mean_total"],
                "rows": {r["task_id"]: round(r["total"], 3) for r in data["rows"]},
            }
    return results


def render(results: dict) -> str:
    lines = ["# Bad Case 报告", ""]
    for section, key in (("Agentic 评测（dev，10 题）", "agentic"), ("开放问题（6 题）", "open_ended")):
        rows = results[key]
        if not rows:
            continue
        counts = Counter(r["class"] for r in rows)
        n = len(rows)
        lines += [f"## {section}", "", f"样本 {n} 条，通过 {counts.get('ok', 0)} 条。", "",
                  "| 失败类型 | 数量 | 占比 | 含义 |", "| --- | ---: | ---: | --- |"]
        for cls, meaning in CLASS_RULES:
            if counts.get(cls):
                lines.append(f"| `{cls}` | {counts[cls]} | {counts[cls]/n:.0%} | {meaning} |")
        if counts.get("ok"):
            lines.append(f"| `ok` | {counts['ok']} | {counts['ok']/n:.0%} | 通过 |")
        lines.append("")
        for cls, _ in CLASS_RULES:
            examples = [r for r in rows if r["class"] == cls]
            if not examples:
                continue
            lines.append(f"### {cls}（{len(examples)} 条）")
            for r in examples[:3]:
                label = r.get("task_id") or r.get("query", "")[:40]
                lines.append(f"- **{label}** `total={r.get('total', '?')}`：{r['evidence']}")
                if r.get("gold"):
                    lines.append(f"  - gold={r['gold']}，实际引用={r['cited']}")
            lines.append("")

    grounded = results.get("grounded") or {}
    if "before" in grounded and "after" in grounded:
        b, a = grounded["before"], grounded["after"]
        moved = [t for t in b["rows"] if abs(a["rows"].get(t, 0) - b["rows"][t]) > 1e-6]
        lines += ["## 训练前后（grounded dev）", "",
                  f"- 均值：{b['mean_total']:.4f} → {a['mean_total']:.4f}",
                  f"- 分数变化的题：{len(moved)}/{len(b['rows'])}", ""]

    lines += [
        "## 第二轮训练改动清单", "",
        "| 优先级 | 动作 | 对应失败类 |", "| --- | --- | --- |",
        "| P0 | 训练目标换成多轮 agentic（现在 `tool_use` 恒为 1，是假信号） | `no_tool` |",
        "| P0 | 奖励里加“引用必须来自 get_statute 读过的条文”，压 `wrong_article` | `wrong_article` |",
        "| P1 | 检索召回补充：把 0 命中的题做成 hard negative 加入训练 | `no_recall` |",
        "| P1 | 系统提示改为“先检索→逐条 get_statute→再作答” | `no_tool` / `wrong_article` |",
        "| P2 | 信息不足题单独加权，避免“硬答”和“一律拒答”两个极端 | `insufficient_handling` |",
        "| P2 | 字数上限放宽到 200 字，`w_points` 提到 0.40 | `missing_points` |",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    results = analyze()
    report = render(results)
    (RESULTS / "bad_case_report.md").write_text(report, encoding="utf-8")
    (RESULTS / "bad_cases.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report)
    print(f"\n[written] {RESULTS / 'bad_case_report.md'}")


if __name__ == "__main__":
    main()
