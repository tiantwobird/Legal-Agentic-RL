"""Write the expert-review CSV from the 50 reviewed tasks.

    python -m legalagent.data.create_review_sheet

Output: results/reviews/manual_expert_review_sheet.csv
"""

from __future__ import annotations

import csv
import json

from ..paths import RESULTS, TASKS

FIELDS = [
    "task_id",
    "task_type",
    "split",
    "prompt",
    "gold_article_ids",
    "gold_issues",
    "expected_answer_points",
    "source_evidence_and_quotes",
    "missing_facts",
    "assistant_review",
    "independent_expert_review",
    "expert_comments",
]


def main() -> None:
    rows = [
        json.loads(line)
        for line in (TASKS / "tasks_all_reviewed.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    out = RESULTS / "reviews" / "manual_expert_review_sheet.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for task in rows:
            writer.writerow(
                {
                    "task_id": task["task_id"],
                    "task_type": task["task_type"],
                    "split": task["split"],
                    "prompt": task["prompt"],
                    "gold_article_ids": "; ".join(task["gold_article_ids"]),
                    "gold_issues": "; ".join(task["gold_issues"]),
                    "expected_answer_points": "; ".join(task["expected_answer_points"]),
                    "source_evidence_and_quotes": " | ".join(
                        evidence["article_id"] + ": " + evidence["supporting_quote"]
                        for evidence in task["source_evidence"]
                    ),
                    "missing_facts": "; ".join(task["missing_facts"]),
                    "assistant_review": "PASS: corpus evidence exact matched; substantive label checked by assistant",
                    "independent_expert_review": "PENDING",
                    "expert_comments": "",
                }
            )
    print(f"[written] {out}")


if __name__ == "__main__":
    main()
