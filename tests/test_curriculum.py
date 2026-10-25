"""Tests for the curriculum loader (pure stdlib, no model, no GPU)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from ._tmp import drop_tmpdir, make_tmpdir
from legalagent.data.curriculum import load_curriculum


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


LABELS = [
    {
        "task_id": "T1",
        "task_type": "fact_application",
        "prompt": "带 gold 提示的完整版本（不应进入 prompt）",
        "gold_article_ids": ["labor_contract_law:39"],
        "expected_answer_points": ["point"],
        "missing_facts": [],
    },
    {
        "task_id": "T2",
        "task_type": "direct_retrieval",
        "prompt": "另一个带 gold 的版本",
        "gold_article_ids": ["labor_law:1"],
        "expected_answer_points": ["point"],
        "missing_facts": [],
    },
]


class CurriculumTests(unittest.TestCase):
    def setUp(self):
        self.tmp = make_tmpdir()
        self.labels = write_jsonl(self.tmp / "labels.jsonl", LABELS)

    def tearDown(self):
        drop_tmpdir(self.tmp)

    def test_prompt_comes_from_curriculum_not_labels(self):
        curr = write_jsonl(self.tmp / "curr.jsonl", [
            {"task_id": "T1", "prompt": "干净的题面", "jurisdiction": "CN", "applicable_date": "2024-06-01"},
        ])
        rows = load_curriculum(curr, self.labels)
        self.assertEqual([r["task_id"] for r in rows], ["T1"])
        self.assertEqual(rows[0]["prompt"], "干净的题面")
        # 标签仍然可用（reward 需要），但 prompt 已被题面覆盖
        self.assertEqual(rows[0]["gold_article_ids"], ["labor_contract_law:39"])

    def test_repeats_are_preserved(self):
        curr = write_jsonl(self.tmp / "curr.jsonl", [
            {"task_id": "T1", "prompt": "a"},
            {"task_id": "T2", "prompt": "b"},
            {"task_id": "T1", "prompt": "a"},
        ])
        rows = load_curriculum(curr, self.labels)
        self.assertEqual([r["task_id"] for r in rows], ["T1", "T2", "T1"])

    def test_limit_applies_after_the_join(self):
        curr = write_jsonl(self.tmp / "curr.jsonl", [
            {"task_id": "T2", "prompt": "b"},
            {"task_id": "T1", "prompt": "a"},
        ])
        rows = load_curriculum(curr, self.labels, limit=1)
        self.assertEqual([r["task_id"] for r in rows], ["T2"])

    def test_unknown_task_id_raises(self):
        curr = write_jsonl(self.tmp / "curr.jsonl", [{"task_id": "T9", "prompt": "x"}])
        with self.assertRaises(KeyError):
            load_curriculum(curr, self.labels)

    def test_empty_curriculum_raises(self):
        curr = write_jsonl(self.tmp / "curr.jsonl", [])
        with self.assertRaises(ValueError):
            load_curriculum(curr, self.labels)


if __name__ == "__main__":
    unittest.main()
