"""Unit tests for the deterministic reward (no model, no GPU)."""

from __future__ import annotations

import unittest

from legalagent.reward import (
    RewardConfig,
    coverage,
    extract_citations,
    reward_fn,
    score_rollout,
    split_claims,
)

TASK = {
    "task_id": "T1",
    "task_type": "fact_application",
    "gold_article_ids": ["labor_contract_law:9", "labor_contract_law:84"],
    "expected_answer_points": [
        "不得要求劳动者提供担保或者以其他名义收取财物；违法收取押金的，责令退还并处罚款"
    ],
    "missing_facts": [],
}

TASK_INS = {
    "task_id": "T2",
    "task_type": "insufficient_information",
    "gold_article_ids": ["labor_contract_law:39"],
    "expected_answer_points": ["不能仅凭迟到就认定严重违纪，需要核查规章制度与证据"],
    "missing_facts": ["合法有效的规章制度及其公示", "迟到次数与严重程度", "相关证据"],
}


def rollout(answer, ids, calls=None):
    if calls is None:
        calls = [{"name": "search_statute", "ok": True}]
    return {
        "answer": answer,
        "tool_ids": ids,
        "tool_calls": calls,
    }


class CitationTests(unittest.TestCase):
    def test_plain_id(self):
        self.assertEqual(extract_citations("引用 labor_contract_law:39"), ["labor_contract_law:39"])

    def test_chinese_named_citation(self):
        self.assertEqual(extract_citations("依据《劳动合同法》第三十九条"), ["labor_contract_law:39"])

    def test_longest_alias_wins(self):
        self.assertEqual(
            extract_citations("《劳动合同法实施条例》第十九条"), ["labor_contract_law_regulation:19"]
        )

    def test_dedup_and_empty(self):
        self.assertEqual(extract_citations("labor_contract_law:39 与 labor_contract_law:39"), ["labor_contract_law:39"])
        self.assertEqual(extract_citations("没有任何引用"), [])

    def test_split_claims(self):
        self.assertEqual(len(split_claims("a；b;c")), 3)

    def test_coverage_bounds(self):
        self.assertGreater(coverage("用人单位应当支付经济补偿", "用人单位应当支付经济补偿"), 0.99)
        self.assertLess(coverage("用人单位应当支付经济补偿", "劳动合同期限"), 0.1)


class ScoreTests(unittest.TestCase):
    def test_tool_and_grounded_beat_lazy(self):
        good = score_rollout(TASK, rollout(
            "不得要求劳动者提供担保或者以其他名义收取财物；违法收取押金的，责令退还并处罚款。"
            "引用：labor_contract_law:9、labor_contract_law:84。",
            ["labor_contract_law:9", "labor_contract_law:84"],
        ))
        lazy = score_rollout(TASK, rollout(
            "公司收押金违法。引用：labor_contract_law:87。", [], calls=[],
        ))
        self.assertGreater(good.total, 0.8)
        self.assertLess(lazy.total, 0.3)
        self.assertIn("labor_contract_law:87", lazy.details["hallucinated"])
        self.assertLessEqual(lazy.components["tool_use"], 0.0 + 1e-9)

    def test_hallucination_is_capped(self):
        bad = score_rollout(TASK, rollout(
            "引用 a:b、c:d、e:f、g:h。", ["labor_contract_law:9"],
        ))
        self.assertLessEqual(bad.details["penalty"], RewardConfig().hallucination_cap)
        self.assertGreaterEqual(bad.total, RewardConfig().min_total)

    def test_gold_recall_partial(self):
        half = score_rollout(TASK, rollout(
            "不得收取财物。引用：labor_contract_law:9。", ["labor_contract_law:9"],
        ))
        # recall=0.5, precision=1.0 -> F1=0.667
        self.assertGreater(half.components["gold"], 0.3)
        self.assertLess(half.components["gold"], 0.8)

    def test_extra_grounded_citation_not_zeroed(self):
        """README: gold is necessary but not exhaustive, so extra grounded ids only cost precision."""
        extra = score_rollout(TASK, rollout(
            "不得收取财物。引用：labor_contract_law:9、labor_contract_law:84、labor_contract_law:39。",
            ["labor_contract_law:9", "labor_contract_law:84", "labor_contract_law:39"],
        ))
        self.assertGreater(extra.components["gold"], 0.6)

    def test_insufficient_task_rewards_naming_facts(self):
        strong = score_rollout(TASK_INS, rollout(
            "仅凭迟到不能直接认定严重违纪。需要补充合法有效的规章制度及其公示、迟到次数与严重程度、相关证据，"
            "否则无法判断解除是否合法。引用：labor_contract_law:39。",
            ["labor_contract_law:39"],
        ))
        weak = score_rollout(TASK_INS, rollout(
            "公司可以解除。引用：labor_contract_law:39。", ["labor_contract_law:39"],
        ))
        self.assertGreater(strong.components["insufficiency"], 0.8)
        self.assertLess(weak.components["insufficiency"], 0.3)
        self.assertTrue(strong.details["hedged"])

    def test_deterministic(self):
        a = score_rollout(TASK, rollout("不得收取财物。引用：labor_contract_law:9。", ["labor_contract_law:9"]))
        b = score_rollout(TASK, rollout("不得收取财物。引用：labor_contract_law:9。", ["labor_contract_law:9"]))
        self.assertEqual(a.as_dict(), b.as_dict())

    def test_cite_all_spam_is_penalized(self):
        """Dumping every id from the tool output must not beat a real answer."""
        ids = [f"labor_contract_law:{n}" for n in (9, 84, 17, 23, 39, 42, 46, 48)]
        spam = score_rollout(TASK, rollout(
            "收取押金违法。引用：" + "、".join(ids) + "。", ids,
        ))
        good = score_rollout(TASK, rollout(
            "不得要求劳动者提供担保或者以其他名义收取财物；违法收取押金的，责令退还并处罚款。"
            "引用：labor_contract_law:9、labor_contract_law:84。",
            ["labor_contract_law:9", "labor_contract_law:84"],
        ))
        self.assertLess(spam.total, 0.5)
        self.assertLess(spam.total, good.total - 0.3)

    def test_missing_citation_is_penalized(self):
        uncited = score_rollout(TASK, rollout(
            "用人单位不得收取押金或要求担保，违法收取的应退还并处罚款。",
            ["labor_contract_law:9", "labor_contract_law:84"],
        ))
        cited = score_rollout(TASK, rollout(
            "用人单位不得收取押金或要求担保，违法收取的应退还并处罚款。"
            "引用：labor_contract_law:9、labor_contract_law:84。",
            ["labor_contract_law:9", "labor_contract_law:84"],
        ))
        self.assertTrue(uncited.details["no_citation"])
        self.assertLess(uncited.total, cited.total - 0.15)

    def test_reward_fn_adapter(self):
        fn = reward_fn({"T1": TASK})
        scores = fn(["不得收取财物。引用：labor_contract_law:9。"], task_id=["T1"])
        self.assertEqual(len(scores), 1)
        self.assertIsInstance(scores[0], float)


if __name__ == "__main__":
    unittest.main(verbosity=2)
