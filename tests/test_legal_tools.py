"""Run: python -m unittest -v test_legal_tools.py"""
import contextlib
import hashlib
import json
import unittest
from pathlib import Path

from ._tmp import drop_tmpdir, make_tmpdir
from legalagent.data.build_db import build_db, read_and_validate
from legalagent.paths import DATA as DATA_DIR
from legalagent.tools.legal_tools import LegalStatuteStore

DB = DATA_DIR / "legal_statutes.sqlite3"
DATA = DATA_DIR / "labor_subset.jsonl"


@contextlib.contextmanager
def temp_dir():
    path = make_tmpdir()
    try:
        yield path
    finally:
        drop_tmpdir(path)


class DatasetTests(unittest.TestCase):
    def test_dataset_shape_and_hashes(self):
        records = read_and_validate(DATA)
        self.assertEqual(len(records), 351)
        self.assertEqual(len({r["law_id"] for r in records}), 5)
        self.assertEqual(len({(r["law_id"], r["version_id"]) for r in records}), 5)
        self.assertTrue(all(r["effective_to"] is None for r in records))

    def test_builder_rejects_corrupt_text(self):
        with temp_dir() as tmp:
            p = Path(tmp) / "bad.jsonl"
            r = json.loads(DATA.read_text(encoding="utf-8").splitlines()[0])
            r["text"] += "（篡改）"
            p.write_text(json.dumps(r, ensure_ascii=False) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sha256 mismatch"):
                build_db(p, Path(tmp) / "bad.db")

    def test_two_revisions_choose_by_date(self):
        with temp_dir() as tmp:
            lines = []
            example = json.loads(DATA.read_text(encoding="utf-8").splitlines()[0])
            for version, start, end, suffix in [
                ("rev_a", "2020-01-01", "2022-01-01", " 旧版"),
                ("rev_b", "2022-01-01", None, " 新版"),
            ]:
                r = dict(example, version_id=version, effective_from=start, effective_to=end,
                         text=example["text"] + suffix)
                r["text_sha256"] = hashlib.sha256(r["text"].encode()).hexdigest()
                lines.append(json.dumps(r, ensure_ascii=False))
            p = Path(tmp) / "two.jsonl"
            db = Path(tmp) / "two.sqlite3"
            p.write_text("\n".join(lines) + "\n", encoding="utf-8")
            build_db(p, db)
            with LegalStatuteStore(db) as store:
                old = store.get_statute("labor_law:1", "2021-12-31")
                new = store.get_statute("labor_law:1", "2022-01-01")
                self.assertEqual(old["article"]["version_id"], "rev_a")
                self.assertEqual(new["article"]["version_id"], "rev_b")


class ToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = LegalStatuteStore(DB)

    @classmethod
    def tearDownClass(cls):
        cls.store.close()

    def test_statute_for_dismissal(self):
        data = self.store.search_statute("严重违反规章制度 解除劳动合同", "2024-01-01")
        self.assertTrue(data["ok"])
        self.assertIn("labor_contract_law:39", [x["article_id"] for x in data["results"][:3]])

    def test_unlawful_dismissal(self):
        data = self.store.search_statute("违法解除劳动合同赔偿金", "2024-01-01")
        self.assertIn("labor_contract_law:87", [x["article_id"] for x in data["results"][:3]])

    def test_double_pay(self):
        data = self.store.search_statute("没签合同 双倍工资", "2024-01-01")
        self.assertEqual(data["results"][0]["article_id"], "labor_contract_law:82")

    def test_arbitration(self):
        data = self.store.search_statute("劳动争议仲裁时效", "2024-01-01")
        self.assertEqual(data["results"][0]["article_id"], "labor_dispute_arbitration_law:27")

    def test_exact_reference(self):
        data = self.store.search_statute("劳动合同法第三十九条", "2024-01-01")
        self.assertEqual(data["results"][0]["article_id"], "labor_contract_law:39")

    def test_article_full_text_and_source(self):
        response = self.store.get_statute("labor_contract_law:39", "2024-01-01")
        self.assertTrue(response["ok"])
        self.assertIn("严重违反用人单位的规章制度", response["article"]["text"])
        self.assertTrue(response["article"]["source_url"].startswith("https://"))
        self.assertEqual(response["temporal_status"], "within_snapshot_date_window")

    def test_history_missing_no_fallback(self):
        response = self.store.get_statute("labor_contract_law:39", "2010-01-01")
        self.assertFalse(response["ok"])
        self.assertEqual(response["error"], "version_unavailable_for_date")

    def test_later_than_snapshot_marked_unverified(self):
        response = self.store.get_statute("labor_contract_law:39", "2026-10-20")
        self.assertTrue(response["ok"])
        self.assertEqual(response["temporal_status"], "unverified_after_snapshot")

    def test_invalid_inputs_safe(self):
        self.assertEqual(self.store.get_statute("labor_contract_law:39 OR 1=1", "2024-01-01")["error"], "invalid_article_id")
        self.assertEqual(self.store.get_statute("labor_contract_law:39", "2024-02-30")["error"], "invalid_date")
        self.assertEqual(self.store.search_statute("", "2024-01-01")["error"], "invalid_query")
        self.assertEqual(self.store.search_statute("解除", "2024-01-01", limit=999)["error"], "invalid_limit")

    def test_historical_coverage_warning(self):
        response = self.store.search_statute("解除劳动合同", "2010-01-01")
        self.assertTrue(response["ok"])
        self.assertIn("Missing historical coverage", " ".join(response["warnings"]))
        self.assertTrue(all(r["effective_from"] <= "2010-01-01" for r in response["results"]))


if __name__ == "__main__":
    unittest.main()
