"""Two deterministic, version-aware tools for Legal Agentic RL.

No external services and no third-party dependencies. This module deliberately
returns an *unknown* historical state instead of substituting a later law version.
"""
from __future__ import annotations

import math
import re
import sqlite3
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any

from ..paths import DATA

DB_PATH = DATA / "legal_statutes.sqlite3"
MAX_QUERY_LEN = 512

# Tiny domain-specific expansions; never used as legal rules or answers.
QUERY_EXPANSIONS: tuple[tuple[str, str], ...] = (
    ("员工", "劳动者"), ("职工", "劳动者"), ("公司", "用人单位"),
    ("企业", "用人单位"), ("辞退", "解除劳动合同"),
    ("被开除", "解除劳动合同"), ("补偿金", "经济补偿"),
    ("没签合同", "未订立书面劳动合同"), ("合同没签", "未订立书面劳动合同"),
    ("拖欠工资", "劳动报酬"), ("加班费", "加班工资"),
    ("违法解除", "违反本法规定解除或者终止劳动合同"),
    ("双倍工资", "二倍的工资"), ("没签合同", "未与劳动者订立书面劳动合同"),
)
# Strong legal concepts used to boost precise terms (not generic "劳动" etc.)
CONCEPTS = (
    "严重违反", "规章制度", "试用期间", "解除劳动合同", "终止劳动合同",
    "违反本法规定解除或者终止劳动合同", "二倍的工资",
    "未订立书面劳动合同", "经济补偿", "赔偿金", "劳动报酬", "加班工资",
    "社会保险", "劳动争议", "仲裁时效", "竞业限制", "裁减人员",
    "违法解除", "提前三十日", "即时解除", "工会", "无固定期限",
    "不符合录用条件", "患病", "医疗期", "工作年限",
)


def parse_date(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("applicable_date must be a YYYY-MM-DD string")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid applicable_date {value!r}; expected YYYY-MM-DD") from exc
    if parsed.isoformat() != value:
        raise ValueError("applicable_date must be YYYY-MM-DD")
    return value


def normalize(text: str) -> str:
    return "".join(re.findall(r"[\u3400-\u9fffA-Za-z0-9]+", text.lower()))


def bigrams(text: str) -> set[str]:
    text = normalize(text)
    if len(text) < 2:
        return {text} if text else set()
    return set(text[i:i+2] for i in range(len(text)-1))


def expand_query(query: str) -> str:
    expanded = [query]
    for trigger, expansion in QUERY_EXPANSIONS:
        if trigger in query and expansion not in query:
            expanded.append(expansion)
    return " ".join(expanded)


def _chinese_number(raw: str) -> int | None:
    """Parse a small Chinese article number, e.g. '第三十九条' -> 39."""
    if raw.isdecimal():
        return int(raw)
    digits = {"零": 0, "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    units = {"十": 10, "百": 100, "千": 1000}
    if not raw or any(c not in digits and c not in units for c in raw):
        return None
    total, current = 0, 0
    for c in raw:
        if c in digits:
            current = digits[c]
        else:
            total += (current or 1) * units[c]
            current = 0
    return total + current


def _explicit_article_reference(query: str) -> int | None:
    m = re.search(r"第\s*([一二三四五六七八九十百千零〇0-9]+)\s*条", query)
    return _chinese_number(m.group(1)) if m else None


def _named_law(query: str) -> str | None:
    """Prefer longest specific law alias; avoid accidental 'labor contract' topic matches."""
    squashed = normalize(query)
    aliases = (
        ("劳动合同法实施条例", "labor_contract_law_regulation"),
        ("劳动争议调解仲裁法", "labor_dispute_arbitration_law"),
        ("劳动合同法", "labor_contract_law"),
        ("中华人民共和国劳动法", "labor_law"),
        ("劳动法", "labor_law"),
    )
    for alias, law_id in aliases:
        if alias in squashed:
            return law_id
    return None


def _checked_date(record: sqlite3.Row) -> str:
    return datetime.fromisoformat(record["source_checked_at"].replace("Z", "+00:00")).date().isoformat()


def _version_state(record: sqlite3.Row, as_of: str) -> tuple[str, list[str]]:
    checked = _checked_date(record)
    if as_of > checked:
        return ("unverified_after_snapshot", [
            f"Requested date {as_of} is later than source_checked_at ({checked}); "
            "this stored revision is not confirmed current at that later date."
        ])
    return ("within_snapshot_date_window", [
        "Time-window match only; the corpus does not independently adjudicate every transitional legal rule."
    ])


class LegalStatuteStore:
    def __init__(self, db_path: str | Path = DB_PATH):
        self.db_path = str(db_path)
        if not Path(self.db_path).is_file():
            raise FileNotFoundError(f"Database missing: {self.db_path}. Run build_db.py first.")
        self.conn = sqlite3.connect(f"file:{Path(self.db_path).resolve()}?mode=ro", uri=True)
        self.conn.row_factory = sqlite3.Row
        self._rows = list(self.conn.execute("SELECT * FROM statute_versions"))
        if not self._rows:
            raise ValueError("Database has no articles")
        self._law_titles = {r["law_id"]: r["law_title"] for r in self._rows}
        self._doc_grams = [bigrams(r["law_title"] + " " + r["text"]) for r in self._rows]
        self._df = Counter(g for grams in self._doc_grams for g in grams)
        self._n = len(self._rows)

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "LegalStatuteStore":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _law_coverage(self, as_of: str) -> list[dict]:
        # As-of earlier than the first *stored* version is not proof of non-existence.
        coverage = []
        for law_id in sorted(self._law_titles):
            law_rows = [r for r in self._rows if r["law_id"] == law_id]
            earliest = min(r["effective_from"] for r in law_rows)
            latest_checked = max(_checked_date(r) for r in law_rows)
            if as_of < earliest:
                state = "historical_version_missing_or_law_not_yet_effective"
            elif as_of > latest_checked:
                state = "unverified_after_snapshot"
            else:
                state = "stored_version_available"
            coverage.append({"law_id": law_id, "law_title": self._law_titles[law_id],
                             "earliest_stored_effective_from": earliest, "status": state})
        return coverage

    def _active_versions(self, as_of: str) -> list[sqlite3.Row]:
        return list(self.conn.execute(
            "SELECT * FROM statute_versions WHERE effective_from <= ? "
            "AND (effective_to IS NULL OR effective_to > ?)", (as_of, as_of)
        ))

    def search_statute(self, query: str, applicable_date: str, limit: int = 5) -> dict:
        """Find stored statutes relevant to a legal question at a given date.

        Args:
            query: Chinese legal search terms, not an SQL query.
            applicable_date: Legal fact/reference date in YYYY-MM-DD format.
            limit: Number of results, from 1 to 20.

        Returns:
            JSON-compatible dictionary with matching article IDs, snippets,
            source links, temporal coverage warnings, and retrieval scores.
        """
        if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY_LEN:
            return {"ok": False, "error": "invalid_query", "message": "query must contain 1–512 characters"}
        try:
            as_of = parse_date(applicable_date)
        except ValueError as exc:
            return {"ok": False, "error": "invalid_date", "message": str(exc)}
        if type(limit) is not int or not 1 <= limit <= 20:
            return {"ok": False, "error": "invalid_limit", "message": "limit must be an integer from 1 to 20"}

        expanded = expand_query(query)
        qgrams = bigrams(expanded)
        query_plain = normalize(query)
        q_concepts = [term for term in CONCEPTS if term in expanded]
        named_law = _named_law(query)
        explicit_number = _explicit_article_reference(query)
        candidates = self._active_versions(as_of)
        ranked: list[tuple[float, sqlite3.Row]] = []
        for r in candidates:
            text = normalize(r["text"])
            title = normalize(r["law_title"])
            grams = bigrams(r["law_title"] + " " + r["text"])
            common = qgrams & grams
            if not common:
                continue
            query_idf = sum(math.log(1 + (self._n + 1) / (1 + self._df.get(g, 0))) for g in qgrams)
            match_idf = sum(math.log(1 + (self._n + 1) / (1 + self._df.get(g, 0))) for g in common)
            lexical_coverage = match_idf / query_idf if query_idf else 0.0
            # Avoid favoring very long omnibus articles for generic queries.
            score = lexical_coverage / (1 + 0.18 * max(0, len(text) - 180) / 180)
            if len(query_plain) >= 4 and query_plain in title + text:
                score += 0.48
            for term in q_concepts:
                if term in r["text"]:
                    score += 0.28
            # A law title specifically named in the query is strong evidence.
            if len(title) > 4 and title in query_plain:
                score += 0.25
            if named_law:
                score += 0.36 if r["law_id"] == named_law else -0.22
            # A precise named citation should always outrank unrelated Article 39s.
            if explicit_number is not None and r["article_number"] == explicit_number:
                score += 4.0 if (not named_law or r["law_id"] == named_law) else 0.0
            ranked.append((score, r))
        ranked.sort(key=lambda x: (-x[0], x[1]["law_id"], x[1]["article_number"]))
        results = []
        for score, r in ranked[:limit]:
            temporal, notes = _version_state(r, as_of)
            results.append({
                "article_id": f"{r['law_id']}:{r['article_number']}",
                "law_title": r["law_title"], "article_number": r["article_number"],
                "article_number_display": r["article_number_display"],
                "version_id": r["version_id"], "effective_from": r["effective_from"],
                "effective_to": r["effective_to"], "temporal_status": temporal,
                "snippet": r["text"][:220] + ("…" if len(r["text"]) > 220 else ""),
                "source_url": r["source_url"], "score": round(score, 5),
            })
        coverage = self._law_coverage(as_of)
        warnings = []
        missing = [x["law_title"] for x in coverage if x["status"].startswith("historical_version")]
        future = [x["law_title"] for x in coverage if x["status"] == "unverified_after_snapshot"]
        if missing:
            warnings.append("Missing historical coverage (or law not yet in force): " + "、".join(missing))
        if future:
            warnings.append("Not revalidated through requested date: " + "、".join(future))
        if not results:
            warnings.append("No matching article in the covered snapshot; do not infer no such legal rule exists")
        return {"ok": True, "query": query, "applicable_date": as_of,
                "retrieval": "local_character_bigram_plus_legal_terms", "results": results,
                "coverage": coverage, "warnings": warnings}

    def get_statute(self, article_id: str, applicable_date: str) -> dict:
        """Read a specific statute article using a stable ID and as-of date.

        Args:
            article_id: Canonical ID from search results, e.g. labor_contract_law:39.
            applicable_date: Legal fact/reference date in YYYY-MM-DD format.

        Returns:
            JSON-compatible dictionary containing the full verified-source
            article text, version metadata, and time-coverage warning.
        """
        try:
            as_of = parse_date(applicable_date)
        except ValueError as exc:
            return {"ok": False, "error": "invalid_date", "message": str(exc)}
        if not isinstance(article_id, str):
            return {"ok": False, "error": "invalid_article_id", "message": "Use <law_id>:<article_number>"}
        match = re.fullmatch(r"([a-z][a-z0-9_]*)\s*:\s*([1-9][0-9]*)", article_id.strip())
        if not match:
            return {"ok": False, "error": "invalid_article_id", "message": "Use ID from search_statute, e.g. labor_contract_law:39"}
        law_id, article_number = match.group(1), int(match.group(2))
        revisions = list(self.conn.execute(
            "SELECT * FROM statute_versions WHERE law_id=? AND article_number=? ORDER BY effective_from",
            (law_id, article_number)
        ))
        if not revisions:
            return {"ok": False, "error": "article_not_in_subset", "article_id": article_id,
                    "message": "This article is not in the local dataset, not proof it does not exist."}
        candidates = [r for r in revisions if r["effective_from"] <= as_of
                      and (r["effective_to"] is None or as_of < r["effective_to"])]
        if not candidates:
            return {"ok": False, "error": "version_unavailable_for_date",
                    "article_id": f"{law_id}:{article_number}", "applicable_date": as_of,
                    "stored_versions": [{"version_id": r["version_id"],
                                         "effective_from": r["effective_from"],
                                         "effective_to": r["effective_to"]} for r in revisions],
                    "message": "No text revision for this date in the local snapshot. "
                               "The historical revision may be missing, or this article may not yet have been in force. "
                               "Never substitute the latest version."}
        if len(candidates) != 1:
            raise RuntimeError("Ambiguous overlapping revisions; database should have rejected these")
        r = candidates[0]
        temporal, notes = _version_state(r, as_of)
        article = {
            "article_id": f"{law_id}:{article_number}",
            "law_id": law_id, "law_title": r["law_title"],
            "article_number": article_number, "article_number_display": r["article_number_display"],
            "part": r["part"], "text": r["text"], "version_id": r["version_id"],
            "effective_from": r["effective_from"], "effective_to": r["effective_to"],
            "status_in_snapshot": r["status"], "law_status_raw": r["law_status_raw"],
            "source_url": r["source_url"], "source_name": r["source_name"],
            "source_checked_at": r["source_checked_at"], "source_hash": r["source_hash"],
            "text_sha256": r["text_sha256"], "effective_from_basis": r["effective_from_basis"],
            "level": r["level"], "issuing_body": r["issuing_body"],
        }
        return {"ok": True, "applicable_date": as_of, "temporal_status": temporal,
                "notes": notes, "article": article}


# Lazy singleton for TRL plain-Python tool calling; tool functions take JSON-safe arguments.
_STORE: LegalStatuteStore | None = None


def _default_store() -> LegalStatuteStore:
    global _STORE
    if _STORE is None:
        _STORE = LegalStatuteStore(DB_PATH)
    return _STORE


def search_statute(query: str, applicable_date: str, limit: int = 5) -> dict:
    """Search Chinese labor laws by query, using a requested legal date.

    Args:
        query: Legal issue or short Chinese phrase to search for.
        applicable_date: Required YYYY-MM-DD date for selecting law revisions.
        limit: Maximum number of articles to return (1 to 20).

    Returns:
        A dictionary with result article IDs, snippets, sources and coverage warnings.
    """
    return _default_store().search_statute(query, applicable_date, limit)


def get_statute(article_id: str, applicable_date: str) -> dict:
    """Retrieve complete text of a cited law article, as of a requested date.

    Args:
        article_id: Stable ID from search_statute, for example labor_contract_law:39.
        applicable_date: Required YYYY-MM-DD date for selecting law revisions.

    Returns:
        A dictionary with full article, revision metadata and validity warnings,
        or an explicit error when the date is not covered by the local snapshot.
    """
    return _default_store().get_statute(article_id, applicable_date)
