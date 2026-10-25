"""Build a read-only-at-runtime, version-aware SQLite snapshot from JSONL.

No third-party dependencies. Run:
    python -m legalagent.data.build_db
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

from ..paths import DATA

REQUIRED = (
    "law_id", "law_title", "article_number", "version_id",
    "effective_from", "text", "source_url", "source_checked_at", "text_sha256",
)


def parse_date(value: str, field: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be YYYY-MM-DD, received: {value!r}")
    try:
        result = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid {field}: {value!r}") from exc
    if result.isoformat() != value:
        raise ValueError(f"{field} must use YYYY-MM-DD format")
    return result


def read_and_validate(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict] = []
    seen: set[tuple[str, int, str]] = set()
    version_dates: dict[tuple[str, str], tuple] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSONL line {line_no}: invalid JSON: {exc}") from exc
        for name in REQUIRED:
            if row.get(name) is None or row[name] == "":
                raise ValueError(f"JSONL line {line_no}: missing {name}")
        if type(row["article_number"]) is not int or row["article_number"] < 1:
            raise ValueError(f"JSONL line {line_no}: invalid article_number")
        if not isinstance(row["text"], str):
            raise ValueError(f"JSONL line {line_no}: text must be a string")
        start = parse_date(row["effective_from"], "effective_from")
        end = (parse_date(row["effective_to"], "effective_to")
               if row.get("effective_to") else None)
        if end and end <= start:
            raise ValueError(f"JSONL line {line_no}: effective_to must be exclusive and later than effective_from")
        try:
            datetime.fromisoformat(row["source_checked_at"].replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"JSONL line {line_no}: invalid source_checked_at") from exc
        digest = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
        if digest.lower() != row["text_sha256"].lower():
            raise ValueError(f"JSONL line {line_no}: text_sha256 mismatch")
        key = (row["law_id"], row["article_number"], row["version_id"])
        if key in seen:
            raise ValueError(f"JSONL line {line_no}: duplicate {key}")
        seen.add(key)
        vk = (row["law_id"], row["version_id"])
        meta = (row["law_title"], row["effective_from"], row.get("effective_to"), row["source_url"])
        if vk in version_dates and version_dates[vk] != meta:
            raise ValueError(f"JSONL line {line_no}: inconsistent version metadata for {vk}")
        version_dates[vk] = meta
        rows.append(row)

    if not rows:
        raise ValueError("Dataset is empty")

    # Reject ambiguous overlapping revisions for the same law and article.
    windows: dict[tuple[str, int], list[tuple[date, date | None, str]]] = defaultdict(list)
    for r in rows:
        windows[(r["law_id"], r["article_number"])].append((
            parse_date(r["effective_from"], "effective_from"),
            parse_date(r["effective_to"], "effective_to") if r.get("effective_to") else None,
            r["version_id"],
        ))
    for key, revisions in windows.items():
        ordered = sorted(revisions, key=lambda x: x[0])
        for a, b in zip(ordered, ordered[1:]):
            if a[1] is None or a[1] > b[0]:
                raise ValueError(f"Overlapping/missing effective_to for {key}: {a[2]} and {b[2]}")
    return rows


def build_db(jsonl_path: str | Path, db_path: str | Path) -> dict:
    jsonl_path, db_path = Path(jsonl_path), Path(db_path)
    records = read_and_validate(jsonl_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = db_path.with_name(db_path.name + ".building")
    if temp_path.exists():
        temp_path.unlink()
    conn = sqlite3.connect(temp_path)
    try:
        conn.executescript("""
            CREATE TABLE statute_versions (
                law_id TEXT NOT NULL,
                law_title TEXT NOT NULL,
                article_number INTEGER NOT NULL,
                article_number_display TEXT,
                part TEXT,
                version_id TEXT NOT NULL,
                effective_from TEXT NOT NULL,
                effective_to TEXT,
                text TEXT NOT NULL,
                status TEXT,
                law_status_raw TEXT,
                level TEXT,
                issuing_body TEXT,
                document_number TEXT,
                source_url TEXT NOT NULL,
                source_name TEXT,
                source_checked_at TEXT NOT NULL,
                source_hash TEXT,
                text_sha256 TEXT NOT NULL,
                effective_from_basis TEXT,
                PRIMARY KEY (law_id, article_number, version_id)
            );
            CREATE INDEX idx_statute_law_article ON statute_versions (law_id, article_number);
            CREATE INDEX idx_statute_dates ON statute_versions (effective_from, effective_to);
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        columns = (
            "law_id", "law_title", "article_number", "article_number_display", "part",
            "version_id", "effective_from", "effective_to", "text", "status",
            "law_status_raw", "level", "issuing_body", "document_number",
            "source_url", "source_name", "source_checked_at", "source_hash",
            "text_sha256", "effective_from_basis",
        )
        conn.executemany(
            f"INSERT INTO statute_versions ({','.join(columns)}) VALUES ({','.join(['?'] * len(columns))})",
            [[r.get(c) for c in columns] for r in records]
        )
        metadata = {
            "schema_version": "1",
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "record_count": str(len(records)),
            "law_count": str(len({r["law_id"] for r in records})),
            "source_jsonl_sha256": hashlib.sha256(jsonl_path.read_bytes()).hexdigest(),
        }
        conn.executemany("INSERT INTO metadata VALUES (?, ?)", metadata.items())
        conn.commit()
        check = conn.execute("PRAGMA integrity_check").fetchone()[0]
        if check != "ok":
            raise RuntimeError(f"SQLite integrity failure: {check}")
    finally:
        conn.close()
    os.replace(temp_path, db_path)
    return {"db_path": str(db_path), "laws": metadata["law_count"], "articles": len(records),
            "versions": len({(r["law_id"], r["version_id"]) for r in records}),
            "source_sha256": metadata["source_jsonl_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default=str(DATA / "labor_subset.jsonl"))
    parser.add_argument("--db", default=str(DATA / "legal_statutes.sqlite3"))
    args = parser.parse_args()
    print(json.dumps(build_db(args.jsonl, args.db), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
