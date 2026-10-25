"""Export a small, citation-grade labour-law subset from the local chinalaw DB.

Every record is built from `chinalaw get <law> --format json`, so the text,
source URL, source hash, check time and revision dates all come from the
verified local database rather than from model memory.

Usage:
    python -m legalagent.data.export_subset                 # writes data/labor_subset.jsonl + data/manifest.json
    python -m legalagent.data.export_subset --include-history
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from ..paths import DATA, ROOT

# law_id (slug used in the dataset) -> chinalaw law id
LAWS: dict[str, dict[str, str]] = {
    "labor_law": {
        "chinalaw_id": "flk-labor-law-1994-2018",
        "title": "中华人民共和国劳动法",
    },
    "labor_contract_law": {
        "chinalaw_id": "2c909fdd678bf17901678bf74d7106b3",
        "title": "中华人民共和国劳动合同法",
    },
    "labor_contract_law_regulation": {
        "chinalaw_id": "gov_xzfgk:284",
        "title": "中华人民共和国劳动合同法实施条例",
    },
    "labor_dispute_arbitration_law": {
        "chinalaw_id": "2c909fdd678bf17901678bf64f28039d",
        "title": "中华人民共和国劳动争议调解仲裁法",
    },
    "labor_dispute_interpretation_1": {
        "chinalaw_id": "ff80808177e75f88017800739d621b12",
        "title": "最高人民法院关于审理劳动争议案件适用法律问题的解释（一）",
    },
}

# Facts the automated cleaner could not fill, verified against the publishing
# decree (国务院令第535号：2008-09-18 公布并施行). Kept separate and explicit so
# a reviewer can delete or correct them without touching the export logic.
OVERRIDES: dict[str, dict[str, str]] = {
    "gov_xzfgk:284": {
        "effective_from": "2008-09-18",
        "status": "effective",
        "basis": "国务院令第535号，2008-09-18 公布，自公布之日起施行",
        "source_url": "https://xzfg.moj.gov.cn/front/law/detail?LawID=284",
    },
}

STATUS_MAP = {
    "current": "effective",
    "amended": "superseded",
    "repealed": "repealed",
    "pending_effective": "pending",
    "unknown": "unknown",
}


def chinalaw_cmd() -> list[str]:
    """Prefer the repo venv's interpreter (-m chinalaw): .cmd shims need cmd.exe."""
    venv_python = Path(
        os.environ.get(
            "CHINALAW_PYTHON",
            str(ROOT / "chinalaw-cli" / ".venv" / "Scripts" / "python.exe"),
        )
    )
    if venv_python.is_file():
        return [str(venv_python), "-m", "chinalaw"]
    return ["chinalaw"]


def run_get(law_id: str, as_of: str | None = None) -> dict:
    cmd = [*chinalaw_cmd(), "get", law_id, "--format", "json"]
    if as_of:
        cmd += ["--as-of", as_of]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise SystemExit(f"chinalaw get {law_id} failed ({proc.returncode}):\n{proc.stderr}\n{proc.stdout}")
    return json.loads(proc.stdout.lstrip("\ufeff"))


def effective_window(law: dict, revision: dict | None) -> tuple[str | None, str | None]:
    """effective_to comes from the next official revision date, else repeal date."""
    revisions = law.get("revisions") or []
    start = (revision or {}).get("effective_at") or law.get("effective_at")
    if revision is None:
        return start, law.get("repealed_at")

    ordered = sorted(revisions, key=lambda r: (r.get("effective_at") or "9999"))
    for index, item in enumerate(ordered):
        if item.get("id") == revision.get("id"):
            if index + 1 < len(ordered):
                return start, ordered[index + 1].get("effective_at")
            return start, law.get("repealed_at")
    return start, law.get("repealed_at")


def build_records(slug: str, law: dict) -> list[dict]:
    meta = LAWS[slug]
    override = OVERRIDES.get(meta["chinalaw_id"], {})
    revision = law.get("selected_revision") or law.get("current_revision") or {}
    effective_from, effective_to = effective_window(law, revision or None)
    if override.get("effective_from") and not effective_from:
        effective_from = override["effective_from"]

    status = STATUS_MAP.get(law.get("status") or "unknown", "unknown")
    if override.get("status"):
        status = override["status"]

    source_hash = revision.get("content_hash") or law.get("source_hash")
    version_id = revision.get("id") or f"{slug}@{str(source_hash)[:12]}"

    rows: list[dict] = []
    for article in law.get("articles") or []:
        text = article.get("text") or ""
        number_raw = article.get("number")
        try:
            number: int | str = int(number_raw)
        except (TypeError, ValueError):
            number = number_raw

        rows.append(
            {
                "law_id": slug,
                "law_title": law.get("title") or meta["title"],
                "article_number": number,
                "article_number_display": article.get("number_display"),
                "part": article.get("part"),
                "version_id": version_id,
                "effective_from": effective_from,
                "effective_to": effective_to,
                "text": text,
                "status": status,
                "law_status_raw": law.get("status"),
                "source_url": law.get("source_url"),
                "source_name": law.get("source_name"),
                "source_hash": source_hash,
                "source_checked_at": law.get("source_checked_at"),
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "effective_from_basis": override.get("basis") if not revision.get("effective_at") else "chinalaw_revision_effective_at",
                "level": law.get("level"),
                "issuing_body": law.get("issuing_body"),
                "document_number": law.get("document_number"),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DATA / "labor_subset.jsonl"))
    parser.add_argument("--manifest", default=str(DATA / "manifest.json"))
    parser.add_argument("--include-history", action="store_true", help="also export superseded revisions via --as-of")
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    manifest: dict = {"generated_by": "export_subset.py", "laws": [], "totals": {}}

    for slug, meta in LAWS.items():
        law = run_get(meta["chinalaw_id"])
        rows = build_records(slug, law)
        all_rows.extend(rows)
        manifest["laws"].append(
            {
                "law_id": slug,
                "chinalaw_id": meta["chinalaw_id"],
                "title": law.get("title"),
                "level": law.get("level"),
                "status": law.get("status"),
                "effective_at": law.get("effective_at"),
                "repealed_at": law.get("repealed_at"),
                "source_url": law.get("source_url"),
                "source_hash": law.get("source_hash"),
                "source_checked_at": law.get("source_checked_at"),
                "article_count": len(rows),
                "revisions": law.get("revisions") or [],
            }
        )
        print(f"{slug:<34} {len(rows):>4} articles  status={law.get('status')}")

        if args.include_history:
            seen = {r["version_id"] for r in rows}
            for revision in law.get("revisions") or []:
                as_of = revision.get("effective_at")
                if not as_of or revision.get("id") in seen:
                    continue
                hist = run_get(meta["chinalaw_id"], as_of=as_of)
                hist_rev = hist.get("selected_revision") or {}
                hist_rows = build_records(slug, hist)
                for row in hist_rows:
                    row["version_id"] = hist_rev.get("id") or row["version_id"]
                all_rows.extend(hist_rows)
                print(f"  + history {as_of}: {len(hist_rows)} articles")

    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        for row in all_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest["totals"] = {
        "records": len(all_rows),
        "laws": len(LAWS),
        "with_effective_from": sum(1 for r in all_rows if r["effective_from"]),
        "missing_effective_from": sorted({r["law_id"] for r in all_rows if not r["effective_from"]}),
        "status_counts": {s: sum(1 for r in all_rows if r["status"] == s) for s in {r["status"] for r in all_rows}},
    }
    with open(args.manifest, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(f"\nwrote {args.out} ({len(all_rows)} records)")
    print(f"wrote {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
