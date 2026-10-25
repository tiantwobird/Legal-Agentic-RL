"""Single source of truth for repository layout.

Every module resolves inputs and outputs through these constants, so no script
depends on the current working directory or on being launched from its own
folder. Import the paths, never rebuild them with `Path(__file__)`.
"""

from __future__ import annotations

from pathlib import Path

#: Repository root: the directory that holds `legalagent/`, `data/`, `runs/`.
ROOT = Path(__file__).resolve().parents[1]

#: Curated corpora and generated datasets (JSONL, SQLite).
DATA = ROOT / "data"

#: Task splits (single active task set: 300/100/150/50).
TASKS = DATA / "tasks"

#: Training outputs (LoRA adapters, trainer state, per-run logs).
RUNS = ROOT / "runs"

#: Evaluation, analysis and baseline reports.
RESULTS = ROOT / "results"

#: Long-form documentation.
DOCS = ROOT / "docs"
