"""Load a curriculum file produced by `analysis.bad_case_sampler`.

The sampler emits *prompt-only* rows (`task_id`, `prompt`, `jurisdiction`,
`applicable_date`) because gold labels must never reach the model. Training still
needs the labels to compute reward, so they are joined back by `task_id` here —
and the prompt is always taken from the curriculum file, never from the label
file, so a label that accidentally embeds answer hints cannot leak into the
prompt.

A curriculum row may repeat a `task_id` (the sampler draws with replacement);
repeats are preserved, and the trainer sees the task that many times. For GRPO a
repeated prompt is not a duplicated batch item — the trainer still generates G
fresh completions for every occurrence, which is exactly the intended weighting.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..paths import TASKS


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_curriculum(
    curriculum_path: str | Path,
    labels_path: str | Path | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Join a prompt-only curriculum with the train labels, preserving order and repeats.

    Raises:
        KeyError: a curriculum `task_id` is not present in the label file. Failing
            loudly beats silently training on a smaller set than requested.
        ValueError: the curriculum produced no rows.
    """
    labels_path = Path(labels_path) if labels_path else TASKS / "tasks_train_labeled.jsonl"
    labels = {row["task_id"]: row for row in read_jsonl(labels_path)}

    rows: list[dict] = []
    missing: list[str] = []
    for item in read_jsonl(curriculum_path):
        task_id = item["task_id"]
        label = labels.get(task_id)
        if label is None:
            missing.append(task_id)
            continue
        rows.append({**label, "prompt": item["prompt"]})

    if missing:
        unique = sorted(set(missing))
        raise KeyError(
            f"{curriculum_path}: {len(unique)} task_id(s) not in {labels_path}: {unique[:5]}"
        )
    if not rows:
        raise ValueError(f"{curriculum_path}: no usable rows")
    return rows[:limit] if limit else rows
