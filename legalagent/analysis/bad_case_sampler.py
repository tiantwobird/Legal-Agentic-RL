#!/usr/bin/env python3
"""Bad-case-aware *task* sampling for a Legal Agent GRPO curriculum.

This is a stage-level task sampler, NOT a GRPO trainer or replay buffer.
Weights are computed from fresh, on-policy TRAIN rollouts and the output
contains model-visible prompts only. Never feed gold labels to the model.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import random
import statistics
from pathlib import Path

from ..paths import RESULTS

ERRORS = {"retrieval_error", "citation_error", "reasoning_error", "tool_planning_error"}


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def save_jsonl(path: str | Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_group(row: dict, g: int | None = None) -> None:
    rewards = row.get("rewards")
    successes = row.get("successes")
    errors = row.get("error_types")
    if not isinstance(rewards, list) or len(rewards) < 2:
        raise ValueError(f"{row.get('task_id')}: rewards must contain >=2 floats")
    if g is not None and len(rewards) != g:
        raise ValueError(f"{row.get('task_id')}: wrong group size")
    if not all(isinstance(r, (int, float)) and math.isfinite(r) for r in rewards):
        raise ValueError(f"{row.get('task_id')}: nonfinite reward")
    if not isinstance(successes, list) or len(successes) != len(rewards) or any(s not in (0, 1) for s in successes):
        raise ValueError(f"{row.get('task_id')}: successes must be aligned 0/1")
    if not isinstance(errors, list) or len(errors) != len(rewards):
        raise ValueError(f"{row.get('task_id')}: error_types must align with rewards")
    for i, issue_list in enumerate(errors):
        if not isinstance(issue_list, list) or any(issue not in ERRORS for issue in issue_list):
            raise ValueError(f"{row.get('task_id')}: invalid error_types[{i}]")


def compute_weights(labels: list[dict], groups: list[dict], explore=0.30,
                    signal_scale=0.20) -> tuple[dict[str, float], dict]:
    """Create a mixture of uniform exploration and reweighted tasks.

    The signal is group reward dispersion (not simply answer correctness).
    Homogeneous *reward* groups offer no outcome-reward advantage; extremely
    hard tasks are kept through the uniform floor but not prioritized.
    """
    if not 0 <= explore <= 1:
        raise ValueError("explore must be between 0 and 1")
    if signal_scale <= 0:
        raise ValueError("signal_scale must be >0")
    by_id = {row["task_id"]: row for row in labels}
    if len(by_id) != len(labels):
        raise ValueError("duplicate ids in train labels")
    grouped: dict[str, dict] = {}
    versions = set()
    error_freq = Counter()
    failed_rollouts = 0
    g = None
    for row in groups:
        task_id = row.get("task_id")
        if task_id not in by_id:
            raise ValueError(f"not a train task: {task_id}")
        if task_id in grouped:
            raise ValueError(f"duplicate group: {task_id}")
        version = row.get("policy_version")
        if not isinstance(version, str) or not version.strip():
            raise ValueError(f"{task_id}: missing policy_version")
        versions.add(version)
        if g is None:
            g = len(row.get("rewards", []))
        validate_group(row, g=g)
        grouped[task_id] = row
        for successful, issue_list in zip(row["successes"], row["error_types"]):
            if not successful:
                failed_rollouts += 1
                error_freq.update(set(issue_list))
    if len(versions) != 1:
        raise ValueError("rollout results must come from exactly one policy version")
    scores = {}
    diagnostics = {}
    for task_id in by_id:
        if task_id not in grouped:
            scores[task_id] = 1.0
            diagnostics[task_id] = {"measured": False, "reason": "unobserved; neutral weight"}
            continue
        row = grouped[task_id]
        success_rate = sum(row["successes"]) / g
        reward_std = statistics.pstdev(row["rewards"])
        signal = min(1.0, reward_std / signal_scale)
        failed_share = 1 - success_rate
        local_errors = set(issue for issue_list in row["error_types"] for issue in issue_list)
        category_component = max((error_freq[e] / max(1, failed_rollouts) for e in local_errors), default=0)
        w = 1 + 1.5 * signal + .5 * failed_share + .7 * category_component
        if reward_std < 1e-8:
            w *= .35  # zero within-group outcome-reward advantage
        if success_rate == 1:
            w *= .55
        w = max(.4, min(3.0, w))
        scores[task_id] = w
        diagnostics[task_id] = {
            "measured": True, "success_rate": success_rate,
            "reward_std": round(reward_std, 6),
            "signal": round(signal, 4),
            "errors": sorted(local_errors), "raw_weight_capped": round(w, 5),
            "group_has_reward_signal": reward_std >= 1e-8,
        }
    total = sum(scores.values())
    n = len(scores)
    probabilities = {task_id: explore / n + (1 - explore) * w / total for task_id, w in scores.items()}
    assert math.isclose(sum(probabilities.values()), 1.0)
    audit = {
        "policy_version": next(iter(versions)) if versions else None,
        "group_size": g,
        "num_train_tasks": n,
        "num_measured_train_tasks": len(grouped),
        "num_homogeneous_reward_groups": sum(not v["group_has_reward_signal"] for v in diagnostics.values() if v["measured"]),
        "explore_fraction": explore,
        "error_counts_across_failed_rollouts": dict(error_freq),
        "weights": {k: round(v, 6) for k, v in scores.items()},
        "probabilities": probabilities,
        "diagnostics": diagnostics,
    }
    return probabilities, audit


def sample_curriculum(prompts: list[dict], probabilities: dict[str, float], count: int, seed: int) -> list[dict]:
    by_id = {row["task_id"]: row for row in prompts}
    if len(by_id) != len(prompts) or set(by_id) != set(probabilities):
        raise ValueError("prompts and probability IDs must match exactly")
    ids = sorted(by_id)
    chosen = random.Random(seed).choices(ids, weights=[probabilities[i] for i in ids], k=count)
    # New *prompts* for the next on-policy stage, not old trajectories.
    return [by_id[i].copy() for i in chosen]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--groups", required=True)
    ap.add_argument("--output", default=str(RESULTS / "curriculum_prompts_only.jsonl"))
    ap.add_argument("--audit", default=str(RESULTS / "curriculum_audit.json"))
    ap.add_argument("--count", type=int, default=35)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--explore", type=float, default=.30)
    args = ap.parse_args()
    labels = read_jsonl(args.labels)
    prompts = read_jsonl(args.prompts)
    groups = read_jsonl(args.groups)
    probs, audit = compute_weights(labels, groups, explore=args.explore)
    selected = sample_curriculum(prompts, probs, args.count, args.seed)
    save_jsonl(args.output, selected)
    audit["chosen_task_counts"] = dict(Counter(x["task_id"] for x in selected))
    Path(args.audit).write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ["policy_version", "num_train_tasks", "num_measured_train_tasks", "num_homogeneous_reward_groups", "chosen_task_counts"]}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
