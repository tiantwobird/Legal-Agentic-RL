"""Deterministic reward for the labor-law RL pilot tasks.

Six components, each traceable to a field in `tasks_*_labeled.jsonl`:

    tool_use        did the model actually call a tool (ok=True)?
    grounding       are the cited article_ids present in tool output?
    gold            recall / precision against `gold_article_ids`
    answer_points   coverage of `expected_answer_points`
    insufficiency   for `insufficient_information` tasks: names missing facts + hedges
    hallucination   penalty for citing ids the tools never returned

Everything is pure Python and replayable: no LLM judge, no randomness, so a GRPO
run sees a stable signal. `gold_article_ids` is "necessary but not exhaustive"
per the dataset README, so extra grounded citations only cost a small precision
term instead of zeroing the score.

    python reward.py        # self-test on synthetic rollouts
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Longest aliases first: 劳动合同法实施条例 must win over 劳动合同法.
LAW_ALIASES: tuple[tuple[str, str], ...] = (
    ("劳动合同法实施条例", "labor_contract_law_regulation"),
    ("劳动争议调解仲裁法", "labor_dispute_arbitration_law"),
    ("劳动争议解释", "labor_dispute_interpretation_1"),
    ("劳动合同法", "labor_contract_law"),
    ("劳动法", "labor_law"),
)

CJK_NUM = {"零": 0, "〇": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
           "六": 6, "七": 7, "八": 8, "九": 9}
UNITS = {"十": 10, "百": 100, "千": 1000}

ID_RE = re.compile(r"\b([a-z][a-z0-9_]*)\s*[:：]\s*(\d+)")
NAMED_RE = re.compile(r"《?([\u4e00-\u9fff]{2,20}?)》?\s*第\s*([零〇一二三四五六七八九十百\d]+)\s*条")

HEDGE_MARKERS = ("无法判断", "不足以", "缺少", "未提供", "需要补充", "无法确定",
                 "信息不足", "未说明", "不详", "补充材料", "难以认定", "取决于")


def _cjk_number(raw: str) -> int | None:
    if raw.isdigit():
        return int(raw)
    if not raw or any(char not in CJK_NUM and char not in UNITS for char in raw):
        return None
    total, current = 0, 0
    for char in raw:
        if char in CJK_NUM:
            current = CJK_NUM[char]
        else:
            total += (current or 1) * UNITS[char]
            current = 0
    return total + current


def normalize(text: str) -> str:
    return "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", text or ""))


def bigrams(text: str) -> set[str]:
    text = normalize(text)
    if len(text) < 2:
        return {text} if text else set()
    return {text[i:i + 2] for i in range(len(text) - 1)}


def coverage(needle: str, haystack: str) -> float:
    """Fraction of the needle's character bigrams that appear in haystack."""
    target = bigrams(needle)
    if not target:
        return 0.0
    return len(target & bigrams(haystack)) / len(target)


def extract_citations(answer: str) -> list[str]:
    """Pull article ids from `law_id:39` and from 《劳动合同法》第三十九条 style text."""
    found: list[str] = []
    for law_id, number in ID_RE.findall(answer or ""):
        found.append(f"{law_id}:{int(number)}")
    for name, raw_number in NAMED_RE.findall(answer or ""):
        number = _cjk_number(raw_number)
        if number is None:
            continue
        for alias, law_id in LAW_ALIASES:
            if alias in name:
                found.append(f"{law_id}:{number}")
                break
    seen: set[str] = set()
    ordered: list[str] = []
    for item in found:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _f1(recall: float, precision: float) -> float:
    if recall <= 0 or precision <= 0:
        return 0.0
    return 2 * recall * precision / (recall + precision)


def split_claims(point: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"[；;]", point) if p.strip()]
    return parts or [point]


@dataclass
class RewardConfig:
    w_tool_use: float = 0.10
    w_grounding: float = 0.10
    w_gold: float = 0.35
    w_points: float = 0.35
    w_insufficiency: float = 0.10
    point_threshold: float = 0.50
    point_scale: float = 0.60
    binary_points: bool = False
    fact_threshold: float = 0.45
    hallucination_penalty: float = 0.15
    hallucination_cap: float = 0.45
    excess_citation_free: int = 3
    excess_citation_penalty: float = 0.03
    excess_citation_cap: float = 0.15
    no_citation_penalty: float = 0.20
    min_total: float = -0.5
    max_total: float = 1.0


@dataclass
class RewardBreakdown:
    total: float
    components: dict[str, float]
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"total": round(self.total, 4), "components": self.components, "details": self.details}


def score_rollout(task: dict[str, Any], rollout: dict[str, Any], cfg: RewardConfig | None = None) -> RewardBreakdown:
    """Score one rollout.

    rollout needs: answer (str), tool_ids (list[str]), tool_calls (list[dict]).
    """
    cfg = cfg or RewardConfig()
    answer = rollout.get("answer") or ""
    tool_ids = set(rollout.get("tool_ids") or [])
    calls = rollout.get("tool_calls") or []
    if not calls:
        calls = [c for entry in rollout.get("trace", []) for c in entry.get("tool_calls", [])]

    successful_calls = [c for c in calls if c.get("ok")]
    cited = extract_citations(answer)
    grounded = [c for c in cited if c in tool_ids]
    hallucinated = [c for c in cited if c not in tool_ids]

    gold = list(task.get("gold_article_ids") or [])
    gold_hit = [c for c in grounded if c in gold]
    gold_recall = len(set(gold_hit)) / len(set(gold)) if gold else 0.0
    gold_precision = len(set(gold_hit)) / len(set(grounded)) if grounded else 0.0

    points = list(task.get("expected_answer_points") or [])
    claim_scores: list[float] = []
    missing_claims: list[str] = []
    for point in points:
        for claim in split_claims(point):
            ratio = coverage(claim, answer)
            if cfg.binary_points:
                score = 1.0 if ratio >= cfg.point_threshold else 0.0
            else:
                # Continuous: statutory wording paraphrases poorly, so partial overlap
                # gets partial credit instead of a cliff at 0.5.
                score = min(1.0, ratio / cfg.point_scale) if cfg.point_scale else 0.0
            claim_scores.append(score)
            if score < 0.5:
                missing_claims.append(claim)
    points_score = sum(claim_scores) / len(claim_scores) if claim_scores else 0.0

    components: dict[str, float] = {
        "tool_use": 1.0 if successful_calls else 0.0,
        "grounding": (len(grounded) / len(cited)) if cited else 0.0,
        "gold": _f1(gold_recall, gold_precision),
        "answer_points": points_score,
    }

    insufficiency_score: float | None = None
    missing_facts = list(task.get("missing_facts") or [])
    if task.get("task_type") == "insufficient_information" or missing_facts:
        ratios = [min(1.0, coverage(fact, answer) / cfg.point_scale) if cfg.point_scale else 0.0
                  for fact in missing_facts]
        covered = [fact for fact, ratio in zip(missing_facts, ratios) if ratio >= 0.5]
        fact_ratio = sum(ratios) / len(ratios) if ratios else 0.0
        hedged = any(marker in answer for marker in HEDGE_MARKERS)
        insufficiency_score = 0.6 * fact_ratio + 0.4 * (1.0 if hedged else 0.0)
        components["insufficiency"] = insufficiency_score
        details_facts = {"missing_facts_covered": covered, "hedged": hedged}
    else:
        details_facts = {}

    weights = {
        "tool_use": cfg.w_tool_use,
        "grounding": cfg.w_grounding,
        "gold": cfg.w_gold,
        "answer_points": cfg.w_points,
        "insufficiency": cfg.w_insufficiency,
    }
    active = {name: weights[name] for name in components}
    weight_sum = sum(active.values())
    base = sum(components[name] * active[name] for name in components) / weight_sum if weight_sum else 0.0

    extra_citations = [c for c in grounded if c not in gold]
    excess = max(0, len(extra_citations) - cfg.excess_citation_free)
    excess_penalty = min(cfg.excess_citation_cap, cfg.excess_citation_penalty * excess)
    no_citation = (not cited) and task.get("task_type") != "insufficient_information"
    penalty = (
        min(cfg.hallucination_cap, cfg.hallucination_penalty * len(hallucinated))
        + excess_penalty
        + (cfg.no_citation_penalty if no_citation else 0.0)
    )
    total = max(cfg.min_total, min(cfg.max_total, base - penalty))

    return RewardBreakdown(
        total=total,
        components={k: round(v, 4) for k, v in components.items()},
        details={
            "task_id": task.get("task_id"),
            "cited": cited,
            "grounded": grounded,
            "hallucinated": hallucinated,
            "gold": gold,
            "gold_missing": [g for g in gold if g not in grounded],
            "missing_claims": missing_claims,
            "penalty": round(penalty, 4),
            "extra_citations": extra_citations,
            "excess_citations": excess,
            "no_citation": no_citation,
            "tool_calls": [c.get("name") for c in calls],
            **details_facts,
        },
    )


def reward_fn(tasks_by_id: dict[str, dict], cfg: RewardConfig | None = None):
    """TRL-style adapter factory: reward_fn(samples, **kwargs) -> list[float].

    Each sample must carry `task_id` in kwargs (TRL passes dataset columns through)
    so the gold labels can be looked up without ever putting them in the prompt.
    """

    def _fn(completions, task_id=None, **kwargs):
        ids = task_id if isinstance(task_id, list) else [task_id] * len(completions)
        scores = []
        for completion, tid in zip(completions, ids):
            task = tasks_by_id[tid]
            rollout = completion if isinstance(completion, dict) else {"answer": str(completion)}
            scores.append(score_rollout(task, rollout, cfg).total)
        return scores

    return _fn


if __name__ == "__main__":
    task_insufficient = {
        "task_id": "T_INS",
        "task_type": "insufficient_information",
        "gold_article_ids": ["labor_contract_law:39", "labor_contract_law:4"],
        "expected_answer_points": ["不能仅凭迟到就认定严重违纪，需要核查规章制度合法性与公示、迟到严重程度及证据"],
        "missing_facts": ["合法有效的规章制度及其公示", "迟到次数与严重程度", "相关证据"],
    }
    task_plain = {
        "task_id": "T_PLAIN",
        "task_type": "fact_application",
        "gold_article_ids": ["labor_contract_law:9", "labor_contract_law:84"],
        "expected_answer_points": ["不得要求劳动者提供担保或者以其他名义收取财物；违法收取押金的，责令退还并处罚款"],
        "missing_facts": [],
    }

    good = {
        "tool_calls": [{"name": "search_statute", "ok": True}, {"name": "get_statute", "ok": True}],
        "tool_ids": ["labor_contract_law:9", "labor_contract_law:84", "labor_contract_law:39"],
        "answer": "用人单位不得要求劳动者提供担保或者以其他名义向劳动者收取财物，违法收取押金的应当退还并处罚款。"
                  "引用：labor_contract_law:9、labor_contract_law:84。",
    }
    lazy = {"tool_calls": [], "tool_ids": [], "answer": "公司收押金违法，应退还。引用：labor_contract_law:87。"}
    insuff = {
        "tool_calls": [{"name": "search_statute", "ok": True}],
        "tool_ids": ["labor_contract_law:39", "labor_contract_law:4"],
        "answer": "仅凭迟到不能直接认定严重违纪。需要补充：合法有效的规章制度及其公示、迟到次数与严重程度、相关证据，"
                  "否则无法判断解除是否合法。引用：labor_contract_law:39、labor_contract_law:4。",
    }

    for name, task, rollout in [
        ("good (fact_application)", task_plain, good),
        ("lazy (no tool, fake id)", task_plain, lazy),
        ("good (insufficient)", task_insufficient, insuff),
    ]:
        result = score_rollout(task, rollout)
        print(f"{name:28} total={result.total:+.3f}  components={result.components}")
        if result.details["hallucinated"]:
            print(f"{'':28} hallucinated={result.details['hallucinated']}")
