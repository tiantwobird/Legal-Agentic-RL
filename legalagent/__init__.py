"""LegalAgent: a version-aware labour-law retrieval agent with GRPO training.

Layers (see REPO_MAP.md):

    data/      export -> SQLite -> reviewed task splits
    tools/     the two deterministic statute tools the agent may call
    reward.py  deterministic six-component reward
    train/     single-turn and agentic GRPO
    eval/      rollout scoring and agentic evaluation
    analysis/  bad-case classification and curriculum sampling
"""

__version__ = "0.1.0"
