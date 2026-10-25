"""CLI contract: a curriculum stage must be able to continue from an adapter.

These tests only build the parsers, so they need neither torch nor a GPU.
"""

from __future__ import annotations

import unittest

from legalagent.train import train_grpo, train_grpo_agentic


class TrainerAdapterFlagTests(unittest.TestCase):
    def test_agentic_defaults_to_no_adapter(self):
        args = train_grpo_agentic.build_parser().parse_args([])
        self.assertEqual(args.adapter, "")

    def test_agentic_accepts_an_adapter_directory(self):
        args = train_grpo_agentic.build_parser().parse_args(["--adapter", "runs/stage0"])
        self.assertEqual(args.adapter, "runs/stage0")

    def test_agentic_still_accepts_curriculum(self):
        args = train_grpo_agentic.build_parser().parse_args(
            ["--adapter", "runs/stage0", "--curriculum", "data/tasks/curriculum_train.jsonl"]
        )
        self.assertEqual(args.curriculum, "data/tasks/curriculum_train.jsonl")

    def test_single_turn_trainer_has_the_same_flag(self):
        args = train_grpo.build_parser().parse_args(["--adapter", "runs/stage0"])
        self.assertEqual(args.adapter, "runs/stage0")


if __name__ == "__main__":
    unittest.main()
