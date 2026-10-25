"""Adapter-directory validation, pure stdlib: no torch, no peft, no GPU."""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest import mock

from ._tmp import drop_tmpdir, make_tmpdir
from legalagent.model_io import ALLOW_BASE_MISMATCH_ENV, check_adapter_dir


def write_adapter(root: Path, **overrides) -> Path:
    config = {
        "peft_type": "LORA",
        "r": 16,
        "base_model_name_or_path": "Qwen/Qwen3-1.7B",
    }
    config.update(overrides)
    path = root / "adapter"
    path.mkdir()
    (path / "adapter_config.json").write_text(
        json.dumps(config, ensure_ascii=False), encoding="utf-8"
    )
    return path


class CheckAdapterDirTests(unittest.TestCase):
    def setUp(self):
        self.tmp = make_tmpdir()

    def tearDown(self):
        drop_tmpdir(self.tmp)

    def test_returns_config_for_a_valid_adapter(self):
        path = write_adapter(self.tmp)
        config = check_adapter_dir(path, "Qwen/Qwen3-1.7B")
        self.assertEqual(config["peft_type"], "LORA")
        self.assertEqual(config["base_model_name_or_path"], "Qwen/Qwen3-1.7B")

    def test_missing_directory_fails_before_torch(self):
        with self.assertRaises(SystemExit) as ctx:
            check_adapter_dir(self.tmp / "nope", "Qwen/Qwen3-1.7B")
        self.assertIn("不存在", str(ctx.exception))

    def test_file_instead_of_directory_is_rejected(self):
        path = self.tmp / "adapter"
        path.write_text("not a dir", encoding="utf-8")
        with self.assertRaises(SystemExit):
            check_adapter_dir(path, "Qwen/Qwen3-1.7B")

    def test_missing_adapter_config_is_rejected(self):
        path = self.tmp / "adapter"
        path.mkdir()
        with self.assertRaises(SystemExit) as ctx:
            check_adapter_dir(path, "Qwen/Qwen3-1.7B")
        self.assertIn("adapter_config.json", str(ctx.exception))

    def test_broken_json_is_rejected(self):
        path = self.tmp / "adapter"
        path.mkdir()
        (path / "adapter_config.json").write_text("{", encoding="utf-8")
        with self.assertRaises(SystemExit):
            check_adapter_dir(path, "Qwen/Qwen3-1.7B")

    def test_wrong_base_model_is_rejected(self):
        path = write_adapter(self.tmp, base_model_name_or_path="Qwen/Qwen3-0.6B")
        with self.assertRaises(SystemExit) as ctx:
            check_adapter_dir(path, "Qwen/Qwen3-1.7B")
        self.assertIn("底座", str(ctx.exception))

    def test_local_mirror_path_counts_as_the_same_base(self):
        # Hub id in the adapter config, local snapshot path on the CLI (or vice versa).
        path = write_adapter(self.tmp, base_model_name_or_path=r"C:\models\Qwen3-1.7B")
        config = check_adapter_dir(path, "Qwen/Qwen3-1.7B")
        self.assertEqual(config["r"], 16)

    def test_env_escape_hatch_allows_a_mismatch(self):
        path = write_adapter(self.tmp, base_model_name_or_path="Qwen/Qwen3-0.6B")
        with mock.patch.dict(os.environ, {ALLOW_BASE_MISMATCH_ENV: "1"}):
            config = check_adapter_dir(path, "Qwen/Qwen3-1.7B")
        self.assertEqual(config["base_model_name_or_path"], "Qwen/Qwen3-0.6B")

    def test_explicit_flag_allows_a_mismatch(self):
        path = write_adapter(self.tmp, base_model_name_or_path="Qwen/Qwen3-0.6B")
        config = check_adapter_dir(path, "Qwen/Qwen3-1.7B", allow_mismatch=True)
        self.assertIsInstance(config, dict)


if __name__ == "__main__":
    unittest.main()
