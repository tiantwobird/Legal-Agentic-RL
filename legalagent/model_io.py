"""One place that decides how models and adapters are loaded.

Every entry point used to hardcode ``local_files_only=True``, which made a fresh
clone fail on its first run: nothing was in the Hugging Face cache, and the error
looked like a broken model path rather than "you are offline by default".

Now each script exposes ``--download`` (and the env var
``LEGALAGENT_ALLOW_DOWNLOAD=1`` does the same for every process), and the
decision is made here. Adapter loading lives here too: `--adapter` must mean
the same thing in collection, training and evaluation.

Transformers / torch imports stay inside the functions so that importing this
module is free — the unit tests and CI never pull the training stack.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_TRUE = {"1", "true", "yes", "on"}

ADAPTER_CONFIG = "adapter_config.json"
ALLOW_BASE_MISMATCH_ENV = "LEGALAGENT_ALLOW_BASE_MISMATCH"

_MISSING_HINT = (
    "{model_id} 不在本地缓存，且当前处于离线模式。\n"
    "首次运行请加 --download（或设置 LEGALAGENT_ALLOW_DOWNLOAD=1）让它下载，"
    "或者先把模型拉到本地。"
)


def allow_download(explicit: bool = False) -> bool:
    """True when the caller passed --download or the env var is set."""
    if explicit:
        return True
    return os.environ.get("LEGALAGENT_ALLOW_DOWNLOAD", "").strip().lower() in _TRUE


def load_tokenizer(model_id: str, download: bool = False):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(
            model_id, local_files_only=not allow_download(download)
        )
    except OSError as exc:
        raise SystemExit(_MISSING_HINT.format(model_id=model_id)) from exc


def load_model(model_id: str, download: bool = False, dtype=None):
    """Load a causal LM; `dtype` is passed through (bf16, "auto", ...)."""
    from transformers import AutoModelForCausalLM

    kwargs = {"local_files_only": not allow_download(download)}
    if dtype is not None:
        kwargs["dtype"] = dtype
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    except OSError as exc:
        raise SystemExit(_MISSING_HINT.format(model_id=model_id)) from exc


def _model_basename(model_id: str) -> str:
    """`Qwen/Qwen3-1.7B`, `./models/qwen3-1.7b` and a Windows path all compare equal."""
    return model_id.strip().replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].lower()


def check_adapter_dir(
    adapter_dir: str | Path,
    model_id: str | None = None,
    allow_mismatch: bool = False,
) -> dict:
    """Validate a LoRA adapter directory *before* importing torch.

    Checks that the directory exists, contains `adapter_config.json`, and (when
    `model_id` is given) that the adapter was trained on the same base model.
    A wrong base silently produces garbage LoRA weights, so this fails loudly.

    Pure stdlib, so tests and `--help` paths never need the training stack.

    Raises:
        SystemExit: with an actionable Chinese message, matching this repo's CLI style.
    """
    path = Path(adapter_dir)
    if not path.exists():
        raise SystemExit(
            f"adapter 目录不存在：{path}\n"
            "先跑上一阶段训练，或把 --adapter 指向 runs/<name>。"
        )
    if not path.is_dir():
        raise SystemExit(f"--adapter 需要目录，实际是文件：{path}")

    config_path = path / ADAPTER_CONFIG
    if not config_path.is_file():
        raise SystemExit(
            f"{path} 里没有 {ADAPTER_CONFIG}，看起来不是 PEFT/LoRA adapter 目录。\n"
            "合并后的完整模型不能当 adapter 用；检查是不是指到了基座模型或 checkpoint 的上级目录。"
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"读不了 {config_path}：{exc}") from exc
    if not isinstance(config, dict):
        raise SystemExit(f"{config_path} 不是一个 JSON 对象")

    base = config.get("base_model_name_or_path")
    mismatched = bool(
        model_id
        and isinstance(base, str)
        and base.strip()
        and _model_basename(base) != _model_basename(model_id)
    )
    if not allow_mismatch:
        allow_mismatch = os.environ.get(ALLOW_BASE_MISMATCH_ENV, "").strip().lower() in _TRUE
    if mismatched and not allow_mismatch:
        raise SystemExit(
            f"adapter 的底座是 {base}，但当前 --model 是 {model_id}。\n"
            "两者必须是同一个模型；确认同源（例如 hub id 与本地镜像路径）后可设 "
            f"{ALLOW_BASE_MISMATCH_ENV}=1 跳过此检查。"
        )
    return config


def load_adapter_model(model, adapter_dir: str | Path, trainable: bool = True):
    """Wrap `model` with an existing LoRA adapter.

    Training passes `trainable=True` and hands the resulting `PeftModel` to the
    trainer with `peft_config=None`: TRL rejects both together, and for GRPO it
    adds a frozen `ref` copy of the adapter so the KL term is measured against
    the policy this stage started from, not against the base model.
    Evaluation passes `trainable=False` to keep the adapter read-only.
    """
    from peft import PeftModel

    try:
        return PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=trainable)
    except Exception as exc:  # noqa: BLE001 - surface the adapter path, keep the cause
        raise SystemExit(
            f"加载 adapter 失败：{adapter_dir}\n"
            f"{type(exc).__name__}: {exc}\n"
            "确认 --model 与 adapter 的底座一致，且 adapter 文件完整。"
        ) from exc


def resolve_device(prefer: str = "cuda") -> str:
    """`prefer` if it is usable, otherwise cpu."""
    import torch

    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"
