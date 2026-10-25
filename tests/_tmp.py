"""Test-scoped temporary directories.

`tempfile.mkdtemp()` is the obvious choice, but `os.mkdir(path, 0o700)` produces
a directory ACL that some Windows sandboxes refuse to write into — the test then
fails on setup with a confusing `PermissionError`, even though nothing is wrong
with the code under test.

So the tests create directories with a plain `Path.mkdir()`, which inherits the
parent's ACL, under `tests/.tmp-test/` (ignored by `.gitignore`). Cleanup is
best-effort: a failed rmtree must not turn a passing assertion into a test error.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

BASE = Path(__file__).resolve().parent / ".tmp-test"


def make_tmpdir() -> Path:
    BASE.mkdir(parents=True, exist_ok=True)
    path = BASE / f"t{uuid4().hex[:10]}"
    path.mkdir()
    return path


def drop_tmpdir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
    try:
        BASE.rmdir()  # 只在没有其它测试目录时成功；否则留给下一次
    except OSError:
        pass
