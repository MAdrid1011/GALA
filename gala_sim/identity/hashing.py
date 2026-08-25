"""Content-addressed identity helpers.

Hashes are deliberately computed from bytes and canonical JSON rather than
from filesystem metadata, so a run can be reproduced on another host.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> bytes:
    """Encode JSON deterministically for manifests and configuration hashes."""

    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(root: Path) -> str:
    """Hash a directory by sorted relative paths, sizes, and file contents."""

    if not root.is_dir():
        raise ValueError(f"tree root is not a directory: {root}")
    digest = hashlib.sha256()
    generated_names = {"__pycache__", "build", "dist"}
    files = (item for item in root.rglob("*") if item.is_file()
             and not any(part in generated_names or part == ".git"
                         for part in item.relative_to(root).parts)
             and not any(part.endswith(".egg-info") for part in item.relative_to(root).parts))
    for path in sorted(files):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()
