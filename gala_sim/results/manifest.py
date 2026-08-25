"""Stable run identity and output helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from gala_sim.identity import canonical_json, sha256_bytes


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    status: str
    model: dict[str, Any]
    dataset: dict[str, Any]
    config_sha256: str
    ablation_bits: str
    random_seed: int
    repository_commit: str
    environment: dict[str, Any]
    trace_sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["manifest_sha256"] = sha256_bytes(canonical_json(value))
        return value


def write_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
                    encoding="utf-8")
