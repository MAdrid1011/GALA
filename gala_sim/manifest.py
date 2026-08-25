"""Input-freeze records for the first R²-Gaussian + Chest combination."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import platform
import subprocess
import copy
from pathlib import Path
from typing import Any

from gala_sim.config import GalaConfig
from gala_sim.identity import canonical_json, sha256_bytes, sha256_file, sha256_tree


@dataclass(frozen=True)
class SourceRecord:
    name: str
    url: str
    commit: str
    root: str
    tree_sha256: str
    upstream_patch_sha256: str
    license_path: str
    license_sha256: str


@dataclass(frozen=True)
class DatasetRecord:
    name: str
    status: str
    source_url: str
    license_url: str
    root: str | None
    manifest_sha256: str | None
    reason: str | None
    files: list[dict[str, Any]] | None = None
    metadata_sha256: str | None = None
    geometry: dict[str, Any] | None = None


def _command(*args: str) -> str | None:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_snapshot() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git": _command("git", "--version"),
        "cuda": _command("nvcc", "--version"),
        "torch": _command("python", "-c", "import torch; print(torch.__version__)"),
        "numba": _command("python", "-c", "import numba; print(numba.__version__)"),
        "ramulator2": None,
    }


def source_record(root: Path, name: str, url: str, commit: str) -> SourceRecord:
    root = root.resolve()
    actual = _command("git", "-C", str(root), "rev-parse", "HEAD")
    if actual != commit:
        raise ValueError(f"{name} is at {actual!r}, expected {commit!r}")
    dirty = _command("git", "-C", str(root), "status", "--porcelain")
    if dirty:
        raise ValueError(f"{name} checkout has uncommitted changes")
    license_path = root / "LICENSE.md"
    if not license_path.is_file():
        raise ValueError(f"{name} license file is missing: {license_path}")
    return SourceRecord(name, url, commit, str(root), sha256_tree(root),
                        sha256_bytes(b""), str(license_path), sha256_file(license_path))


def dataset_record(root: Path | None, name: str, source_url: str, license_url: str,
                   reason: str | None = None) -> DatasetRecord:
    if root is None:
        if not reason:
            raise ValueError("a missing dataset requires a machine-readable reason")
        return DatasetRecord(name, "unavailable_data", source_url, license_url, None, None, reason)
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"dataset root is not a directory: {root}")
    metadata_path = root / "meta_data.json"
    metadata: dict[str, Any] | None = None
    metadata_digest: str | None = None
    geometry: dict[str, Any] | None = None
    if metadata_path.is_file():
        try:
            parsed = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"dataset metadata is not valid JSON: {metadata_path}") from error
        if not isinstance(parsed, dict):
            raise ValueError("dataset metadata root must be an object")
        metadata = parsed
        metadata_digest = sha256_file(metadata_path)
        if isinstance(parsed.get("scanner"), dict):
            geometry = parsed["scanner"]
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()
                       and ".git" not in item.relative_to(root).parts):
        files.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                      "sha256": sha256_file(path)})
    return DatasetRecord(name, "planned", source_url, license_url, str(root), sha256_tree(root), None,
                         files, metadata_digest, geometry)


def build_freeze_record(config: GalaConfig, source: SourceRecord, dataset: DatasetRecord,
                        seed: int, repository: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": "gala-input-freeze-v1",
        "workflow_step": "freeze_inputs",
        "status": "planned" if dataset.status == "planned" and config.ready else dataset.status,
        "model": asdict(source),
        "dataset": asdict(dataset),
        "config": {"path": str(config.path), "sha256": config.sha256, "ready": config.ready},
        "random_seed": seed,
        "repository": {"root": str(repository.resolve()), "commit": _command("git", "-C", str(repository), "rev-parse", "HEAD")},
        "environment": environment_snapshot(),
    }
    record["run_manifest_sha256"] = sha256_bytes(canonical_json(record))
    return record


def write_freeze_record(record: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def verify_freeze_record(record: dict[str, Any]) -> None:
    """Check the self-hash before a record is used as a workflow input."""

    recorded = record.get("run_manifest_sha256")
    if not isinstance(recorded, str):
        raise ValueError("freeze record has no run_manifest_sha256")
    unsigned = copy.deepcopy(record)
    del unsigned["run_manifest_sha256"]
    if sha256_bytes(canonical_json(unsigned)) != recorded:
        raise ValueError("freeze record self-hash does not match its contents")
