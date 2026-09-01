#!/usr/bin/env python3
"""Atomically migrate a legacy local runtime into the ignored workspace."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gala_sim.workspace import WorkspacePaths


@dataclass(frozen=True)
class ProcessUse:
    pid: int
    kind: str
    path: str


@dataclass(frozen=True)
class MigrationReport:
    source: str
    destination: str
    moved: bool
    blockers: tuple[ProcessUse, ...]
    relocated: tuple[tuple[str, str], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "destination": self.destination,
            "moved": self.moved,
            "blockers": [asdict(item) for item in self.blockers],
            "relocated": [list(item) for item in self.relocated],
        }


def find_process_uses(root: Path) -> tuple[ProcessUse, ...]:
    """Return open files and process working directories below root."""

    root = root.resolve()
    found: set[ProcessUse] = set()
    try:
        completed = subprocess.run(
            ["lsof", "-Fpn", "+D", str(root)],
            text=True, capture_output=True, check=False,
        )
    except FileNotFoundError as error:
        raise RuntimeError("lsof is required for safe workspace migration") from error
    if completed.returncode not in {0, 1}:
        raise RuntimeError("lsof could not inspect the legacy runtime")
    output = completed.stdout
    pid: int | None = None
    for line in output.splitlines():
        if line.startswith("p") and line[1:].isdigit():
            pid = int(line[1:])
        elif line.startswith("n") and pid is not None:
            path = Path(line[1:])
            if _is_within(path, root):
                found.add(ProcessUse(pid, "open_file", str(path)))

    for entry in Path("/proc").glob("[0-9]*/cwd"):
        try:
            cwd = entry.resolve(strict=True)
        except (FileNotFoundError, PermissionError, OSError):
            continue
        if _is_within(cwd, root):
            found.add(ProcessUse(int(entry.parent.name), "working_directory", str(cwd)))
    return tuple(sorted(found, key=lambda item: (item.pid, item.kind, item.path)))


def migrate_workspace(
    source: Path,
    repository: Path,
    *,
    dry_run: bool = False,
    process_finder: Callable[[Path], Iterable[ProcessUse]] = find_process_uses,
) -> MigrationReport:
    source = source.resolve()
    paths = WorkspacePaths.discover(repository=repository)
    destination = paths.root
    if source == destination:
        raise ValueError("source is already the managed workspace")
    if not source.is_dir():
        raise FileNotFoundError(f"legacy runtime is missing: {source}")
    if destination.exists():
        raise FileExistsError(f"workspace destination already exists: {destination}")
    if source.stat().st_dev != paths.repository.stat().st_dev:
        raise ValueError("legacy runtime and repository are on different filesystems")

    blockers = tuple(process_finder(source))
    if blockers:
        return MigrationReport(str(source), str(destination), False, blockers, ())
    if dry_run:
        return MigrationReport(str(source), str(destination), False, (), ())

    source.replace(destination)
    paths.ensure()
    relocated = _normalize_legacy_layout(paths)
    return MigrationReport(str(source), str(destination), True, (), tuple(relocated))


def _normalize_legacy_layout(paths: WorkspacePaths) -> list[tuple[str, str]]:
    root = paths.root
    moved: list[tuple[str, str]] = []

    chest = root / "data/chest"
    archive = chest / "cone_ntrain_50_angle_360.zip"
    if archive.is_file():
        _move(archive, paths.cache / "downloads/chest" / archive.name, root, moved)
    extracted = chest / "extracted/cone_ntrain_50_angle_360"
    if extracted.is_dir():
        _move(extracted, paths.datasets / "chest", root, moved)

    ramulator = root / "ramulator2"
    if ramulator.is_dir():
        _move(ramulator, paths.upstream / "ramulator2", root, moved)

    legacy = paths.archive / "legacy"
    managed = {path.name for path in paths.managed_directories}
    for item in sorted(root.iterdir(), key=lambda path: path.name):
        if item.name not in managed:
            _move(item, legacy / item.name, root, moved)

    _move_nonempty(root / "data", legacy / "data", root, moved)
    return moved


def _move_nonempty(
    source: Path, destination: Path, root: Path, moved: list[tuple[str, str]],
) -> None:
    if not source.exists():
        return
    for directory in sorted(
        (item for item in source.rglob("*") if item.is_dir()), reverse=True,
    ):
        try:
            directory.rmdir()
        except OSError:
            pass
    try:
        source.rmdir()
    except OSError:
        _move(source, destination, root, moved)


def _move(
    source: Path, destination: Path, root: Path, moved: list[tuple[str, str]],
) -> None:
    if destination.exists():
        raise FileExistsError(f"migration destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.replace(destination)
    moved.append((source.relative_to(root).as_posix(), destination.relative_to(root).as_posix()))


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root)
        return True
    except (ValueError, OSError):
        return False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = migrate_workspace(args.source, args.repository, dry_run=args.dry_run)
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as error:
        print(f"migrate_workspace: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report.as_dict(), sort_keys=True))
    return 2 if report.blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
