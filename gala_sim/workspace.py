"""Portable repository and local-workspace discovery."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable


_DIRECTORY_NAMES = (
    "upstream", "datasets", "build", "cache", "traces", "profiles",
    "results", "archive",
)


def _repository_from(start: Path) -> Path | None:
    candidate = start.resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for root in (candidate, *candidate.parents):
        if (root / "pyproject.toml").is_file() and (root / "configs").is_dir():
            return root
    return None


@dataclass(frozen=True)
class WorkspacePaths:
    """Resolved repository and ignored workspace directories."""

    repository: Path
    root: Path

    @classmethod
    def discover(
        cls,
        repository: Path | None = None,
        workspace: Path | None = None,
    ) -> "WorkspacePaths":
        """Resolve repository and workspace without depending on process CWD."""

        search_points: Iterable[Path]
        if repository is not None:
            search_points = (Path(repository),)
        else:
            search_points = (Path.cwd(), Path(__file__).resolve())
        repository_root = next(
            (found for point in search_points if (found := _repository_from(point)) is not None),
            None,
        )
        if repository_root is None:
            raise FileNotFoundError(
                "GALA repository root was not found; pass an explicit repository path"
            )

        selected = workspace
        if selected is None:
            environment = os.environ.get("GALA_WORKSPACE")
            selected = Path(environment).expanduser() if environment else None
        if selected is None:
            selected = repository_root / "workspace"
        selected = Path(selected).expanduser()
        if not selected.is_absolute():
            selected = repository_root / selected
        return cls(repository_root.resolve(), selected.resolve())

    @property
    def upstream(self) -> Path:
        return self.root / "upstream"

    @property
    def datasets(self) -> Path:
        return self.root / "datasets"

    @property
    def build(self) -> Path:
        return self.root / "build"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    @property
    def traces(self) -> Path:
        return self.root / "traces"

    @property
    def profiles(self) -> Path:
        return self.root / "profiles"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def archive(self) -> Path:
        return self.root / "archive"

    @property
    def managed_directories(self) -> tuple[Path, ...]:
        return tuple(self.root / name for name in _DIRECTORY_NAMES)

    def ensure(self) -> "WorkspacePaths":
        self.root.mkdir(parents=True, exist_ok=True)
        for directory in self.managed_directories:
            directory.mkdir(exist_ok=True)
        return self

    def reference(self, path: Path) -> str:
        """Serialize an internal path without embedding its machine root."""

        resolved = Path(path).resolve()
        try:
            relative = resolved.relative_to(self.root)
            return "workspace://" + ("" if relative == Path(".") else relative.as_posix())
        except ValueError:
            pass
        try:
            relative = resolved.relative_to(self.repository)
            return "repo://" + ("" if relative == Path(".") else relative.as_posix())
        except ValueError as error:
            raise ValueError("path is outside the repository and workspace") from error

    def resolve_reference(self, value: str) -> Path:
        if value.startswith("workspace://"):
            return self._safe_join(self.root, value.removeprefix("workspace://"))
        if value.startswith("repo://"):
            return self._safe_join(self.repository, value.removeprefix("repo://"))
        raise ValueError("portable path must use repo:// or workspace://")

    @staticmethod
    def _safe_join(root: Path, value: str) -> Path:
        candidate = (root / value).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError("portable path escapes its root") from error
        return candidate
