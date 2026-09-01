from __future__ import annotations

from pathlib import Path

from gala_sim.workspace import WorkspacePaths


def _repository(root: Path) -> Path:
    (root / "configs").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    return root


def test_workspace_discovers_repository_from_nested_path(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repo")
    nested = repository / "tools" / "nested"
    nested.mkdir(parents=True)
    paths = WorkspacePaths.discover(repository=nested)
    assert paths.repository == repository.resolve()
    assert paths.root == repository.resolve() / "workspace"


def test_workspace_precedence_and_directory_creation(
    monkeypatch, tmp_path: Path,
) -> None:
    repository = _repository(tmp_path / "repo")
    environment_root = tmp_path / "environment-workspace"
    explicit_root = tmp_path / "explicit-workspace"
    monkeypatch.setenv("GALA_WORKSPACE", str(environment_root))
    assert WorkspacePaths.discover(repository=repository).root == environment_root.resolve()
    paths = WorkspacePaths.discover(repository=repository, workspace=explicit_root)
    assert paths.root == explicit_root.resolve()
    paths.ensure()
    assert all(path.is_dir() for path in paths.managed_directories)


def test_workspace_serializes_internal_paths_portably(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repo")
    paths = WorkspacePaths.discover(repository=repository)
    assert paths.reference(repository / "configs" / "model.yaml") == "repo://configs/model.yaml"
    assert paths.reference(paths.datasets / "walnut" / "metadata.json") == (
        "workspace://datasets/walnut/metadata.json"
    )
