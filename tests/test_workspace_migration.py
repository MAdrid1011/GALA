from __future__ import annotations

from pathlib import Path

from tools.migrate_workspace import ProcessUse, find_process_uses, migrate_workspace


def _repository(root: Path) -> Path:
    (root / "configs").mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    return root


def test_migration_stops_when_a_process_uses_the_source(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repo")
    source = tmp_path / "legacy"
    source.mkdir()
    blocker = ProcessUse(42, "open_file", str(source / "active.csv"))
    report = migrate_workspace(
        source, repository, process_finder=lambda root: (blocker,),
    )
    assert not report.moved
    assert report.blockers == (blocker,)
    assert source.is_dir()
    assert not (repository / "workspace").exists()


def test_migration_renames_and_normalizes_legacy_content(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repo")
    source = tmp_path / "legacy"
    extracted = source / "data/chest/extracted/cone_ntrain_50_angle_360"
    extracted.mkdir(parents=True)
    (extracted / "meta_data.json").write_text("{}", encoding="utf-8")
    archive = source / "data/chest/cone_ntrain_50_angle_360.zip"
    archive.write_bytes(b"zip")
    (source / "upstream/r2_gaussian").mkdir(parents=True)
    (source / "experiments").mkdir()
    (source / "experiments/result.csv").write_text("cycles\n1\n", encoding="utf-8")

    report = migrate_workspace(
        source, repository, process_finder=lambda root: (),
    )
    workspace = repository / "workspace"
    assert report.moved
    assert not source.exists()
    assert (workspace / "datasets/chest/meta_data.json").is_file()
    assert (workspace / "cache/downloads/chest/cone_ntrain_50_angle_360.zip").is_file()
    assert (workspace / "upstream/r2_gaussian").is_dir()
    assert (workspace / "archive/legacy/experiments/result.csv").is_file()


def test_migration_dry_run_changes_nothing(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "repo")
    source = tmp_path / "legacy"
    source.mkdir()
    report = migrate_workspace(
        source, repository, dry_run=True, process_finder=lambda root: (),
    )
    assert not report.moved
    assert source.is_dir()
    assert not (repository / "workspace").exists()


def test_open_file_scanner_uses_machine_readable_lsof_fields(
    monkeypatch, tmp_path: Path,
) -> None:
    source = tmp_path / "legacy"
    source.mkdir()
    active = source / "active.csv"
    active.write_text("sample", encoding="utf-8")

    class Completed:
        returncode = 0
        stdout = f"p42\nn{active}\n"
        stderr = ""

    observed = {}

    def run(command, **kwargs):
        observed["command"] = command
        return Completed()

    monkeypatch.setattr("tools.migrate_workspace.subprocess.run", run)
    uses = find_process_uses(source)
    assert ProcessUse(42, "open_file", str(active)) in uses
    assert observed["command"][:2] == ["lsof", "-Fpn"]
