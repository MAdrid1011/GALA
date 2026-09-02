from __future__ import annotations

from pathlib import Path
import subprocess
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_build_metadata_uses_a_supported_license_file() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["license"] == "Apache-2.0"
    license_path = ROOT / "LICENSE"
    assert license_path.is_file()
    assert "Apache License" in license_path.read_text(encoding="utf-8")


def _public_files() -> list[Path]:
    repository = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"], cwd=ROOT,
        text=True, capture_output=True, check=False,
    )
    if repository.returncode == 0:
        output = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=ROOT,
        )
        return [ROOT / item.decode() for item in output.split(b"\0") if item]
    excluded = {".pytest_cache", "__pycache__", "build", "dist"}
    return [
        path for path in ROOT.rglob("*")
        if path.is_file()
        and not any(part in excluded or part.endswith(".egg-info")
                    for part in path.relative_to(ROOT).parts)
    ]


def test_public_tree_has_no_machine_paths_or_non_english_text() -> None:
    forbidden = (
        "/" + "home/", "/" + "tmp/", "GALA" + "-runtime",
        "IMPLEMENTATION" + "_STATUS", "11_" + "EXPERIMENT_RECORDS",
    )
    for path in _public_files():
        if not path.is_file() or path.name == "LICENSE":
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue
        text = data.decode("utf-8")
        assert not any("\u4e00" <= character <= "\u9fff" for character in text), path
        for token in forbidden:
            assert token not in text, (path, token)


def test_public_tree_has_no_generated_experiment_artifacts() -> None:
    forbidden_roots = {"workspace", "records", "profiles", "traces", "results", "runs"}
    generated_suffixes = {".arrow", ".parquet", ".npy", ".npz", ".pt", ".pth", ".ckpt", ".log"}
    for path in _public_files():
        relative = path.relative_to(ROOT)
        assert not (relative.parts and relative.parts[0] in forbidden_roots), relative
        assert relative.suffix not in generated_suffixes, relative
