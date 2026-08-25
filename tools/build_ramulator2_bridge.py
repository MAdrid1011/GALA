#!/usr/bin/env python3
"""Build the GALA C ABI bridge against a verified Ramulator 2 checkout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from gala_sim.identity import sha256_file


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ramulator-source", required=True, type=Path)
    parser.add_argument("--ramulator-library", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--compiler", default="c++")
    args = parser.parse_args(argv)
    repository = REPOSITORY
    source = args.ramulator_source.resolve()
    library = args.ramulator_library.resolve()
    output = args.output.resolve()
    bridge_source = repository / "native" / "ramulator2_bridge.cpp"
    try:
        if not (source / "src" / "ramulator").is_dir():
            raise RuntimeError("Ramulator source root lacks src/ramulator")
        if not library.is_file():
            raise RuntimeError("Ramulator shared library does not exist")
        commit = _git(source, "rev-parse", "HEAD")
        if _git(source, "status", "--porcelain"):
            raise RuntimeError("Ramulator source checkout has uncommitted changes")
        version = _git(source, "describe", "--tags", "--always")
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            args.compiler, "-std=c++20", "-O2", "-fPIC", "-shared",
            f"-DGALA_RAMULATOR_VERSION=\"{version}\"",
            "-I", str(source / "src"), str(bridge_source), str(library),
            f"-Wl,-rpath,{library.parent}", "-o", str(output),
        ]
        subprocess.run(command, check=True)
        compiler_version = subprocess.check_output(
            [args.compiler, "--version"], text=True,
        ).splitlines()[0]
        manifest = {
            "schema_version": "gala-ramulator2-bridge-build-v1",
            "ramulator_commit": commit,
            "ramulator_version": version,
            "ramulator_library": str(library),
            "ramulator_library_sha256": sha256_file(library),
            "bridge_source": str(bridge_source),
            "bridge_source_sha256": sha256_file(bridge_source),
            "bridge_library": str(output),
            "bridge_library_sha256": sha256_file(output),
            "compiler": compiler_version,
            "command": command,
        }
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Ramulator 2 bridge build failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
