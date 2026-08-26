"""Run the official training entrypoint with bounded GPU stage profiling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import subprocess
import sys

from gala_sim.identity import sha256_file

from .stage_profile import GpuStageProfileSession, parse_iteration_range, ranges_as_strings


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m gala_sim.adapters.stage_runner")
    parser.add_argument("--profile-output", type=Path, required=True)
    parser.add_argument(
        "--profile-iteration-range", action="append", required=True,
        type=parse_iteration_range, metavar="START:END",
    )
    parser.add_argument("train_script", type=Path)
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    return parser


def _repository_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    train_script = args.train_script.resolve()
    if not train_script.is_file():
        raise FileNotFoundError(train_script)
    ranges = tuple(args.profile_iteration_range)
    identity = {
        "repository_commit": _repository_commit(),
        "train_script": str(train_script),
        "train_script_sha256": sha256_file(train_script),
        "working_directory": str(Path.cwd().resolve()),
        "iteration_ranges": ranges_as_strings(ranges),
        "argv": [str(train_script), *args.train_args],
    }
    session = GpuStageProfileSession(
        output=args.profile_output,
        iteration_ranges=ranges,
        run_identity=identity,
    )
    session.install()
    original_argv = sys.argv
    completed = False
    try:
        sys.argv = [str(train_script), *args.train_args]
        runpy.run_path(str(train_script), run_name="__main__")
        completed = True
        result = session.finish()
    finally:
        session.restore()
        sys.argv = original_argv
    if not completed:
        return 2
    print(json.dumps({
        "output": str(args.profile_output.resolve()),
        "stage_count": len(result["stage_summaries"]),
        "status": result["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
