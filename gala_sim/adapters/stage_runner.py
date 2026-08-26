"""Run the official training entrypoint with bounded GPU stage profiling."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

from gala_sim.identity import sha256_file, sha256_tree
from gala_sim.manifest import dataset_record, verify_freeze_record
from gala_sim.tools.gpu_profile_campaign import GpuProfileCampaign

from .stage_profile import (
    GpuStageProfileSession, IterationRange, parse_iteration_range, ranges_as_strings,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m gala_sim.adapters.stage_runner")
    parser.add_argument("--profile-output", type=Path, required=True)
    parser.add_argument(
        "--capture-profiler-api", action="store_true",
        help="control an external profiler at each selected iteration range",
    )
    parser.add_argument(
        "--profile-iteration-range", action="append", default=[],
        type=parse_iteration_range, metavar="START:END",
    )
    parser.add_argument("--profile-campaign", type=Path)
    parser.add_argument("--input-freeze", type=Path)
    parser.add_argument(
        "--profile-mode", choices=("full_timing", "representative"),
    )
    parser.add_argument("--profile-tool", choices=("cuda_event", "nsys", "ncu"))
    parser.add_argument("train_script", type=Path)
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    return parser


def _repository_commit() -> str:
    root = Path(__file__).resolve().parents[2]
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def _profile_selection(
    args: argparse.Namespace,
) -> tuple[tuple[IterationRange, ...], dict[str, object] | None]:
    if args.profile_campaign is None:
        if args.profile_mode is not None or args.profile_tool is not None:
            raise ValueError("--profile-mode/--profile-tool require --profile-campaign")
        if not args.profile_iteration_range:
            raise ValueError("GPU profiling requires iteration ranges or a campaign")
        return tuple(args.profile_iteration_range), None
    if args.profile_iteration_range:
        raise ValueError("campaign profiling cannot use manual iteration ranges")
    if args.profile_mode is None or args.profile_tool is None:
        raise ValueError("--profile-campaign requires --profile-mode and --profile-tool")
    campaign = GpuProfileCampaign.load(args.profile_campaign)
    if args.profile_mode == "full_timing":
        if args.profile_tool != "cuda_event" or args.capture_profiler_api:
            raise ValueError("full timing requires cuda_event without profiler capture")
        ranges = campaign.cuda_event_ranges
    else:
        expected_control = args.profile_tool in {"nsys", "ncu"}
        if args.profile_tool not in {"nsys", "ncu"}:
            raise ValueError("representative profiling requires nsys or ncu")
        if args.capture_profiler_api != expected_control:
            raise ValueError("NSYS and NCU require profiler API control")
        ranges = tuple(
            IterationRange(item.iteration, item.iteration)
            for item in campaign.representatives
        )
    identity = campaign.manifest()
    identity["profile_mode"] = args.profile_mode
    identity["profile_tool"] = args.profile_tool
    return ranges, identity


def _frozen_profile_identity(
    freeze_path: Path, train_script: Path, train_args: list[str],
) -> dict[str, object]:
    freeze_path = freeze_path.resolve()
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    verify_freeze_record(freeze)
    training = freeze.get("training")
    command = training.get("command") if isinstance(training, dict) else None
    if not isinstance(command, dict):
        raise ValueError("input freeze has no official command")
    frozen_argv = command.get("argv")
    working_directory = command.get("working_directory")
    if not isinstance(frozen_argv, list) or len(frozen_argv) != 6:
        raise ValueError("input freeze command is unsupported for profiling")
    if train_script != (Path(str(working_directory)) / str(frozen_argv[1])).resolve():
        raise ValueError("training script does not match input freeze")
    if Path.cwd().resolve() != Path(str(working_directory)).resolve():
        raise ValueError("working directory does not match input freeze")
    model = freeze.get("model")
    dataset = freeze.get("dataset")
    if not isinstance(model, dict) or not isinstance(dataset, dict):
        raise ValueError("input freeze model or dataset identity is invalid")
    actual_commit = subprocess.check_output(
        ["git", "-C", str(working_directory), "rev-parse", "HEAD"], text=True,
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(working_directory), "status", "--porcelain"], text=True,
    ).strip()
    if actual_commit != model.get("commit") or dirty:
        raise ValueError("training source does not match the frozen clean commit")
    if sha256_tree(Path(str(working_directory))) != model.get("tree_sha256"):
        raise ValueError("training source tree does not match input freeze")
    expected = [str(value) for value in frozen_argv[2:]]
    if len(train_args) != len(expected) or train_args[:3] != expected[:3]:
        raise ValueError("training arguments do not match input freeze")
    if expected[2] != "-m" or train_args[2] != "-m":
        raise ValueError("profile command may only replace the frozen model output")
    profile_model_output = Path(train_args[3]).resolve()
    if profile_model_output.exists():
        raise ValueError(f"profile model output already exists: {profile_model_output}")
    live_dataset = dataset_record(
        Path(train_args[1]), str(dataset["name"]), str(dataset["source_url"]),
        str(dataset["license_url"]),
    )
    if live_dataset.manifest_sha256 != dataset.get("manifest_sha256"):
        raise ValueError("training dataset does not match input freeze")
    return {
        "path": str(freeze_path),
        "sha256": sha256_file(freeze_path),
        "run_manifest_sha256": str(freeze["run_manifest_sha256"]),
        "model_commit": str(model["commit"]),
        "model_tree_sha256": str(model["tree_sha256"]),
        "dataset_manifest_sha256": str(dataset["manifest_sha256"]),
        "frozen_command_sha256": str(command["sha256"]),
        "profile_model_output": str(profile_model_output),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    train_script = args.train_script.resolve()
    if not train_script.is_file():
        raise FileNotFoundError(train_script)
    ranges, campaign_identity = _profile_selection(args)
    if campaign_identity is not None and args.input_freeze is None:
        raise ValueError("campaign profiling requires --input-freeze")
    if campaign_identity is None and args.input_freeze is not None:
        raise ValueError("--input-freeze requires --profile-campaign")
    freeze_identity = (
        _frozen_profile_identity(args.input_freeze, train_script, args.train_args)
        if args.input_freeze is not None else None
    )
    identity = {
        "process_id": os.getpid(),
        "repository_commit": _repository_commit(),
        "train_script": str(train_script),
        "train_script_sha256": sha256_file(train_script),
        "working_directory": str(Path.cwd().resolve()),
        "iteration_ranges": ranges_as_strings(ranges),
        "cuda_profiler_api_control": bool(args.capture_profiler_api),
        "argv": [str(train_script), *args.train_args],
        "profiling_campaign": campaign_identity,
        "input_freeze": freeze_identity,
    }
    session = GpuStageProfileSession(
        output=args.profile_output,
        iteration_ranges=ranges,
        run_identity=identity,
        control_cuda_profiler=args.capture_profiler_api,
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
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
