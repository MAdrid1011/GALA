"""Run auditable NCU preflight and formal invocation jobs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import shutil
import statistics
import subprocess
import time
from typing import Any, Mapping

from gala_sim.config import load_config
from gala_sim.identity import canonical_json, sha256_bytes, sha256_file
from gala_sim.manifest import verify_freeze_record
from gala_sim.tools.gpu_profile_artifacts import (
    bind_ncu_profile_to_plan, classify_sass_csv, parse_ncu_csv,
)
from gala_sim.tools.gpu_ncu_plan import _implementation_hashes, _kernel_id_filter
from gala_sim.tools.preflight import sample_gpustat


RUN_SCHEMA_VERSION = "gala-ncu-capture-run-v3"
PLAN_SCHEMA_VERSION = "gala-ncu-launch-signature-plan-v5"


def _read_json(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"JSON document is not an object: {path}")
    return document


def _load_plan(path: Path) -> dict[str, Any]:
    plan = _read_json(path)
    digest = plan.get("content_sha256")
    payload = {key: value for key, value in plan.items() if key != "content_sha256"}
    stability = plan.get("stability_validation")
    if (
        plan.get("schema_version") != PLAN_SCHEMA_VERSION
        or not isinstance(digest, str)
        or sha256_bytes(canonical_json(payload)) != digest
        or not isinstance(stability, Mapping)
        or stability.get("status") != "passed"
    ):
        raise ValueError("NCU plan identity is invalid")
    repository = Path(__file__).resolve().parents[2]
    repository_status = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain"], text=True,
    ).strip()
    if (
        plan.get("implementation_sha256") != _implementation_hashes(repository)
        or repository_status
    ):
        raise ValueError("NCU plan requires its clean profiling implementation")
    return plan


def _capture_job(plan: Mapping[str, Any], index: int) -> Mapping[str, Any]:
    matches = [
        item for item in plan.get("capture_groups", ())
        if isinstance(item, Mapping) and int(item.get("job_index", -1)) == index
    ]
    if len(matches) != 1:
        raise ValueError(f"NCU capture job is invalid: {index}")
    arguments = matches[0].get("ncu_arguments")
    if (
        not isinstance(arguments, list)
        or not all(isinstance(value, str) and value for value in arguments)
        or "--launch-count" in arguments
        or "--kill" in arguments
        or arguments[:2] != ["--profile-from-start", "off"]
    ):
        raise ValueError("NCU capture job arguments violate the formal contract")
    return matches[0]


def _freeze_command(
    freeze_path: Path, model_output: Path, iterations: int | None,
) -> tuple[list[str], Path]:
    freeze = _read_json(freeze_path)
    verify_freeze_record(freeze)
    command = freeze.get("training", {}).get("command")
    if not isinstance(command, Mapping) or not isinstance(command.get("argv"), list):
        raise ValueError("input freeze has no official training command")
    argv = [str(value) for value in command["argv"]]
    if len(argv) != 6 or argv[4] != "-m":
        raise ValueError("input freeze training command is unsupported")
    argv[5] = str(model_output)
    if iterations is not None:
        argv.extend([
            "--iterations", str(iterations),
            "--test_iterations", str(iterations),
            "--save_iterations", str(iterations),
        ])
    working_directory = Path(str(command["working_directory"])).resolve()
    if not working_directory.is_dir():
        raise ValueError("frozen working directory is unavailable")
    return argv, working_directory


def _runner_command(
    repository: Path, plan: Mapping[str, Any], job: Mapping[str, Any],
    freeze_path: Path, output: Path, *, preflight_iterations: int | None,
) -> tuple[list[str], Path, dict[str, str]]:
    ncu = shutil.which("ncu")
    if ncu is None:
        raise FileNotFoundError("ncu")
    official, working_directory = _freeze_command(
        freeze_path, output / "model", preflight_iterations,
    )
    python_executable = official[0]
    train_script = str((working_directory / official[1]).resolve())
    train_args = official[2:]
    stage_arguments = [
        python_executable, "-m", "gala_sim.adapters.stage_runner",
        "--profile-output", str(output / "stages.json"),
        "--capture-profiler-api",
    ]
    if preflight_iterations is None:
        stage_arguments.extend([
            "--profile-campaign", str(plan["campaign"]["path"]),
            "--input-freeze", str(freeze_path.resolve()),
            "--profile-mode", "representative", "--profile-tool", "ncu",
        ])
    else:
        stage_arguments.extend(["--profile-iteration-range", "1:1"])
    stage_arguments.extend([train_script, *train_args])
    report_base = output / "profile"
    if preflight_iterations is None:
        ncu_arguments = [str(value) for value in job["ncu_arguments"]]
    else:
        ncu_arguments = [str(value) for value in job["ncu_arguments"]]
        if job.get("capture_mode") != "kernel_invocations":
            raise ValueError("NCU preflight requires an invocation capture job")
        by_name: dict[str, list[int]] = {}
        for launch in job.get("expected_launches", ()):
            if not isinstance(launch, Mapping) or int(launch.get("iteration", -1)) != 1:
                continue
            by_name.setdefault(str(launch["kernel_name"]), []).append(
                int(launch["kernel_name_ordinal_in_invocation_capture"])
            )
        requested = int(plan.get("ncu", {}).get("preflight_profile_launch_count", 0))
        eligible = sorted(
            ((len(ordinals), name, sorted(set(ordinals))) for name, ordinals in by_name.items()),
            reverse=True,
        )
        if requested <= 0 or not eligible or eligible[0][0] < requested:
            raise ValueError("NCU job cannot provide the frozen preflight launch count")
        _, name, ordinals = eligible[0]
        index = ncu_arguments.index("--kernel-id") + 1
        ncu_arguments[index] = _kernel_id_filter([name], ordinals[:requested])
    command = [
        ncu, "--force-overwrite", "--export", str(report_base),
        *ncu_arguments, *stage_arguments,
    ]
    environment = dict(os.environ)
    previous = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(repository) if not previous else str(repository) + os.pathsep + previous
    )
    return command, working_directory, environment


def _export_report(report: Path, page: str, output: Path) -> None:
    ncu = shutil.which("ncu")
    if ncu is None:
        raise FileNotFoundError("ncu")
    command = [ncu, "--import", str(report), "--csv", "--page", page]
    if page == "source":
        command.extend(["--section", "SourceCounters"])
    with output.open("w", encoding="utf-8") as stream:
        subprocess.run(command, check=True, stdout=stream, text=True)


def _csv_launch_count(path: Path) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = next((index for index, line in enumerate(lines) if line.startswith('"ID"')), None)
    if header is None:
        raise ValueError("NCU details CSV has no header")
    return len({row["ID"] for row in csv.DictReader(lines[header:])})


def _write_status(path: Path, document: Mapping[str, Any]) -> None:
    payload = dict(document)
    payload["record_sha256"] = sha256_bytes(canonical_json(payload))
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_progress(path: Path) -> tuple[int, int]:
    try:
        status = path.stat()
    except FileNotFoundError:
        return 0, 0
    return status.st_size, status.st_mtime_ns


def _terminate_process_group(process: subprocess.Popen[str], grace_seconds: float) -> str:
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=grace_seconds)
        return "SIGTERM"
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        return "SIGKILL"


def _sample_has_job_gpu_activity(
    sample: Mapping[str, Any], process_group_id: int,
) -> bool:
    if float(sample.get("utilization_percent", 0.0)) <= 0.0:
        return False
    for process in sample.get("compute_processes", ()):
        if not isinstance(process, Mapping):
            continue
        try:
            if os.getpgid(int(process["pid"])) == process_group_id:
                return True
        except (KeyError, OSError, TypeError, ValueError):
            continue
    return False


def _run_and_sample(
    command: list[str], working_directory: Path, environment: Mapping[str, str],
    log_path: Path, report_path: Path, interval_seconds: float,
    inactivity_seconds: float, termination_grace_seconds: float,
) -> tuple[int, float, list[dict[str, Any]], dict[str, Any]]:
    start = time.time()
    monotonic_start = time.monotonic()
    last_progress = monotonic_start
    maximum_inactivity = 0.0
    samples = []
    previous_files = (_file_progress(log_path), _file_progress(report_path))
    termination_signal = None
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=working_directory, env=dict(environment), stdout=log,
            stderr=subprocess.STDOUT, text=True, start_new_session=True,
        )
        while process.poll() is None:
            sample: dict[str, Any]
            try:
                sample = asdict(sample_gpustat())
            except RuntimeError as error:
                sample = {"timestamp": time.time(), "error": str(error)}
            samples.append(sample)
            current_files = (_file_progress(log_path), _file_progress(report_path))
            gpu_active = _sample_has_job_gpu_activity(sample, process.pid)
            if gpu_active or current_files != previous_files:
                last_progress = time.monotonic()
            previous_files = current_files
            inactivity = time.monotonic() - last_progress
            maximum_inactivity = max(maximum_inactivity, inactivity)
            if inactivity >= inactivity_seconds:
                termination_signal = _terminate_process_group(
                    process, termination_grace_seconds,
                )
                break
            try:
                process.wait(timeout=interval_seconds)
            except subprocess.TimeoutExpired:
                pass
        returncode = int(process.wait())
    watchdog = {
        "status": "terminated" if termination_signal else "passed",
        "inactivity_limit_seconds": inactivity_seconds,
        "termination_grace_seconds": termination_grace_seconds,
        "maximum_observed_inactivity_seconds": maximum_inactivity,
        "termination_signal": termination_signal,
        "final_log_size_bytes": _file_progress(log_path)[0],
        "final_report_size_bytes": _file_progress(report_path)[0],
    }
    return returncode, time.time() - start, samples, watchdog


def _sample_summary(samples: list[Mapping[str, Any]]) -> dict[str, Any]:
    utilization = [
        float(item["utilization_percent"]) for item in samples
        if "utilization_percent" in item
    ]
    memory = [
        int(item["memory_used_bytes"]) for item in samples if "memory_used_bytes" in item
    ]
    if not utilization:
        return {"status": "failed_preflight", "sample_count": 0}
    return {
        "status": "passed", "sample_count": len(utilization),
        "mean_utilization_percent": statistics.fmean(utilization),
        "median_utilization_percent": statistics.median(utilization),
        "maximum_memory_used_bytes": max(memory),
    }


def _run(args: argparse.Namespace) -> int:
    repository = Path(__file__).resolve().parents[2]
    plan_path = args.plan.resolve()
    freeze_path = args.input_freeze.resolve()
    plan = _load_plan(plan_path)
    job = _capture_job(plan, args.capture_job_index)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    config = load_config(args.architecture)
    interval = float(config.value("preflight.gpustat_interval_seconds"))
    threshold = float(config.value("preflight.long_run_threshold_seconds"))
    floor = float(config.value("preflight.gpu_utilization_floor_percent"))
    inactivity = float(plan.get("ncu", {}).get("watchdog_inactivity_seconds", 0.0))
    termination_grace = float(
        plan.get("ncu", {}).get("watchdog_termination_grace_seconds", 0.0)
    )
    if inactivity <= 0 or termination_grace <= 0:
        raise ValueError("NCU watchdog timing must be positive")
    preflight_iterations = None
    if args.mode == "preflight":
        preflight_iterations = int(config.value("preflight.warmup_iterations")) + int(
            config.value("preflight.measure_iterations")
        )
    command, working_directory, environment = _runner_command(
        repository, plan, job, freeze_path, output,
        preflight_iterations=preflight_iterations,
    )
    start_iso = datetime.now(timezone.utc).isoformat()
    running = {
        "schema_version": RUN_SCHEMA_VERSION, "status": "running",
        "mode": args.mode, "started_at": start_iso,
        "plan": {
            "path": str(plan_path), "sha256": sha256_file(plan_path),
            "content_sha256": plan["content_sha256"],
        },
        "capture_job_index": args.capture_job_index,
        "selected_kernel_launch_count": job["selected_kernel_launch_count"],
        "input_freeze": {"path": str(freeze_path), "sha256": sha256_file(freeze_path)},
        "repository_commit": subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True,
        ).strip(),
        "command": command, "working_directory": str(working_directory),
        "preflight_iterations": preflight_iterations,
    }
    _write_status(output / "status.json", running)
    returncode, duration, samples, watchdog = _run_and_sample(
        command, working_directory, environment, output / "stdout.log",
        output / "profile.ncu-rep", interval, inactivity, termination_grace,
    )
    report = output / "profile.ncu-rep"
    details = output / "profile.details.csv"
    source = output / "profile.source.csv"
    artifacts: dict[str, Any] = {}
    prediction = None
    failure = None
    try:
        if watchdog["status"] == "terminated":
            failure = "watchdog_inactivity_timeout"
        elif returncode != 0 or not report.is_file():
            failure = "ncu_command_failed"
        else:
            _export_report(report, "details", details)
            observed_launches = _csv_launch_count(details)
            artifacts = {
                "report_sha256": sha256_file(report),
                "details_sha256": sha256_file(details),
                "observed_kernel_launch_count": observed_launches,
            }
            if args.mode == "preflight":
                native = _read_json(args.native_preflight.resolve())
                native_wall = float(native["calibration"]["wall_seconds"])
                native_prediction = float(native["prediction"]["predicted_seconds"])
                if native.get("status") != "passed" or observed_launches <= 0:
                    raise ValueError("native preflight or NCU launch count is invalid")
                overhead = max(0.0, duration - native_wall)
                prediction = {
                    "native_predicted_seconds": native_prediction,
                    "native_calibration_wall_seconds": native_wall,
                    "ncu_calibration_wall_seconds": duration,
                    "profiled_launch_count": observed_launches,
                    "seconds_per_profiled_launch": overhead / observed_launches,
                    "formal_selected_launch_count": job["selected_kernel_launch_count"],
                    "predicted_formal_job_seconds": (
                        native_prediction
                        + overhead * int(job["selected_kernel_launch_count"])
                        / observed_launches
                    ),
                    "uncertainty": "linear_kernel_replay_overhead_from_one_real_launch",
                }
            if args.mode == "run":
                _export_report(report, "source", source)
                raw = parse_ncu_csv(details)
                stages = _read_json(output / "stages.json")
                stages["source"] = str((output / "stages.json").resolve())
                stages["source_sha256"] = sha256_file(output / "stages.json")
                bound = bind_ncu_profile_to_plan(
                    raw, plan, args.capture_job_index, stages,
                )
                bound_path = output / "profile.bound.json"
                bound_path.write_text(
                    json.dumps(bound, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                classified = classify_sass_csv(source, bound)
                sass_path = output / "profile.sass.json"
                sass_path.write_text(
                    json.dumps(classified, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                artifacts.update({
                    "source_sha256": sha256_file(source),
                    "bound_sha256": sha256_file(bound_path),
                    "sass_sha256": sha256_file(sass_path),
                    "bound_status": bound["status"],
                    "sass_status": classified["status"],
                })
                if bound["status"] != "passed" or classified["status"] != "passed":
                    failure = "ncu_evidence_binding_failed"
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        failure = f"artifact_processing_failed:{error}"
    summary = _sample_summary(samples)
    if args.mode == "preflight":
        predicted_job = float((prediction or {}).get("predicted_formal_job_seconds", 0.0))
        if summary.get("status") != "passed":
            failure = failure or "gpu_sampling_unavailable"
        elif predicted_job >= threshold and float(
            summary.get("mean_utilization_percent", 0.0)
        ) < floor:
            failure = failure or "gpu_utilization_floor_failed"
    final = {
        **running,
        "status": "passed" if failure is None else "failed_preflight",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": duration, "returncode": returncode,
        "long_run_threshold_seconds": threshold,
        "gpu_utilization_floor_percent": floor,
        "watchdog": watchdog,
        "gpu_sampling": summary, "gpu_samples": samples,
        "prediction": prediction, "artifacts": artifacts, "failure": failure,
    }
    _write_status(output / "status.json", final)
    print(json.dumps({
        "output": str(output), "status": final["status"],
        "duration_seconds": duration, "failure": failure,
    }, sort_keys=True))
    return 0 if final["status"] == "passed" else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m gala_sim.tools.gpu_ncu_runner")
    parser.add_argument("mode", choices=("preflight", "run"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--capture-job-index", type=int, required=True)
    parser.add_argument("--input-freeze", type=Path, required=True)
    parser.add_argument("--architecture", type=Path, required=True)
    parser.add_argument("--native-preflight", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "preflight" and args.native_preflight is None:
        parser.error("preflight requires --native-preflight")
    if args.mode == "run" and args.native_preflight is not None:
        parser.error("run does not accept --native-preflight")
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
