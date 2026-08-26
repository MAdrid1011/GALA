from __future__ import annotations

import csv
import argparse
import json
from pathlib import Path
import sqlite3

import pytest

from gala_sim.adapters.stage_profile import (
    GpuStageProfileSession,
    IterationRange,
    parse_iteration_range,
)
from gala_sim.tools.gpu_calibration import CalibrationConfig
from gala_sim.tools.gpu_profile_artifacts import parse_ncu_csv, parse_nsys_sqlite
from gala_sim.tools.gpu_profile_artifacts import classify_sass_csv
from gala_sim.tools.gpu_profile_campaign import GpuProfileCampaign
from gala_sim.tools.gpu_ncu_plan import (
    NcuPlanConfig, build_ncu_plan, main as ncu_plan_main,
    validate_ncu_measurement,
)
from gala_sim.adapters.stage_runner import _frozen_profile_identity, _profile_selection
from gala_sim.tools.gpu_normalization import normalize_stage_profiles


class _FakeCudaRuntime:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def cudaProfilerStart(self) -> int:
        self.calls.append("start")
        return 0

    def cudaProfilerStop(self) -> int:
        self.calls.append("stop")
        return 0


class _FakeCuda:
    def __init__(self, runtime: _FakeCudaRuntime) -> None:
        self.runtime = runtime

    def cudart(self) -> _FakeCudaRuntime:
        return self.runtime


class _FakeTorch:
    def __init__(self, runtime: _FakeCudaRuntime) -> None:
        self.cuda = _FakeCuda(runtime)


def test_gpu_profile_iteration_ranges_and_summary(tmp_path: Path) -> None:
    assert parse_iteration_range("600:601") == IterationRange(600, 601)
    with pytest.raises(ValueError, match="overlap"):
        GpuStageProfileSession(
            tmp_path / "profile.json",
            (IterationRange(1, 2), IterationRange(2, 3)),
            {},
        )
    session = GpuStageProfileSession(
        tmp_path / "profile.json", (IterationRange(600, 601),), {"commit": "abc"}
    )
    session._records.extend([
        {"iteration": 600, "stage": "backward", "call_index": 1, "milliseconds": 2.0},
        {"iteration": 601, "stage": "backward", "call_index": 2, "milliseconds": 4.0},
    ])
    result = session._result()
    assert result["stage_summaries"]["backward"] == {
        "call_count": 2,
        "total_ms": 6.0,
        "median_ms": 3.0,
        "minimum_ms": 2.0,
        "maximum_ms": 4.0,
    }
    assert result["formal_performance_eligible"] is False
    assert result["requested_iteration_coverage"]["status"] == "incomplete"


def test_gpu_profile_controls_cuda_profiler_ranges(tmp_path: Path) -> None:
    runtime = _FakeCudaRuntime()
    session = GpuStageProfileSession(
        tmp_path / "profile.json", (IterationRange(2, 2),), {},
        control_cuda_profiler=True,
    )
    session._torch = _FakeTorch(runtime)
    session._current_iteration = 2
    session._start_cuda_profiler()
    assert session._cuda_profiler_active is True
    session._stop_cuda_profiler()
    assert runtime.calls == ["start", "stop"]
    assert session._cuda_profiler_active is False


def test_gpu_calibration_config_is_strict() -> None:
    config = CalibrationConfig(
        dtype="float32",
        batch_sizes=(32, 64),
        input_min=0.5,
        input_max=1.5,
        warmup_repetitions=1,
        measure_repetitions=2,
        launches_per_measurement=4,
        atomic_contention_divisor=2,
    )
    assert config.batch_sizes == (32, 64)
    with pytest.raises(ValueError, match="invalid"):
        CalibrationConfig(
            dtype="float32",
            batch_sizes=(64, 32),
            input_min=0.5,
            input_max=1.5,
            warmup_repetitions=1,
            measure_repetitions=2,
            launches_per_measurement=4,
            atomic_contention_divisor=2,
        )


def test_formal_gpu_profile_campaign_covers_frozen_schedule() -> None:
    root = Path(__file__).resolve().parents[1]
    campaign = GpuProfileCampaign.load(
        root / "configs/profiling/r2_gaussian_chest_campaign.yaml"
    )
    manifest = campaign.manifest()
    assert manifest["cuda_event_iteration_ranges"] == ["1:30000"]
    assert manifest["nsys_capture_range_end"] == "repeat-shutdown:9"
    assert {
        item["iteration"] for item in manifest["representative_iterations"]
        if "periodic_evaluation" in item["roles"]
    } == {1, 5000, 10000, 20000, 30000}
    assert manifest["formal_performance_eligible"] is False


def test_stage_runner_campaign_selection_is_frozen() -> None:
    root = Path(__file__).resolve().parents[1]
    campaign = root / "configs/profiling/r2_gaussian_chest_campaign.yaml"
    full_args = argparse.Namespace(
        profile_campaign=campaign, profile_mode="full_timing",
        profile_tool="cuda_event", profile_iteration_range=[], capture_profiler_api=False,
    )
    full_ranges, full_identity = _profile_selection(full_args)
    assert full_ranges == (IterationRange(1, 30000),)
    assert full_identity is not None
    assert full_identity["profile_mode"] == "full_timing"
    representative_args = argparse.Namespace(
        profile_campaign=campaign, profile_mode="representative",
        profile_tool="nsys", profile_iteration_range=[], capture_profiler_api=True,
    )
    representative_ranges, _ = _profile_selection(representative_args)
    assert len(representative_ranges) == 9
    assert representative_ranges[0] == IterationRange(1, 1)
    with pytest.raises(ValueError, match="manual iteration"):
        _profile_selection(argparse.Namespace(
            profile_campaign=campaign, profile_mode="representative",
            profile_tool="nsys", profile_iteration_range=[IterationRange(2, 2)],
            capture_profiler_api=True,
        ))
    with pytest.raises(ValueError, match="require profiler API"):
        _profile_selection(argparse.Namespace(
            profile_campaign=campaign, profile_mode="representative",
            profile_tool="ncu", profile_iteration_range=[], capture_profiler_api=False,
        ))
    ncu_ranges, _ = _profile_selection(argparse.Namespace(
        profile_campaign=campaign, profile_mode="representative",
        profile_tool="ncu", profile_iteration_range=[], capture_profiler_api=True,
    ))
    assert ncu_ranges == representative_ranges


def test_gpu_profile_campaign_evidence_requires_every_role() -> None:
    root = Path(__file__).resolve().parents[1]
    campaign = GpuProfileCampaign.load(
        root / "configs/profiling/r2_gaussian_chest_campaign.yaml"
    )
    stages = set(campaign.required_stage_roles)
    stage_profile = {
        "schema_version": "fixture", "status": "passed",
        "iteration_ranges": [{"start": 1, "end": 30000}],
        "requested_iteration_coverage": {"status": "complete"},
        "run_identity": {"profiling_campaign": {
            "campaign_sha256": campaign.manifest()["campaign_sha256"],
            "profile_mode": "full_timing",
        }},
    }
    profiles = []
    manifest = campaign.manifest()
    for item in campaign.representatives:
        profiles.append({
            "status": "passed", "source_sha256": str(item.iteration) * 64,
            "run_identity": {
                "profiling_campaign_sha256": manifest["campaign_sha256"],
            },
            "kernel_coverage": {"status": "complete", "records": [{
                "iteration": item.iteration, "kernel_count": 2,
                "assigned_kernel_count": 2, "unassigned_kernel_count": 0,
                "multiply_assigned_kernel_count": 0,
            }]},
            "stage_calls": [
                {"iteration": item.iteration, "stage": stage} for stage in stages
            ],
        })
    result = campaign.validate_evidence(stage_profile, profiles)
    assert result["status"] == "passed"
    assert result["kernel_coverage"]["kernel_count"] == 18
    incomplete = campaign.validate_evidence(stage_profile, profiles[:-1])
    assert incomplete["status"] == "provisional_profiling_evidence"
    assert "missing_nsys_iteration:30000" in incomplete["reasons"]


def _nsys_plan_profiles(tmp_path: Path, campaign: GpuProfileCampaign) -> list[Path]:
    paths = []
    campaign_sha256 = campaign.manifest()["campaign_sha256"]
    stages = tuple(campaign.required_stage_roles)
    for item in campaign.representatives:
        calls = []
        kernel_count = 0
        call_index = 1
        for stage in stages:
            repeat = 5 if item.iteration == 1 and stage == "projection_forward" else 1
            kernels = [{
                "name": f"{stage}_kernel", "grid": [4, 1, 1],
                "block": [32, 1, 1], "duration_ms": 0.1,
            } for _ in range(repeat)]
            calls.append({
                "stage": stage, "iteration": item.iteration,
                "call_index": call_index, "kernel_launch_count": len(kernels),
                "kernels": kernels,
            })
            kernel_count += len(kernels)
            call_index += 1
        profile = {
            "schema_version": "fixture", "status": "passed",
            "source": str(tmp_path / f"profile.{item.iteration}.sqlite"),
            "source_sha256": str(item.iteration) * 64,
            "run_identity": {
                "status": "passed",
                "profiling_campaign_sha256": campaign_sha256,
            },
            "kernel_coverage": {"status": "complete", "records": [{
                "iteration": item.iteration, "kernel_count": kernel_count,
                "assigned_kernel_count": kernel_count, "unassigned_kernel_count": 0,
                "multiply_assigned_kernel_count": 0,
            }]},
            "stage_calls": calls,
        }
        path = tmp_path / f"inventory.{item.iteration}.json"
        path.write_text(json.dumps(profile), encoding="utf-8")
        paths.append(path)
    return paths


def test_ncu_plan_preserves_multiplicity_and_selects_validation_occurrences(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    config = NcuPlanConfig.load(
        root / "configs/profiling/r2_gaussian_chest_ncu.yaml"
    )
    campaign = GpuProfileCampaign.load(config.campaign)
    paths = _nsys_plan_profiles(tmp_path, campaign)
    result = build_ncu_plan(config, paths)
    signature = next(
        item for item in result["signatures"]
        if item["stage"] == "projection_forward"
    )
    initial = next(
        item for item in signature["representative_iterations"]
        if item["iteration"] == 1
    )
    assert result["status"] == "planned"
    assert result["formal_performance_eligible"] is False
    assert initial["multiplicity"] == 5
    assert [
        item["signature_occurrence_ordinal"] for item in initial["required_samples"]
    ] == [1, 3, 5]
    assert result["coverage"]["observed_kernel_launch_count"] == 121
    assert result["coverage"]["representative_iteration_signature_count"] == 117
    assert all(
        include["filter"].endswith("/")
        for group in result["capture_groups"] for include in group["nvtx_includes"]
    )
    assert all(
        group["ncu_arguments"][-2:] == ["--check-exit-code", "1"]
        and "--launch-count" not in group["ncu_arguments"]
        and "--kill" not in group["ncu_arguments"]
        for group in result["capture_groups"]
    )
    assert all(
        "--section" in group["ncu_arguments"]
        and group["ncu_arguments"][group["ncu_arguments"].index("--section") + 1]
        == "SourceCounters"
        for group in result["capture_groups"]
    )


def test_ncu_plan_rejects_inexact_nsys_coverage(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = NcuPlanConfig.load(
        root / "configs/profiling/r2_gaussian_chest_ncu.yaml"
    )
    campaign = GpuProfileCampaign.load(config.campaign)
    paths = _nsys_plan_profiles(tmp_path, campaign)
    profile = json.loads(paths[0].read_text(encoding="utf-8"))
    profile["kernel_coverage"]["records"][0]["assigned_kernel_count"] -= 1
    paths[0].write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly once"):
        build_ncu_plan(config, paths)


def test_ncu_measurement_validator_requires_exact_call_sequences(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = NcuPlanConfig.load(root / "configs/profiling/r2_gaussian_chest_ncu.yaml")
    campaign = GpuProfileCampaign.load(config.campaign)
    plan = build_ncu_plan(config, _nsys_plan_profiles(tmp_path, campaign))
    metrics = {
        "dram_read_bytes": 64.0, "dram_write_bytes": 32.0,
        "fp32_ffma": 10.0, "fp32_fadd": 4.0, "fp32_fmul": 2.0,
        "xu_instructions": 8.0, "atomic_requests": 3.0,
    }
    launches = []
    launch_id = 0
    for group in plan["capture_groups"]:
        iteration = int(group["iteration"])
        for call in group["nvtx_includes"]:
            for ordinal, signature in enumerate(call["kernel_sequence"], start=1):
                launch_id += 1
                launches.append({
                    "stage": signature["stage"], "iteration": iteration,
                    "call_index": int(call["call_index"]), "launch_id": str(launch_id),
                    "kernel_name": signature["kernel_name"],
                    "grid_size": signature["grid"], "block_size": signature["block"],
                    "kernel_ordinal_in_call": ordinal,
                    "metrics": metrics,
                })
    profile = {
        "status": "passed",
        "run_identity": {
            "status": "passed", "profiling_campaign_sha256": plan["campaign"]["sha256"],
        },
        "incomplete_launches": [], "launches": launches,
    }
    result = validate_ncu_measurement(plan, [profile])
    assert result["status"] == "passed"
    assert result["exact_call_coverage"] is True
    assert result["counter_reuse_gate"]["status"] == "passed"
    assert sum(
        item["multiplicity"] for item in result["aggregated_signatures"]
    ) == plan["coverage"]["observed_kernel_launch_count"]
    assert sum(
        item["kernel_launch_count"] for item in result["stage_summaries"].values()
    ) == plan["coverage"]["observed_kernel_launch_count"]
    assert all(
        item["weight_eligible"] is False
        for item in result["stage_summaries"].values()
    )
    tampered_plan = {**plan, "content_sha256": "0" * 64}
    invalid_plan = validate_ncu_measurement(tampered_plan, [profile])
    assert "ncu_plan_identity_invalid" in invalid_plan["reasons"]
    repeated = next(
        (index for index in range(1, len(launches))
         if launches[index]["iteration"] == launches[index - 1]["iteration"]
         and launches[index]["stage"] == launches[index - 1]["stage"]
         and launches[index]["kernel_name"] == launches[index - 1]["kernel_name"]),
        None,
    )
    assert repeated is not None
    launches[repeated]["metrics"] = {**metrics, "dram_read_bytes": 65.0}
    measured = validate_ncu_measurement(plan, [profile])
    assert measured["status"] == "passed"
    assert not any(
        reason.startswith("counter_reuse_disagreement:")
        for reason in measured["reasons"]
    )
    assert any(
        item["aggregation_mode"] == "exact_observed_launches"
        for item in measured["aggregated_signatures"]
    )


def test_ncu_plan_validation_cli_returns_nonzero_for_provisional(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({
        "schema_version": "invalid", "content_sha256": "0" * 64,
        "campaign": {"sha256": "a" * 64}, "capture_groups": [],
    }), encoding="utf-8")
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps({
        "status": "failed_preflight", "launches": [], "incomplete_launches": [],
    }), encoding="utf-8")
    assert ncu_plan_main([
        "--plan", str(plan_path), "--ncu-profile", str(profile_path),
        "--output", str(tmp_path / "evidence.json"),
    ]) == 2


def test_frozen_profile_identity_allows_only_model_output_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    working = tmp_path / "upstream"
    working.mkdir()
    script = working / "train.py"
    script.write_text("pass\n", encoding="utf-8")
    freeze_path = tmp_path / "freeze.json"
    freeze_path.write_text("{}", encoding="utf-8")
    freeze = {
        "run_manifest_sha256": "a" * 64,
        "model": {"commit": "b" * 40, "tree_sha256": "e" * 64},
        "dataset": {
            "manifest_sha256": "c" * 64, "name": "Chest",
            "source_url": "https://data.example", "license_url": "https://license.example",
        },
        "training": {"command": {
            "argv": ["python", "train.py", "-s", "/data/chest", "-m", "/old"],
            "working_directory": str(working), "sha256": "d" * 64,
        }},
    }
    monkeypatch.setattr("gala_sim.adapters.stage_runner.verify_freeze_record", lambda value: None)
    monkeypatch.setattr("gala_sim.adapters.stage_runner.json.loads", lambda value: freeze)
    monkeypatch.setattr(
        "gala_sim.adapters.stage_runner.subprocess.check_output",
        lambda command, text: "b" * 40 + "\n" if "rev-parse" in command else "",
    )
    monkeypatch.setattr("gala_sim.adapters.stage_runner.sha256_tree", lambda path: "e" * 64)
    monkeypatch.setattr(
        "gala_sim.adapters.stage_runner.dataset_record",
        lambda *args: argparse.Namespace(manifest_sha256="c" * 64),
    )
    monkeypatch.chdir(working)
    identity = _frozen_profile_identity(
        freeze_path, script.resolve(), ["-s", "/data/chest", "-m", str(tmp_path / "new")]
    )
    assert identity["model_commit"] == "b" * 40
    with pytest.raises(ValueError, match="arguments"):
        _frozen_profile_identity(
            freeze_path, script.resolve(),
            ["-s", "/data/other", "-m", str(tmp_path / "new")],
        )
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        _frozen_profile_identity(
            freeze_path, script.resolve(), ["-s", "/data/chest", "-m", str(existing)]
        )


def test_nsys_stage_parser_maps_kernels_and_memcpy(tmp_path: Path) -> None:
    database = tmp_path / "profile.sqlite"
    connection = sqlite3.connect(database)
    connection.executescript(
        "CREATE TABLE NVTX_EVENTS(start INTEGER, end INTEGER, text TEXT);"
        "CREATE TABLE StringIds(id INTEGER, value TEXT);"
        "CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME("
        "start INTEGER, end INTEGER, correlationId INTEGER);"
        "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL("
        "start INTEGER, end INTEGER, gridX INTEGER, gridY INTEGER, gridZ INTEGER,"
        "blockX INTEGER, blockY INTEGER, blockZ INTEGER, demangledName INTEGER,"
        "correlationId INTEGER);"
        "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY("
        "start INTEGER, end INTEGER, bytes INTEGER, correlationId INTEGER);"
    )
    connection.execute(
        "INSERT INTO NVTX_EVENTS VALUES(?,?,?)",
        (100, 300, "gala_stage:backward:iteration=600:call=9"),
    )
    connection.execute(
        "INSERT INTO NVTX_EVENTS VALUES(?,?,?)",
        (
            90, 310, "gala_iteration:training:iteration=600:call=1:campaign="
            + "a" * 64,
        ),
    )
    connection.execute("INSERT INTO StringIds VALUES(?,?)", (7, "backward_kernel"))
    connection.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?)", (120, 130, 11)
    )
    connection.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?)", (140, 150, 12)
    )
    connection.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,?,?,?,?,?,?)",
        (320, 420, 2, 1, 1, 128, 1, 1, 7, 11),
    )
    connection.execute(
        "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES(?,?,?,?)", (425, 450, 4096, 12)
    )
    connection.commit()
    connection.close()

    result = parse_nsys_sqlite(database)
    backward = result["stage_summaries"]["backward"]
    assert backward["kernel_launch_count"] == 1
    assert backward["gpu_kernel_time_ms"] == 0.0001
    assert backward["memcpy_bytes"] == 4096
    assert backward["kernels"][0]["name"] == "backward_kernel"
    assert result["kernel_coverage"]["status"] == "complete"
    assert result["kernel_coverage"]["assigned_kernel_count"] == 1
    assert result["run_identity"]["profiling_campaign_sha256"] == "a" * 64


def test_ncu_stage_parser_builds_counter_inputs(tmp_path: Path) -> None:
    path = tmp_path / "ncu.csv"
    range_column = "thread Domain:Push/Pop_Range"
    fieldnames = [
        "ID", range_column, "Kernel Name", "Block Size", "Grid Size",
        "Metric Name", "Metric Value",
    ]
    metrics = {
        "dram__bytes_read.sum": "64",
        "dram__bytes_write.sum": "32",
        "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum": "10",
        "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum": "4",
        "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum": "2",
        "smsp__inst_executed_pipe_xu.sum": "8",
        "lts__t_requests_op_atom.sum": "3",
    }
    with path.open("w", encoding="utf-8", newline="") as stream:
        stream.write("==PROF== fixture\n")
        writer = csv.DictWriter(stream, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for name, value in metrics.items():
            writer.writerow({
                "ID": "0",
                    range_column: (
                        "gala_stage:projection_forward:iteration=600:call=1:campaign="
                        + "a" * 64
                    ),
                    "Kernel Name": "exp_kernel",
                    "Block Size": "(32, 1, 1)",
                    "Grid Size": "(4, 1, 1)",
                "Metric Name": name,
                "Metric Value": value,
            })
    result = parse_ncu_csv(path)
    stage = result["stage_summaries"]["projection_forward"]
    assert stage["kernel_launch_count"] == 1
    assert stage["dram_bytes"] == 96
    assert stage["fp32_operations"] == 26
    assert stage["fp32_fma_equivalent"] == 13
    assert stage["atomic_requests"] == 3
    assert stage["unclassified_transcendental_operations"] == 8
    assert stage["weight_eligible"] is False
    assert result["run_identity"]["profiling_campaign_sha256"] == "a" * 64
    assert result["launches"][0]["kernel_ordinal_in_call"] == 1


def test_ncu_stage_parser_accepts_cross_thread_start_stop_ranges(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ncu.csv"
    range_column = "Id:Domain:Start/Stop_Range"
    fieldnames = [
        "ID", range_column, "Kernel Name", "Block Size", "Grid Size",
        "Metric Name", "Metric Value",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for name in (
            "dram__bytes_read.sum", "dram__bytes_write.sum",
            "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
            "smsp__inst_executed_pipe_xu.sum", "lts__t_requests_op_atom.sum",
        ):
            writer.writerow({
                "ID": "0",
                range_column: (
                    "0:<default domain>:gala_ncu_stage:backward:iteration=499:"
                    "call=1084:campaign=" + "a" * 64
                ),
                "Kernel Name": "worker_kernel", "Block Size": "(32, 1, 1)",
                "Grid Size": "(4, 1, 1)", "Metric Name": name,
                "Metric Value": "1",
            })
    result = parse_ncu_csv(path)
    assert result["status"] == "passed"
    assert result["launches"][0]["stage"] == "backward"
    assert result["launches"][0]["call_index"] == 1084


def test_ncu_sass_parser_keeps_dynamic_opcode_evidence_provisional(tmp_path: Path) -> None:
    path = tmp_path / "source.csv"
    fieldnames = [
        "Address", "Source", "Instructions Executed",
        "Predicated-On Thread Instructions Executed",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Kernel Name", "exp_kernel"])
        writer.writerow(fieldnames)
        writer.writerow(["0x1", "      MUFU.EX2 R0, R1", "100", "100"])
        writer.writerow(["0x2", "      MUFU.RCP R0, R1", "40", "40"])
    profile = {
        "stage_summaries": {"projection_forward": {
            "unclassified_xu_instructions": 140.0,
            "unclassified_transcendental_operations": 140.0,
            "weight_eligible": False,
        }},
        "launches": [{
            "stage": "projection_forward", "iteration": 1, "call_index": 1,
            "launch_id": "0", "kernel_name": "exp_kernel",
        }],
    }
    result = classify_sass_csv(path, profile)
    launch = result["launches"][0]
    assert launch["exp_operations"] == 100
    assert launch["rcp_operations"] == 40
    assert result["stage_summaries"]["projection_forward"]["weight_eligible"] is False
    assert result["stage_summaries"]["projection_forward"]["sass_classification"]["xu_coverage_complete"] is False


def test_ncu_sass_parser_classifies_known_mufu_and_atomic_variants(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source.csv"
    fieldnames = [
        "Address", "Source", "Instructions Executed",
        "Predicated-On Thread Instructions Executed",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Kernel Name", "known_kernel"])
        writer.writerow(fieldnames)
        rows = [
            ("0x1", "      MUFU.RSQ R0, R1", "4", "4"),
            ("0x2", "      MUFU.RCP64H R0, R1", "5", "5"),
            ("0x3", "      MUFU.RSQ64H R0, R1", "6", "6"),
            ("0x3b", "      F2I.TRUNC.NTZ R0, R1", "3", "3"),
            ("0x4", "      ATOMS.ADD R0, [R1], R2", "7", "7"),
            ("0x5", "      RED.E.ADD.STRONG.GPU [R1], R2", "8", "8"),
            ("0x6", "      REDUX.OR R0, R1", "9", "9"),
        ]
        writer.writerows(rows)
    profile = {
        "stage_summaries": {"projection_forward": {
            "unclassified_xu_instructions": 18.0,
            "unclassified_transcendental_operations": 18.0,
            "weight_eligible": False,
        }},
        "launches": [{
            "stage": "projection_forward", "iteration": 1, "call_index": 1,
            "launch_id": "0", "kernel_name": "known_kernel",
            "metrics": {"xu_instructions": 18.0},
        }],
    }
    result = classify_sass_csv(path, profile)
    launch = result["launches"][0]
    assert launch["sqrt_operations"] == 10
    assert launch["rcp_operations"] == 5
    assert launch["conversion_operations"] == 3
    assert launch["atomic_operations"] == 15
    assert launch["sass_classification"]["status"] == "passed"
    assert launch["sass_classification"]["unsupported_opcodes"] == {}
    assert result["stage_summaries"]["projection_forward"]["weight_eligible"] is True


def test_ncu_sass_parser_preserves_identical_consecutive_launches(
    tmp_path: Path,
) -> None:
    path = tmp_path / "source.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        for _launch in range(2):
            for _duplicate_export in range(2):
                writer.writerow(["Kernel Name", "same_kernel"])
                writer.writerow([
                    "Address", "Source", "Instructions Executed",
                    "Predicated-On Thread Instructions Executed",
                ])
                writer.writerow(["0x1", "      MUFU.EX2 R0, R1", "4", "128"])
    launches = [{
        "stage": "projection_forward", "iteration": 1, "call_index": 1,
        "launch_id": str(index), "kernel_name": "same_kernel",
        "metrics": {"xu_instructions": 4.0},
    } for index in range(2)]
    result = classify_sass_csv(path, {
        "stage_summaries": {"projection_forward": {
            "unclassified_xu_instructions": 8.0,
            "unclassified_transcendental_operations": 8.0,
        }},
        "launches": launches,
    })
    assert result["status"] == "passed"
    assert result["unmatched_source_sections"] == []
    assert result["unmatched_launch_ids"] == []
    assert [launch["exp_operations"] for launch in result["launches"]] == [128, 128]


def _calibration(scale: float, *, missing: str | None = None) -> dict:
    categories = (
        "fp32_fma", "exp", "log", "rcp", "sqrt", "memory_bandwidth",
        "atomic", "kernel_launch", "synchronization",
    )
    vectors = {}
    units = {
        "fp32_fma": "fma", "exp": "element", "log": "element",
        "rcp": "element", "sqrt": "element", "memory_bandwidth": "byte",
        "atomic": "atomic_request", "kernel_launch": "launch",
        "synchronization": "synchronize_call",
    }
    for category in categories:
        if category == missing:
            continue
        unit = units[category]
        vectors[category] = {"1": {
            "median_seconds_per_work": 1e-6 * scale, "work_unit": unit,
        }}
        if category not in {"kernel_launch", "synchronization"}:
            vectors[category]["1000"] = {
                "median_seconds_per_work": 1e-6 * scale, "work_unit": unit,
            }
    vectors["synchronization"] = {
        "idle": {"median_seconds_per_work": 1e-6 * scale, "work_unit": "call"}
    }
    return {"status": "passed", "vectors": vectors, "device": {"name": "fixture"}}


def test_gpu_stage_normalization_weights_sum_and_convert() -> None:
    stage_profile = {
        "stage_coverage": {"status": "partial"},
        "stage_summaries": {"projection_forward": {"total_ms": 10.0}},
    }
    nsys_profile = {
        "status": "passed", "kernel_coverage": {"status": "complete"},
    }
    ncu_profile = {"stage_summaries": {"projection_forward": {
        "fp32_fma_equivalent": 100.0,
        "dram_bytes": 200.0,
        "atomic_requests": 10.0,
        "kernel_launch_count": 2,
        "unclassified_transcendental_operations": 0,
        "weight_eligible": True,
    }}}
    result = normalize_stage_profiles(
        stage_profile, ncu_profile, _calibration(1.0), _calibration(2.0),
        nsys_profile=nsys_profile,
        required_stages=("projection_forward",),
    )
    stage = result["stages"]["projection_forward"]
    assert result["status"] == "passed"
    assert result["formal_performance_eligible"] is True
    assert stage["weight_sum"] == pytest.approx(1.0)
    assert stage["orin_ms"] == pytest.approx(20.0)


def test_gpu_stage_normalization_withheld_when_orin_vector_is_incomplete() -> None:
    stage_profile = {
        "stage_coverage": {"status": "complete"},
        "stage_summaries": {"projection_forward": {"total_ms": 10.0}},
    }
    ncu_profile = {"stage_summaries": {"projection_forward": {
        "fp32_fma_equivalent": 100.0,
        "dram_bytes": 200.0,
        "unclassified_transcendental_operations": 0,
        "weight_eligible": True,
    }}}
    nsys_profile = {
        "status": "passed", "kernel_coverage": {"status": "complete"},
    }
    result = normalize_stage_profiles(
        stage_profile, ncu_profile, _calibration(1.0),
        _calibration(2.0, missing="memory_bandwidth"),
        nsys_profile=nsys_profile,
        required_stages=("projection_forward",),
    )
    stage = result["stages"]["projection_forward"]
    assert result["status"] == "provisional_normalization"
    assert result["formal_performance_eligible"] is False
    assert stage["weight_sum"] == pytest.approx(1.0)
    assert stage["orin_ms"] is None


def test_gpu_stage_normalization_requires_exact_nsys_kernel_coverage() -> None:
    stage_profile = {
        "stage_coverage": {"status": "complete"},
        "stage_summaries": {"projection_forward": {"total_ms": 10.0}},
    }
    ncu_profile = {"stage_summaries": {"projection_forward": {
        "fp32_fma_equivalent": 100.0,
        "unclassified_transcendental_operations": 0,
        "weight_eligible": True,
    }}}
    result = normalize_stage_profiles(
        stage_profile, ncu_profile, _calibration(1.0), _calibration(2.0),
        nsys_profile={
            "status": "passed", "kernel_coverage": {"status": "incomplete"},
        },
        required_stages=("projection_forward",),
    )
    assert result["formal_performance_eligible"] is False
    assert "nsys_kernel_coverage_incomplete" in result["provisional_reasons"]
