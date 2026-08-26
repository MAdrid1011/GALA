from __future__ import annotations

import csv
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
from gala_sim.tools.gpu_normalization import normalize_stage_profiles


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
        (90, 310, "gala_iteration:training:iteration=600:call=1"),
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
                    range_column: "gala_stage:projection_forward:iteration=600:call=1",
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
