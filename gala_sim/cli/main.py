"""Machine-readable simulator command line."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import shlex
import sys
import time

from gala_sim.ablation import (
    run_archive_matrix, run_archive_speedup_diagnostic, run_matrix,
)
from gala_sim.campaign import (
    ALL_COMBINATIONS, DATASET_IDS, MODEL_IDS, run_campaigns,
)
from gala_sim.config import load_config, pending_parameters
from gala_sim.results import (
    AblationRow, asic_speedup, comparison_baseline, write_ablation_csv,
)
from gala_sim.results.run import RunOutputWriter
from gala_sim.results.manifest import write_json
from gala_sim.identity import sha256_file
from gala_sim.mechanisms import (
    CANONICAL_VARIANT_POLICIES, CYCLE_POLICY_NAMES,
)
from gala_sim.timing import (
    CycleConfig, CycleEngine, RelationPacketPlan, analyze_cycle_lower_bounds,
)
from gala_sim.timing.memory import NativeRamulator2Binding, Ramulator2Backend
from gala_sim.timing.resources import ResourceUsage
from gala_sim.tools.cycle_preflight import run_cycle_preflight, write_cycle_preflight
from gala_sim.tools.relation_capacity import run_relation_capacity_preflight
from gala_sim.tools.representative_packets import (
    build_representative_packet_trace, plan_representative_packet_groups,
)
from gala_sim.tools.cycle_throughput import (
    ThroughputConverged, ThroughputDiagnosticConfig, ThroughputMonitor,
    require_empty_diagnostic_output,
)
from gala_sim.tools.inactivity import (
    InactivityTimeoutError, InactivityWatchdog, observe_cycle_progress,
)
from gala_sim.tools.preflight import run_native_preflight
from gala_sim.adapters.native_reference import run_native_reference
from gala_sim.assets import AssetCatalog, AssetSelection, acquire_assets
from gala_sim.workspace import WorkspacePaths
from gala_sim.trace import (
    CAPTURED_PACKET_SAMPLE_SCHEMA_VERSION, CapturedPacketSpec, QueryDomain,
    QUERY_PACKET_SAMPLE_SCHEMA_VERSIONS, QueryPacketSampleConfig, QueryRange,
    TraceReader, TraceSampleConfig, TraceValidationConfig, TraceWriter,
    VirtualPacketArchiveReader,
    complete_captured_packet_sample, dependency_closed_query_sample,
    derive_quick_relation_packets, real_query_packet_sample,
    snapshot_live_archive_prefix,
    validate_packet_derivation, validate_trace,
)


def _iteration_range(value: str) -> tuple[int, int]:
    start_text, separator, end_text = value.partition(":")
    if not separator:
        raise argparse.ArgumentTypeError("iteration range must use START:END")
    try:
        start, end = int(start_text), int(end_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("iteration range bounds must be integers") from error
    if start <= 0 or end < start:
        raise argparse.ArgumentTypeError("iteration range must have 1 <= START <= END")
    return start, end


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gala-sim")
    parser.add_argument(
        "--repository", type=Path, default=None,
        help="repository root used to resolve relative configuration paths",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    workspace = commands.add_parser("workspace-check")
    workspace.add_argument("--workspace", type=Path, default=None)
    acquire = commands.add_parser("acquire")
    acquire.add_argument("--workspace", type=Path, default=None)
    acquire.add_argument("--all", dest="acquire_all", action="store_true")
    acquire.add_argument("--models", type=_csv_ids, default=())
    acquire.add_argument("--datasets", type=_csv_ids, default=())
    acquire.add_argument("--dry-run", action="store_true")
    acquire.add_argument("--no-resume", action="store_true")
    campaign = commands.add_parser(
        "campaign-ablation",
        help="run bounded CPU trace, bound, and seven-variant validation campaigns",
    )
    campaign.add_argument("--workspace", type=Path, default=None)
    campaign.add_argument("--models", type=_csv_ids, default=())
    campaign.add_argument("--datasets", type=_csv_ids, default=())
    campaign.add_argument("--all", dest="campaign_all", action="store_true")
    campaign.add_argument("--parallel-workers", type=int, default=1)
    campaign_modes = campaign.add_mutually_exclusive_group()
    campaign_modes.add_argument(
        "--representative-archive",
        action="store_true",
        help=(
            "exercise the packet-archive and representative replay path for "
            "bounded CPU validation"
        ),
    )
    campaign_modes.add_argument(
        "--official-trace",
        action="store_true",
        help=(
            "capture a model-specific trace through the official entrypoint "
            "before bounds and ablations"
        ),
    )
    campaign.add_argument(
        "--iterations", type=int, default=1,
        help="repeat the bounded optimizer transaction for amortized validation",
    )
    campaign.add_argument(
        "--capture-iteration-range", type=_iteration_range, default=(1, 1),
        metavar="START:END",
        help="official trace window; execution stops when this window closes",
    )
    campaign.add_argument(
        "--gpu-measurement-root",
        type=Path,
        default=None,
        help=(
            "root containing MODEL/DATASET/gpu-compiler-measurement.json; "
            "official campaigns reject missing or mismatched evidence"
        ),
    )
    config = commands.add_parser("config-check")
    config.add_argument("--config", type=Path, required=True)
    preflight = commands.add_parser("cycle-preflight")
    preflight.add_argument("--config", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)
    preflight.add_argument("--ramulator-binding", default=None,
                           help="Python object spec module:attribute exposing the async binding API")
    preflight.add_argument("--ramulator-build-manifest", type=Path, default=None)
    preflight.add_argument("--ramulator-config", type=Path, default=None)
    preflight.add_argument("--resource-usage", type=Path, default=None,
                           help="JSON ResourceUsage snapshot")
    relation_capacity = commands.add_parser("relation-capacity-preflight")
    relation_capacity.add_argument("--archive", type=Path, required=True)
    relation_capacity.add_argument("--config", type=Path, required=True)
    relation_capacity.add_argument("--output", type=Path, required=True)
    native = commands.add_parser("native-preflight")
    native.add_argument("--config", type=Path, required=True)
    native.add_argument("--freeze", type=Path, required=True)
    native.add_argument("--output", type=Path, required=True)
    reference = commands.add_parser("native-reference")
    reference.add_argument("--config", type=Path, required=True)
    reference.add_argument("--freeze", type=Path, required=True)
    reference.add_argument("--preflight", type=Path, required=True)
    reference.add_argument("--output", type=Path, required=True)
    trace = commands.add_parser("trace-validate")
    trace.add_argument("--trace", type=Path, required=True)
    trace.add_argument("--scan-events", type=int, default=None)
    trace.add_argument("--index-directory", type=Path, default=None)
    archive_validate = commands.add_parser("trace-archive-validate")
    archive_validate.add_argument("--archive", type=Path, required=True)
    archive_validate.add_argument("--output", type=Path, required=True)
    archive_validate.add_argument("--prefetch-chunks", type=int, default=1)
    archive_validate.add_argument("--parallel-workers", type=int, default=1)
    archive_snapshot = commands.add_parser("trace-archive-snapshot")
    archive_snapshot.add_argument("--archive", type=Path, required=True)
    archive_snapshot.add_argument("--output", type=Path, required=True)
    archive_snapshot.add_argument("--initial-gaussian-count", type=int, required=True)
    archive_snapshot.add_argument("--through-iteration", type=int, default=None)
    representative = commands.add_parser("representative-packet-plan")
    representative.add_argument("--archive", type=Path, required=True)
    representative.add_argument("--campaign", type=Path)
    representative.add_argument("--expected-groups", type=int, required=True)
    representative.add_argument(
        "--single-iteration", type=int,
        help="select all templates from one closed iteration instead of a phase window",
    )
    representative.add_argument(
        "--live-prefix", action="store_true",
        help="plan only campaign windows closed by a live archive snapshot",
    )
    representative_trace = commands.add_parser("representative-packet-trace")
    representative_trace.add_argument("--archive", type=Path, required=True)
    representative_trace.add_argument("--plan", type=Path, required=True)
    representative_trace.add_argument("--window-index", type=int, required=True)
    representative_trace.add_argument("--max-events", type=int, required=True)
    representative_trace.add_argument("--query-lanes", type=int, required=True)
    representative_trace.add_argument(
        "--model-id",
        help="required only for an archive captured before provenance metadata",
    )
    representative_trace.add_argument(
        "--dataset-id",
        help="required only for an archive captured before provenance metadata",
    )
    representative_trace.add_argument("--output", type=Path, required=True)
    representative.add_argument("--output", type=Path, required=True)
    sample = commands.add_parser("trace-sample")
    sample.add_argument("--trace", type=Path, required=True)
    sample.add_argument("--output", type=Path, required=True)
    sample.add_argument(
        "--query-range", action="append", required=True, type=_query_range,
        help="dependency-closed START:COUNT query range; may be repeated",
    )
    sample.add_argument("--max-events", type=int, required=True)
    sample.add_argument("--max-dependencies", type=int, required=True)
    sample.add_argument("--scan-events", type=int, required=True)
    sample.add_argument(
        "--scan-backend", choices=("auto", "cpu", "cuda"), default="auto",
    )
    query_packets = commands.add_parser("trace-query-packets")
    query_packets.add_argument("--trace", type=Path, required=True)
    query_packets.add_argument("--output", type=Path, required=True)
    query_packets.add_argument(
        "--query-range", action="append", required=True, type=_query_range,
        help="consecutive-iteration START:COUNT range; may be repeated",
    )
    query_packets.add_argument("--scan-events", type=int, required=True)
    query_packets.add_argument(
        "--scan-backend", choices=("auto", "cpu", "cuda"), default="auto",
    )
    query_packets.add_argument("--query-lanes", type=int, required=True)
    query_packets.add_argument("--ssim-radius", type=int, default=0)
    packetize = commands.add_parser("trace-packetize")
    packetize.add_argument("--trace", type=Path, required=True)
    packetize.add_argument("--output", type=Path, required=True)
    packetize.add_argument(
        "--query-domain", action="append", required=True, type=_query_domain,
        help="TEMPLATE:BASE:DIMxDIM[xDIM] row-major query domain",
    )
    packetize.add_argument("--query-lanes", type=int, required=True)
    captured = commands.add_parser("trace-captured-packets")
    captured.add_argument("--manifest", type=Path, required=True)
    captured.add_argument("--output", type=Path, required=True)
    captured.add_argument("--max-events", type=int, required=True)
    captured.add_argument("--query-lanes", type=int, required=True)
    captured.add_argument("--initial-gaussian-count", type=int, required=True)
    replay = commands.add_parser("cycle-replay")
    replay.add_argument("--trace", type=Path, required=True)
    replay.add_argument("--config", type=Path, required=True)
    replay.add_argument("--ramulator-binding", default=None,
                        help="Python object spec module:attribute exposing the async binding API")
    replay.add_argument("--ramulator-build-manifest", type=Path, default=None)
    replay.add_argument("--ramulator-config", type=Path, default=None)
    replay.add_argument("--resource-usage", type=Path, required=True,
                        help="JSON ResourceUsage snapshot")
    replay.add_argument("--policy", default="base", choices=CYCLE_POLICY_NAMES)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--quick-validation", action="store_true")
    replay.add_argument("--throughput-progress", action="store_true")
    replay.add_argument("--stop-when-throughput-stable", action="store_true")
    replay.add_argument("--compute-telemetry", action="store_true")
    bounds = commands.add_parser("cycle-bounds")
    bounds.add_argument("--trace", type=Path, required=True)
    bounds.add_argument("--config", type=Path, required=True)
    bounds.add_argument("--ramulator-binding", default=None)
    bounds.add_argument("--ramulator-build-manifest", type=Path, default=None)
    bounds.add_argument("--ramulator-config", type=Path, default=None)
    bounds.add_argument("--resource-usage", type=Path, required=True)
    bounds.add_argument("--output", type=Path, required=True)
    bounds.add_argument("--quick-validation", action="store_true")
    bounds.add_argument("--base-cycles", type=int, required=True)
    bounds.add_argument("--query-target", type=float, required=True)
    bounds.add_argument("--residency-target", type=float, required=True)
    bounds.add_argument("--full-target", type=float, required=True)
    ablation = commands.add_parser("ablation")
    ablation.add_argument("--trace", type=Path, required=True)
    ablation.add_argument("--config", type=Path, required=True)
    ablation.add_argument("--ramulator-binding", default=None,
                          help="Python object spec module:attribute exposing the async binding API")
    ablation.add_argument("--ramulator-build-manifest", type=Path, default=None)
    ablation.add_argument("--ramulator-config", type=Path, default=None)
    ablation.add_argument("--resource-usage", type=Path, required=True,
                          help="JSON ResourceUsage snapshot")
    ablation.add_argument("--output", type=Path, required=True)
    ablation.add_argument("--quick-validation", action="store_true")
    ablation.add_argument("--parallel-workers", type=int, default=1)
    archive_ablation = commands.add_parser("archive-ablation")
    archive_ablation.add_argument("--archive", type=Path, required=True)
    archive_ablation.add_argument("--config", type=Path, required=True)
    archive_ablation.add_argument("--ramulator-binding", default=None,
                                  help="Python object spec module:attribute exposing the async binding API")
    archive_ablation.add_argument("--ramulator-build-manifest", type=Path, default=None)
    archive_ablation.add_argument("--ramulator-config", type=Path, default=None)
    archive_ablation.add_argument("--resource-usage", type=Path, required=True,
                                  help="JSON ResourceUsage snapshot")
    archive_ablation.add_argument("--output", type=Path, required=True)
    archive_ablation.add_argument("--model", required=True)
    archive_ablation.add_argument("--dataset", required=True)
    archive_ablation.add_argument("--quick-validation", action="store_true")
    archive_ablation.add_argument("--parallel-workers", type=int, default=1)
    archive_ablation.add_argument(
        "--stop-when-speedup-stable", action="store_true",
        help=(
            "stop all seven variants at a common stable iteration boundary; "
            "writes a development projection instead of a formal matrix"
        ),
    )
    return parser


def _csv_ids(value: str) -> tuple[str, ...]:
    identifiers = tuple(item.strip() for item in value.split(",") if item.strip())
    if not identifiers:
        raise argparse.ArgumentTypeError("asset list must contain an identifier")
    return identifiers


def _query_range(value: str) -> QueryRange:
    start, separator, count = value.partition(":")
    if not separator:
        raise argparse.ArgumentTypeError("query range must use START:COUNT")
    try:
        return QueryRange(int(start), int(count))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _query_domain(value: str) -> QueryDomain:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "query domain must use TEMPLATE:BASE:DIMxDIM[xDIM]"
        )
    try:
        shape = tuple(int(extent) for extent in parts[2].split("x"))
        return QueryDomain(int(parts[0]), int(parts[1]), shape)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _repository_path(path: Path, repository: Path | None) -> Path:
    if path.is_absolute():
        return path
    return WorkspacePaths.discover(repository=repository).repository / path


def _load_binding(spec: str | None, build_manifest: Path | None,
                  configuration: Path | None) -> Ramulator2Backend | None:
    if spec and (build_manifest is not None or configuration is not None):
        raise ValueError("choose either a Python binding or a native build manifest")
    if (build_manifest is None) != (configuration is None):
        raise ValueError("native Ramulator binding requires build manifest and configuration")
    if build_manifest is not None and configuration is not None:
        return Ramulator2Backend(
            NativeRamulator2Binding.from_build_manifest(build_manifest, configuration)
        )
    if not spec:
        return None
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("Ramulator binding must use module:attribute syntax")
    module = importlib.import_module(module_name)
    binding = getattr(module, attribute)
    if isinstance(binding, type):
        binding = binding()
    required = ("metadata", "try_issue", "tick", "drain_completions", "clone")
    missing = [name for name in required if not callable(getattr(binding, name, None))]
    if missing:
        raise ValueError("Ramulator binding object lacks: " + ", ".join(missing))
    return Ramulator2Backend(binding)


def _load_resource_usage(path: Path | None) -> ResourceUsage | None:
    if path is None:
        return None
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("resource usage snapshot must be a JSON object")
    return ResourceUsage(
        shared_sram_bytes=int(document["shared_sram_bytes"]),
        pods=int(document["pods"]),
        clusters=int(document["clusters"]),
        fma_lanes=int(document["fma_lanes"]),
        transcendental_lanes=int(document["transcendental_lanes"]),
        external_channels=int(document["external_channels"]),
        regions={str(key): int(value) for key, value in document["regions"].items()},
    )


def _write_archive_ablation_outputs(
    runs, *, output: Path, config: CycleConfig, model: str, dataset: str,
    formal_performance_eligible: bool, validation_report: dict[str, object],
    archive: Path,
) -> None:
    """Write the same auditable rows as expanded-trace ablation replay."""
    if output.exists():
        raise ValueError("archive ablation output CSV must not already exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    base_cycles = next(run.result.total_cycles for run in runs if run.variant.bits == "0000")
    breakdown_root = output.parent / f"{output.name}.modules"
    result_root = output.parent / f"{output.name}.runs"
    if any(path.exists() for path in (breakdown_root, result_root)):
        raise ValueError("archive ablation artifact directories must not already exist")
    rows: list[AblationRow] = []
    paths: dict[str, str] = {}
    for run in runs:
        bits = run.variant.bits
        variant_root = result_root / bits
        RunOutputWriter(variant_root).write_cycles(run.result)
        stall_counts: dict[str, int] = {}
        for stall in run.result.stalls:
            stall_counts[stall.module] = stall_counts.get(stall.module, 0) + stall.count
        breakdown = breakdown_root / f"{bits}.json"
        write_json({
            "schema_version": "gala-ablation-module-breakdown-v1",
            "result_scope": (
                "formal_performance" if formal_performance_eligible
                else "quick_cycle_validation"
            ),
            "formal_performance_eligible": formal_performance_eligible,
            "bits": bits,
            "run_id": f"{model}-{dataset}-archive-{bits}",
            "policy": run.result.policy,
            "total_cycles": run.result.total_cycles,
            "module_counters": run.result.module_counters,
            "module_busy_cycles": {
                name: int(counters.get("busy_cycles", 0))
                for name, counters in run.result.module_counters.items()
            },
            "stall_counts_by_module": stall_counts,
            "event_counts": run.result.event_counts,
            "variant_result_path": str(variant_root.relative_to(output.parent)),
        }, breakdown)
        paths[bits] = str(breakdown.relative_to(output.parent))
        rows.append(AblationRow(
            model=model, dataset=dataset, bits=bits,
            cycles=run.result.total_cycles,
            comparison_baseline=comparison_baseline(bits),
            speedup_vs_base_asic=asic_speedup(
                bits, base_cycles=base_cycles, cycles=run.result.total_cycles,
            ),
            gpu_base_seconds=None, speedup_vs_gpu_base=None,
            local_gpu_seconds=None, orin_seconds=None, speedup_vs_orin=None,
            psnr_delta_db=None, ssim_delta=None, lpips_delta=None,
            config_sha256=str(config.config_sha256 or ""), status="passed",
            module_breakdown_path=paths[bits],
            run_id=f"{model}-{dataset}-archive-{bits}",
        ))
    write_ablation_csv(rows, output)
    full_cycles = next(run.result.total_cycles for run in runs if run.variant.bits == "1111")
    write_json({
        "schema_version": "gala-ablation-manifest-v2",
        "result_scope": (
            "formal_performance" if formal_performance_eligible
            else "quick_cycle_validation"
        ),
        "formal_performance_eligible": formal_performance_eligible,
        "archive": str(archive.resolve()),
        "archive_validation": validation_report,
        "module_breakdown_directory": str(breakdown_root.relative_to(output.parent)),
        "variant_result_directory": str(result_root.relative_to(output.parent)),
        "module_breakdown_paths": paths,
        "full_alias": {
            "policy": "full", "variant_bits": "1111",
            "run_id": f"{model}-{dataset}-archive-1111",
            "cycles": full_cycles,
            "selection_contract_equal": (
                CycleEngine._selection_for_policy("full")
                == CycleEngine._selection_for_policy("variant:1111")
            ),
        },
    }, output.with_suffix(output.suffix + ".manifest.json"))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "workspace-check":
            paths = WorkspacePaths.discover(
                repository=args.repository, workspace=args.workspace,
            ).ensure()
            report = {
                "ready": True,
                "repository": paths.reference(paths.repository),
                "workspace": paths.reference(paths.root),
                "directories": [paths.reference(path) for path in paths.managed_directories],
            }
            print(json.dumps(report, sort_keys=True))
            return 0
        if args.command == "acquire":
            if not args.acquire_all and not args.models and not args.datasets:
                raise ValueError("acquire requires --all, --models, or --datasets")
            paths = WorkspacePaths.discover(
                repository=args.repository, workspace=args.workspace,
            )
            report = acquire_assets(
                AssetCatalog.load(paths.repository / "configs"),
                AssetSelection(args.models, args.datasets, args.acquire_all),
                paths, resume=not args.no_resume, dry_run=args.dry_run,
            )
            print(json.dumps(report.as_dict(), sort_keys=True))
            return 0
        if args.command == "campaign-ablation":
            if args.campaign_all:
                selections = ALL_COMBINATIONS
            else:
                models = args.models or MODEL_IDS
                datasets = args.datasets or DATASET_IDS
                selections = tuple((model, dataset) for model in models for dataset in datasets)
            paths = WorkspacePaths.discover(
                repository=args.repository, workspace=args.workspace,
            )
            results = run_campaigns(
                selections,
                workspace=paths,
                parallel_workers=args.parallel_workers,
                iterations=args.iterations,
                representative_archive=args.representative_archive,
                official_trace=args.official_trace,
                capture_iteration_range=args.capture_iteration_range,
                gpu_measurement_root=args.gpu_measurement_root,
            )
            print(json.dumps({
                "status": "passed",
                "result_scope": (
                    "official_model_trace_validation"
                    if args.official_trace else (
                        "quick_cpu_packet_archive_validation"
                        if args.representative_archive
                        else "quick_cpu_trace_validation"
                    )
                ),
                "formal_performance_eligible": False,
                "iterations": args.iterations,
                "campaigns": [
                    {
                        "model_id": item.model_id,
                        "dataset_id": item.dataset_id,
                        "trace": paths.reference(item.trace_root),
                        "bounds": paths.reference(item.bounds_path),
                        "ablation": paths.reference(item.ablation_path),
                        "base_cycles": item.base_cycles,
                        "cycles": dict(item.ablation_cycles),
                        "upper_bound_status": dict(item.upper_bound_status),
                        "oracle_cycles": dict(item.oracle_cycles),
                        "oracle_speedups_vs_base_asic": dict(
                            item.oracle_speedups_vs_base_asic
                        ),
                        "gpu_base_speedups": dict(item.gpu_base_speedups),
                    }
                    for item in results
                ],
            }, sort_keys=True))
            return 0
        if args.command == "config-check":
            config = load_config(_repository_path(args.config, args.repository))
            print(json.dumps({"sha256": config.sha256, "ready": config.ready,
                              "pending": pending_parameters(config.parameters)},
                             sort_keys=True))
            return 0 if config.ready else 2
        if args.command == "cycle-preflight":
            config = load_config(_repository_path(args.config, args.repository))
            binding = _load_binding(
                args.ramulator_binding, args.ramulator_build_manifest,
                args.ramulator_config,
            )
            usage = _load_resource_usage(args.resource_usage)
            command = "gala-sim cycle-preflight --config " + str(args.config)
            report = run_cycle_preflight(
                config, memory_backend=binding, resource_usage=usage,
                reproduction=command,
            )
            write_cycle_preflight(report, args.output)
            print(json.dumps(report.as_dict(), sort_keys=True))
            return 0 if report.status == "passed" else 2
        if args.command == "relation-capacity-preflight":
            config = load_config(_repository_path(args.config, args.repository))
            report = run_relation_capacity_preflight(
                args.archive,
                config,
                progress=lambda item: print(
                    json.dumps({"phase": "relation_capacity", **item}, sort_keys=True),
                    file=sys.stderr,
                    flush=True,
                ),
            )
            write_json(report, args.output)
            print(json.dumps(report, sort_keys=True))
            return 0 if report["status"] == "passed" else 2
        if args.command == "native-preflight":
            config = load_config(_repository_path(args.config, args.repository))
            freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
            command = shlex.join([
                "gala-sim", "native-preflight", "--config", str(args.config),
                "--freeze", str(args.freeze), "--output", str(args.output),
            ])
            report = run_native_preflight(
                config, freeze, args.output, reproduction=command,
            )
            print(json.dumps(report.as_dict(), sort_keys=True))
            return 0 if report.status == "passed" else 2
        if args.command == "native-reference":
            config = load_config(_repository_path(args.config, args.repository))
            freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
            preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
            result = run_native_reference(config, freeze, preflight, args.output)
            print(json.dumps(result, sort_keys=True))
            return 0
        if args.command == "trace-archive-validate":
            reader = VirtualPacketArchiveReader(args.archive)
            last_progress_percent = -1

            def report_archive_progress(completed: int, total: int) -> None:
                nonlocal last_progress_percent
                percent = 100 if total == 0 else min(100, completed * 100 // total)
                if percent <= last_progress_percent and completed != total:
                    return
                last_progress_percent = percent
                print(json.dumps({
                    "phase": "archive_validation",
                    "completed_chunks": completed,
                    "total_chunks": total,
                    "percent": percent,
                }, sort_keys=True), file=sys.stderr, flush=True)

            report = reader.validate(
                promote=True, prefetch_chunks=args.prefetch_chunks,
                parallel_workers=args.parallel_workers,
                progress=(
                    report_archive_progress if args.parallel_workers > 1 else None
                ),
            )
            write_json(report, args.output)
            print(json.dumps(report, sort_keys=True))
            return 0
        if args.command == "trace-archive-snapshot":
            report = snapshot_live_archive_prefix(
                args.archive,
                args.output,
                initial_gaussian_count=args.initial_gaussian_count,
                through_iteration=args.through_iteration,
            )
            print(json.dumps({
                "status": "passed",
                "iteration_count": report["iteration_count"],
                "chunk_count": report["chunk_count"],
                "output": str(args.output.resolve()),
            }, sort_keys=True))
            return 0
        if args.command == "representative-packet-plan":
            report = plan_representative_packet_groups(
                args.archive,
                args.campaign,
                expected_group_count=args.expected_groups,
                live_prefix=args.live_prefix,
                single_iteration=args.single_iteration,
            )
            write_json(report, args.output)
            print(json.dumps({
                "status": "passed",
                "groups": report["group_count"],
                "iterations": report["iteration_count"],
                "campaign_complete": report["campaign_complete"],
                "output": str(args.output.resolve()),
            }, sort_keys=True))
            return 0
        if args.command == "representative-packet-trace":
            trace = build_representative_packet_trace(
                args.archive,
                args.plan,
                window_index=args.window_index,
                max_events=args.max_events,
                query_lanes=args.query_lanes,
                model_id=args.model_id,
                dataset_id=args.dataset_id,
            )
            TraceWriter().write(trace, args.output)
            sample = trace.metadata["trace_sample"]
            print(json.dumps({
                "status": "passed",
                "iterations": sample["iterations"],
                "events": trace.event_count,
                "dependencies": int(trace.dependencies.size),
                "output": str(args.output.resolve()),
            }, sort_keys=True))
            return 0
        if args.command == "archive-ablation":
            reader = VirtualPacketArchiveReader(args.archive)
            validation_report = reader.validate(promote=True)
            quick_scope = bool(args.quick_validation)
            if not quick_scope and not validation_report["formal_performance_eligible"]:
                raise ValueError(
                    "archive is not formally eligible; use --quick-validation only for a development replay"
                )
            gala_config = load_config(_repository_path(args.config, args.repository))
            binding = _load_binding(
                args.ramulator_binding, args.ramulator_build_manifest,
                args.ramulator_config,
            )
            usage = _load_resource_usage(args.resource_usage)
            preflight = run_cycle_preflight(
                gala_config, memory_backend=binding, resource_usage=usage,
                reproduction=("gala-sim archive-ablation --config " + str(args.config)),
            )
            preflight_output = args.output.parent / f"{args.output.name}.preflight"
            if preflight.status != "passed":
                write_cycle_preflight(preflight, preflight_output)
                print(json.dumps(preflight.as_dict(), sort_keys=True))
                return 2
            if binding is None:
                raise ValueError("archive ablation requires a Ramulator 2 binding")
            config = CycleConfig.from_gala(
                gala_config, binding, resource_usage=usage,
            )
            max_events = int(gala_config.value("trace.chunk_events"))
            max_frontier_events = max_events * int(
                gala_config.value("trace.max_inflight_chunks")
            )
            base_cycles: int | None = None
            progress_interval_seconds = float(
                gala_config.value("diagnostic.throughput_report_interval_seconds")
            )
            inactivity_timeout_seconds = float(
                gala_config.value("diagnostic.inactivity_timeout_seconds")
            )
            if args.stop_when_speedup_stable:
                if args.parallel_workers != 1:
                    raise ValueError(
                        "stable-speedup archive replay currently requires one "
                        "synchronized matrix worker"
                    )
                if args.output.exists():
                    raise ValueError(
                        "stable-speedup diagnostic output must not already exist"
                    )

                def speedup_progress(report) -> None:
                    latest = report["samples"][-1]
                    cycle_ratios = latest["cumulative_cycle_ratio_vs_0000"]
                    interval_ratios = latest["interval_cycle_ratio_vs_0000"]
                    print(json.dumps({
                        "phase": "archive_speedup_diagnostic",
                        "iteration": latest["iteration_id"],
                        "completed_iterations": latest["completed_iterations"],
                        "total_iterations": latest["total_iterations"],
                        "completion_fraction": latest["completion_fraction"],
                        "cycles_by_variant": latest["cycles_by_variant"],
                        "cycle_ratio_vs_0000": cycle_ratios,
                        "interval_cycle_ratio_vs_0000": interval_ratios,
                        "speedup_vs_base_asic": {
                            bits: cycle_ratios[bits]
                            for bits in ("1010", "0101", "1111")
                        },
                        "interval_speedup_vs_base_asic": {
                            bits: interval_ratios[bits]
                            for bits in ("1010", "0101", "1111")
                        } if interval_ratios else {},
                        "stability": report["stability"],
                    }, sort_keys=True), file=sys.stderr, flush=True)

                diagnostic = run_archive_speedup_diagnostic(
                    args.archive, config,
                    ThroughputDiagnosticConfig.from_gala(gala_config),
                    max_events=max_events,
                    max_frontier_events=max_frontier_events,
                    max_atomic_packet_events=max_frontier_events,
                    progress=speedup_progress,
                )
                write_json(diagnostic, args.output)
                print(json.dumps({
                    "status": "diagnostic_stopped"
                    if diagnostic["termination"] == "stopped_on_stable_speedup"
                    else "diagnostic_complete",
                    "termination": diagnostic["termination"],
                    "output": str(args.output.resolve()),
                    "measured_iteration_count": diagnostic["measured_iteration_count"],
                    "formal_performance_eligible": False,
                }, sort_keys=True))
                return 0
            variant_watchdogs: dict[str, InactivityWatchdog] = {}

            def cycle_progress(variant, item) -> None:
                watchdog = variant_watchdogs.setdefault(
                    variant.bits,
                    InactivityWatchdog(timeout_seconds=inactivity_timeout_seconds),
                )
                observe_cycle_progress(
                    watchdog,
                    phase=item.phase,
                    completed_events=item.completed_events,
                    completed_iterations=item.completed_iterations,
                    cpu_seconds=time.process_time(),
                )
                print(json.dumps({
                    "variant": variant.bits,
                    "phase": item.phase,
                    "completed_events": item.completed_events,
                    "accepted_events": item.total_events,
                    "completed_iterations": item.completed_iterations,
                    "total_iterations": item.total_iterations,
                    "simulated_cycles": item.simulated_cycles,
                    "elapsed_seconds": item.elapsed_seconds,
                    "watchdog": watchdog.report(status="active"),
                }, sort_keys=True), file=sys.stderr, flush=True)

            def report_progress(run) -> None:
                nonlocal base_cycles
                if run.variant.bits == "0000":
                    base_cycles = run.result.total_cycles
                if base_cycles is None:
                    raise AssertionError("archive Base ASIC result must precede ablation progress")
                print(json.dumps({
                    "variant": run.variant.bits,
                    "cycles": run.result.total_cycles,
                    "comparison_baseline": comparison_baseline(run.variant.bits),
                    "speedup_vs_base_asic": asic_speedup(
                        run.variant.bits, base_cycles=base_cycles,
                        cycles=run.result.total_cycles,
                    ),
                    "speedup_vs_gpu_base": None,
                    "completed": True,
                }, sort_keys=True), file=sys.stderr, flush=True)

            runs = run_archive_matrix(
                args.archive, config,
                max_events=max_events,
                max_frontier_events=max_frontier_events,
                max_atomic_packet_events=max_frontier_events,
                progress_interval_seconds=progress_interval_seconds,
                progress=report_progress,
                cycle_progress=cycle_progress,
                parallel_workers=args.parallel_workers,
            )
            _write_archive_ablation_outputs(
                runs, output=args.output, config=config,
                model=args.model, dataset=args.dataset,
                formal_performance_eligible=(
                    bool(validation_report["formal_performance_eligible"])
                    and not quick_scope
                ),
                validation_report=validation_report,
                archive=args.archive,
            )
            print(json.dumps({"variants": len(runs), "status": "passed"}, sort_keys=True))
            return 0
        if args.command == "trace-captured-packets":
            if args.output.exists() and any(args.output.iterdir()):
                raise ValueError("captured packet output directory must be empty")
            document = json.loads(args.manifest.read_text(encoding="utf-8"))
            if (
                not isinstance(document, dict)
                or document.get("schema_version")
                != CAPTURED_PACKET_SAMPLE_SCHEMA_VERSION
                or not isinstance(document.get("packets"), list)
            ):
                raise ValueError("captured packet manifest is malformed")
            specs = tuple(
                CapturedPacketSpec.from_dict(item)
                for item in document["packets"]
                if isinstance(item, dict)
            )
            if len(specs) != len(document["packets"]):
                raise ValueError("captured packet manifest contains a non-object packet")
            captured_trace = complete_captured_packet_sample(
                specs,
                max_events=args.max_events,
                query_lanes=args.query_lanes,
                initial_gaussian_count=args.initial_gaussian_count,
            )
            validate_trace(captured_trace)
            plan = RelationPacketPlan.from_trace(
                captured_trace, query_lanes=args.query_lanes,
            )
            TraceWriter().write(captured_trace, args.output, validate=False)
            print(json.dumps({
                "status": "passed",
                "events": captured_trace.event_count,
                "dependencies": int(captured_trace.dependencies.size),
                "logical_relation_events": int(sum(
                    packet["relation_count"]
                    for packet in captured_trace.metadata["trace_sample"]["packets"]
                )),
                "physical_relation_packets": plan.relation_packet_count,
                "formal_performance_eligible": False,
            }, sort_keys=True))
            return 0
        trace = TraceReader().read(args.trace, validate=False, mmap_mode="r")
        if args.command == "trace-validate":
            if args.index_directory is not None and args.scan_events is None:
                parser.error("--index-directory requires --scan-events")
            validation_config = (
                TraceValidationConfig(
                    scan_events=args.scan_events,
                    index_directory=args.index_directory,
                )
                if args.scan_events is not None else None
            )
            validate_trace(trace, config=validation_config)
            print(json.dumps({"events": trace.event_count, "status": "passed"}, sort_keys=True))
            return 0
        if args.command == "trace-sample":
            manifest_path = args.trace / "chunk_manifest.json"
            if not manifest_path.is_file():
                manifest_path = args.trace / "metadata.json"
            sample_trace = dependency_closed_query_sample(
                trace, TraceSampleConfig(
                    query_ranges=tuple(args.query_range), max_events=args.max_events,
                    max_dependencies=args.max_dependencies, scan_events=args.scan_events,
                    scan_backend=args.scan_backend,
                ),
                source_identity=sha256_file(manifest_path),
                progress=lambda stage, current, total: print(json.dumps({
                    "stage": stage, "current": current, "total": total,
                }, sort_keys=True), file=sys.stderr, flush=True),
            )
            TraceWriter().write(sample_trace, args.output, validate=False)
            sample_metadata = sample_trace.metadata["trace_sample"]
            print(json.dumps({
                "events": sample_trace.event_count,
                "dependencies": int(sample_trace.dependencies.size),
                "formal_performance_eligible": False,
                "source_identity": sample_metadata["source_identity"],
                "status": "passed",
            }, sort_keys=True))
            return 0
        if args.command == "trace-query-packets":
            if args.output.exists() and any(args.output.iterdir()):
                raise ValueError("query packet output directory must be empty")
            manifest_path = args.trace / "chunk_manifest.json"
            if not manifest_path.is_file():
                manifest_path = args.trace / "metadata.json"
            packet_trace = real_query_packet_sample(
                trace,
                QueryPacketSampleConfig(
                    query_ranges=tuple(args.query_range),
                    scan_events=args.scan_events,
                    scan_backend=args.scan_backend,
                    query_lanes=args.query_lanes,
                    ssim_radius=args.ssim_radius,
                ),
                source_identity=sha256_file(manifest_path),
                progress=lambda stage, current, total: print(json.dumps({
                    "stage": stage, "current": current, "total": total,
                }, sort_keys=True), file=sys.stderr, flush=True),
            )
            plan = RelationPacketPlan.from_trace(
                packet_trace, query_lanes=args.query_lanes,
            )
            TraceWriter().write(packet_trace, args.output, validate=False)
            packet_metadata = packet_trace.metadata["trace_sample"]
            print(json.dumps({
                "events": packet_trace.event_count,
                "dependencies": int(packet_trace.dependencies.size),
                "logical_relation_events": int(sum(
                    int(packet["source_relation_count"])
                    for packet in packet_metadata["packets"]
                )),
                "physical_relation_packets": plan.relation_packet_count,
                "formal_performance_eligible": False,
                "source_identity": packet_metadata["source_identity"],
                "status": "passed",
            }, sort_keys=True))
            return 0
        if args.command == "trace-packetize":
            if args.trace.resolve() == args.output.resolve():
                raise ValueError("packetized output must differ from its source trace")
            if args.output.exists() and any(args.output.iterdir()):
                raise ValueError("packetized output directory must be empty")
            derived = derive_quick_relation_packets(
                trace, tuple(args.query_domain), query_lanes=args.query_lanes,
            )
            validate_packet_derivation(trace, derived)
            plan = RelationPacketPlan.from_trace(
                derived, query_lanes=args.query_lanes,
            )
            TraceWriter().write(derived, args.output, validate=False)
            derivation = derived.metadata["relation_packet_derivation"]
            print(json.dumps({
                "status": "passed",
                "events": derived.event_count,
                "dependencies": int(derived.dependencies.size),
                "logical_relation_events": derivation["logical_relation_events"],
                "physical_relation_packets": plan.relation_packet_count,
                "mean_active_lanes": (
                    derivation["logical_relation_events"] / plan.relation_packet_count
                ),
                "formal_performance_eligible": False,
            }, sort_keys=True))
            return 0
        sample_metadata = trace.metadata.get("trace_sample")
        window_metadata = trace.metadata.get("trace_window")
        if sample_metadata is not None:
            if (
                not isinstance(sample_metadata, dict)
                or sample_metadata.get("formal_performance_eligible") is not False
                or sample_metadata.get("result_scope") != "quick_cycle_validation"
            ):
                raise ValueError("sampled trace metadata is malformed")
            sample_schema = sample_metadata.get("schema_version")
            if sample_schema in (
                *QUERY_PACKET_SAMPLE_SCHEMA_VERSIONS,
                CAPTURED_PACKET_SAMPLE_SCHEMA_VERSION,
            ):
                eligible_policies = sample_metadata.get("eligible_policies")
                if (
                    eligible_policies is None
                    and sample_schema == CAPTURED_PACKET_SAMPLE_SCHEMA_VERSION
                ):
                    # Captured-packet v1 predates the explicit policy field,
                    # but its complete forward/backward expansion has this
                    # fixed scope. New samples always write the declaration.
                    eligible_policies = [
                        "base", "query", "residency", "full",
                        "query_oracle", "residency_oracle",
                        *CANONICAL_VARIANT_POLICIES,
                    ]
                if not isinstance(eligible_policies, list):
                    raise ValueError("sampled trace policy scope is malformed")
                stateful = sample_metadata.get("state_versions_preserved") is True
                if args.command == "cycle-replay" and args.policy not in {
                    str(policy) for policy in eligible_policies
                }:
                    raise ValueError(
                        "query packet sample policy is outside its declared scope"
                    )
                if args.command in {"ablation", "cycle-bounds"} and (
                    sample_schema == CAPTURED_PACKET_SAMPLE_SCHEMA_VERSION
                    or stateful
                ):
                    if args.command == "ablation" and not set(
                        CANONICAL_VARIANT_POLICIES
                    ).issubset({str(policy) for policy in eligible_policies}):
                        raise ValueError(
                            "sample policy scope does not cover the seven canonical evaluations"
                        )
                elif args.command != "cycle-replay":
                    raise ValueError(
                        "only stateful query packet samples support bounds or ablation"
                    )
        if window_metadata is not None and (
            not isinstance(window_metadata, dict)
            or window_metadata.get("schema_version") != "gala-iteration-window-v1"
            or window_metadata.get("result_scope") != "quick_trace_validation"
            or window_metadata.get("formal_performance_eligible") is not False
        ):
            raise ValueError("trace iteration window metadata is malformed")
        quick_scope = sample_metadata is not None or window_metadata is not None
        if quick_scope and not args.quick_validation:
            raise ValueError(
                "partial traces require the explicit --quick-validation scope"
            )
        if args.quick_validation and not quick_scope:
            raise ValueError("--quick-validation requires a sampled or windowed trace")
        gala_config = load_config(_repository_path(args.config, args.repository))
        binding = _load_binding(
            args.ramulator_binding, args.ramulator_build_manifest,
            args.ramulator_config,
        )
        usage = _load_resource_usage(args.resource_usage)
        preflight = run_cycle_preflight(
            gala_config, memory_backend=binding, resource_usage=usage,
            reproduction=("gala-sim " + args.command + " --config " + str(args.config)),
        )
        if preflight.status != "passed":
            write_cycle_preflight(preflight, args.output)
            print(json.dumps(preflight.as_dict(), sort_keys=True))
            return 2
        if binding is None:
            raise ValueError("formal cycle run requires a Ramulator 2 binding")
        config = CycleConfig.from_gala(gala_config, binding, resource_usage=usage)
        if args.command == "cycle-bounds":
            validate_trace(trace)
            report = analyze_cycle_lower_bounds(
                CycleEngine(config, policy="base"),
                trace,
                base_asic_cycles=args.base_cycles,
                targets={
                    "query": args.query_target,
                    "residency": args.residency_target,
                    "full": args.full_target,
                },
            )
            args.output.mkdir(parents=True, exist_ok=True)
            write_json(report.as_dict(), args.output / "cycle_lower_bounds.json")
            write_json({
                "result_scope": (
                    "quick_cycle_validation" if quick_scope else "formal_performance"
                ),
                "formal_performance_eligible": not quick_scope,
                "trace_sample": sample_metadata,
                "trace_window": window_metadata,
            }, args.output / "manifest.json")
            write_json({
                "status": "passed",
                "checks": {
                    "necessary_bounds_only": True,
                    "formal_performance_eligible": not quick_scope,
                },
            }, args.output / "status.json")
            print(json.dumps({
                "status": "passed",
                "reachability": [
                    {
                        "scenario": item.scenario,
                        "lower_bound_cycles": item.lower_bound_cycles,
                        "maximum_possible_speedup_vs_base_asic": (
                            item.maximum_possible_speedup_vs_base_asic
                        ),
                        "status": item.status,
                    }
                    for item in report.reachability
                ],
            }, sort_keys=True))
            return 0
        if args.command == "cycle-replay":
            if args.stop_when_throughput_stable:
                require_empty_diagnostic_output(args.output)
            monitor_config = ThroughputDiagnosticConfig.from_gala(gala_config)
            inactivity_watchdog = InactivityWatchdog(
                timeout_seconds=float(
                    gala_config.value("diagnostic.inactivity_timeout_seconds")
                )
            )
            monitor: ThroughputMonitor | None = None
            if args.throughput_progress or args.stop_when_throughput_stable:
                monitor = ThroughputMonitor(
                    monitor_config,
                    stop_when_stable=args.stop_when_throughput_stable,
                    clock_frequency_hz=config.clock_frequency_hz,
                )

            def cycle_progress(progress) -> None:
                observe_cycle_progress(
                    inactivity_watchdog,
                    phase=progress.phase,
                    completed_events=progress.completed_events,
                    completed_iterations=progress.completed_iterations,
                    cpu_seconds=time.process_time(),
                )
                if monitor is None:
                    print(json.dumps({
                        "phase": progress.phase,
                        "runtime": {
                            "status": "active",
                            "completed_events": progress.completed_events,
                            "total_events": progress.total_events,
                            "completed_iterations": progress.completed_iterations,
                            "total_iterations": progress.total_iterations,
                            "elapsed_seconds": progress.elapsed_seconds,
                        },
                        "watchdog": inactivity_watchdog.report(status="active"),
                    }, sort_keys=True), file=sys.stderr, flush=True)
                    return
                try:
                    report = monitor.observe(progress)
                except ThroughputConverged as converged:
                    runtime = (
                        converged.report["runtime_samples"][-1]
                        if converged.report["runtime_samples"] else {
                            "completed_events": progress.completed_events,
                            "total_events": progress.total_events,
                            "elapsed_seconds": progress.elapsed_seconds,
                            "interval_events_per_second": None,
                        }
                    )
                    print(json.dumps({
                        "phase": progress.phase,
                        "runtime": {"status": "active", **runtime},
                        "throughput": converged.report["samples"][-1],
                        "stability": converged.report["stability"],
                    }, sort_keys=True), file=sys.stderr, flush=True)
                    raise
                runtime = (
                    report["runtime_samples"][-1]
                    if report["runtime_samples"] else {
                        "completed_events": progress.completed_events,
                        "total_events": progress.total_events,
                        "elapsed_seconds": progress.elapsed_seconds,
                        "interval_events_per_second": None,
                    }
                )
                print(json.dumps({
                    "phase": progress.phase,
                    "runtime": {"status": "active", **runtime},
                    "throughput": (
                        report["samples"][-1] if report["samples"] else None
                    ),
                    "stability": report["stability"],
                    "watchdog": inactivity_watchdog.report(status="active"),
                }, sort_keys=True), file=sys.stderr, flush=True)

            try:
                result = CycleEngine(config, policy=args.policy).run(
                    trace,
                    progress=cycle_progress,
                    progress_interval_events=monitor_config.report_interval_events,
                    progress_interval_seconds=monitor_config.report_interval_seconds,
                    collect_compute_telemetry=args.compute_telemetry,
                )
            except InactivityTimeoutError as error:
                writer = RunOutputWriter(args.output)
                watchdog_report = inactivity_watchdog.report(status="terminated")
                write_json({
                    "result_scope": "aborted_inactivity_watchdog",
                    "formal_performance_eligible": False,
                    "termination": "watchdog_inactivity_timeout",
                    "policy": args.policy,
                    "watchdog": watchdog_report,
                }, args.output / "manifest.json")
                writer.write_status(
                    "failed_cycle", reason="watchdog_inactivity_timeout",
                    checks={"error": str(error), "watchdog": watchdog_report},
                )
                print(json.dumps({
                    "status": "failed_cycle",
                    "reason": "watchdog_inactivity_timeout",
                    "watchdog": watchdog_report,
                }, sort_keys=True), file=sys.stderr, flush=True)
                return 2
            except ThroughputConverged as converged:
                diagnostic = {
                    **converged.report,
                    "termination": "stopped_on_stable_throughput",
                    "policy": args.policy,
                }
                write_json(diagnostic, args.output / "throughput.json")
                write_json({
                    "result_scope": "development_throughput_projection",
                    "formal_performance_eligible": False,
                    "termination": diagnostic["termination"],
                }, args.output / "manifest.json")
                write_json({
                    "status": "passed",
                    "checks": {
                        "formal_performance_eligible": False,
                        "termination": diagnostic["termination"],
                    },
                }, args.output / "status.json")
                print(json.dumps({
                    "status": "diagnostic_stopped",
                    "output": str((args.output / "throughput.json").resolve()),
                    "formal_performance_eligible": False,
                }, sort_keys=True))
                return 0
            writer = RunOutputWriter(args.output)
            writer.write_cycles(result)
            if monitor is not None:
                write_json({
                    **monitor.report(),
                    "termination": "complete_trace_replay",
                    "policy": args.policy,
                }, args.output / "throughput.json")
            writer.write_manifest({
                "result_scope": (
                    "quick_cycle_validation" if quick_scope else "formal_performance"
                ),
                "formal_performance_eligible": not quick_scope,
                "trace_sample": sample_metadata,
                "trace_window": window_metadata,
            })
            writer.write_status("passed", checks={
                "formal_performance_eligible": not quick_scope,
            })
            print(json.dumps({"cycles": result.total_cycles, "policy": result.policy}, sort_keys=True))
            return 0
        base_cycles: int | None = None

        def report_progress(run) -> None:
            nonlocal base_cycles
            if run.variant.bits == "0000":
                base_cycles = run.result.total_cycles
            if base_cycles is None:
                raise AssertionError("Base ASIC result must precede ablation progress")
            progress_record: dict[str, object] = {
                "variant": run.variant.bits,
                "cycles": run.result.total_cycles,
                "comparison_baseline": comparison_baseline(run.variant.bits),
                "speedup_vs_base_asic": asic_speedup(
                    run.variant.bits, base_cycles=base_cycles,
                    cycles=run.result.total_cycles,
                ),
                "speedup_vs_gpu_base": None,
                "completed": True,
            }
            print(json.dumps(progress_record, sort_keys=True), file=sys.stderr, flush=True)

        runs = run_matrix(
            trace,
            config,
            progress=report_progress,
            parallel_workers=args.parallel_workers,
        )
        base_cycles = runs[0].result.total_cycles
        model = str(trace.metadata.get("model", "unknown"))
        dataset = str(trace.metadata.get("dataset", "unknown"))
        source_identity = str(
            sample_metadata.get("source_identity", "trace")
            if isinstance(sample_metadata, dict) else "trace"
        )
        run_prefix = f"{model}-{dataset}-{source_identity[:12]}"
        breakdown_root = args.output.parent / f"{args.output.name}.modules"
        breakdown_paths: dict[str, str] = {}
        rows: list[AblationRow] = []
        for run in runs:
            breakdown_path = breakdown_root / f"{run.variant.bits}.json"
            stall_counts: dict[str, int] = {}
            for stall in run.result.stalls:
                stall_counts[stall.module] = stall_counts.get(stall.module, 0) + stall.count
            write_json({
                "schema_version": "gala-ablation-module-breakdown-v1",
                "result_scope": "quick_cycle_validation" if quick_scope else "formal_performance",
                "formal_performance_eligible": not quick_scope,
                "bits": run.variant.bits,
                "run_id": f"{run_prefix}-{run.variant.bits}",
                "policy": run.result.policy,
                "total_cycles": run.result.total_cycles,
                "module_counters": run.result.module_counters,
                "module_busy_cycles": {
                    name: int(counters.get("busy_cycles", 0))
                    for name, counters in run.result.module_counters.items()
                },
                "stall_counts_by_module": stall_counts,
                "event_counts": run.result.event_counts,
            }, breakdown_path)
            breakdown_paths[run.variant.bits] = str(
                breakdown_path.relative_to(args.output.parent)
            )
            rows.append(AblationRow(
                model=model,
                dataset=dataset,
                bits=run.variant.bits,
                cycles=run.result.total_cycles,
                comparison_baseline=comparison_baseline(run.variant.bits),
                speedup_vs_base_asic=asic_speedup(
                    run.variant.bits, base_cycles=base_cycles,
                    cycles=run.result.total_cycles,
                ),
                gpu_base_seconds=None,
                speedup_vs_gpu_base=None,
                local_gpu_seconds=None,
                orin_seconds=None,
                speedup_vs_orin=None,
                psnr_delta_db=None,
                ssim_delta=None,
                lpips_delta=None,
                # The replay configuration is authoritative here.  A trace
                # may carry the hash of the configuration used during capture,
                # which can differ for a valid post-capture ASIC tuning run.
                config_sha256=str(config.config_sha256 or ""),
                status="passed",
                module_breakdown_path=breakdown_paths[run.variant.bits],
                run_id=f"{run_prefix}-{run.variant.bits}",
            ))
        write_ablation_csv(rows, args.output)
        write_json({
            "schema_version": "gala-ablation-manifest-v2",
            "result_scope": "quick_cycle_validation" if quick_scope else "formal_performance",
            "formal_performance_eligible": not quick_scope,
            "trace_sample": sample_metadata,
            "trace_window": window_metadata,
            "hash_validation": "disabled_by_user_request",
            "module_breakdown_directory": str(
                breakdown_root.relative_to(args.output.parent)
            ),
            "module_breakdown_paths": breakdown_paths,
            "full_alias": {
                "policy": "full",
                "variant_bits": "1111",
                "run_id": f"{run_prefix}-1111",
                "cycles": next(
                    run.result.total_cycles for run in runs
                    if run.variant.bits == "1111"
                ),
                "selection_contract_equal": (
                    CycleEngine._selection_for_policy("full")
                    == CycleEngine._selection_for_policy("variant:1111")
                ),
            },
        }, args.output.with_suffix(args.output.suffix + ".manifest.json"))
        print(json.dumps({"variants": len(runs), "status": "passed"}, sort_keys=True))
        return 0
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        print(f"gala-sim: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
