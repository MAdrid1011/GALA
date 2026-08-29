"""Machine-readable simulator command line."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import shlex
import sys
import time

from gala_sim.ablation import run_matrix
from gala_sim.config import load_config, pending_parameters
from gala_sim.results import AblationRow, write_ablation_csv
from gala_sim.results.run import RunOutputWriter
from gala_sim.results.manifest import write_json
from gala_sim.identity import sha256_file
from gala_sim.timing import (
    CycleConfig, CycleEngine, RelationPacketPlan, analyze_cycle_lower_bounds,
)
from gala_sim.timing.memory import NativeRamulator2Binding, Ramulator2Backend
from gala_sim.timing.resources import ResourceUsage
from gala_sim.tools.cycle_preflight import run_cycle_preflight, write_cycle_preflight
from gala_sim.tools.cycle_throughput import (
    ThroughputConverged, ThroughputDiagnosticConfig, ThroughputMonitor,
    require_empty_diagnostic_output,
)
from gala_sim.tools.inactivity import (
    InactivityTimeoutError, InactivityWatchdog, observe_cycle_progress,
)
from gala_sim.tools.preflight import run_native_preflight
from gala_sim.adapters.native_reference import run_native_reference
from gala_sim.trace import (
    CAPTURED_PACKET_SAMPLE_SCHEMA_VERSION, CapturedPacketSpec, QueryDomain,
    QUERY_PACKET_SAMPLE_SCHEMA_VERSION, QueryPacketSampleConfig, QueryRange,
    TraceReader, TraceSampleConfig, TraceValidationConfig, TraceWriter,
    complete_captured_packet_sample, dependency_closed_query_sample,
    derive_quick_relation_packets, real_query_packet_sample,
    validate_packet_derivation, validate_trace,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gala-sim")
    commands = parser.add_subparsers(dest="command", required=True)
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
    replay.add_argument("--policy", default="base")
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
    return parser


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


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "config-check":
            config = load_config(args.config)
            print(json.dumps({"sha256": config.sha256, "ready": config.ready,
                              "pending": pending_parameters(config.parameters)},
                             sort_keys=True))
            return 0 if config.ready else 2
        if args.command == "cycle-preflight":
            config = load_config(args.config)
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
        if args.command == "native-preflight":
            config = load_config(args.config)
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
            config = load_config(args.config)
            freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
            preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
            result = run_native_reference(config, freeze, preflight, args.output)
            print(json.dumps(result, sort_keys=True))
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
            if sample_metadata.get("schema_version") == QUERY_PACKET_SAMPLE_SCHEMA_VERSION:
                if args.command != "cycle-replay":
                    raise ValueError(
                        "query packet samples only support query-scheduler cycle replay"
                    )
                eligible_policies = sample_metadata.get("eligible_policies")
                if not isinstance(eligible_policies, list) or args.policy not in {
                    str(policy) for policy in eligible_policies
                }:
                    raise ValueError(
                        "query packet sample policy is outside its declared scope"
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
        gala_config = load_config(args.config)
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
                "speedup_vs_base_asic": base_cycles / run.result.total_cycles,
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
        rows = [AblationRow(
            model=str(trace.metadata.get("model", "unknown")),
            dataset=str(trace.metadata.get("dataset", "unknown")), bits=run.variant.bits,
            cycles=run.result.total_cycles,
            speedup_vs_base_asic=runs[0].result.total_cycles / run.result.total_cycles,
            local_gpu_seconds=None, orin_seconds=None, speedup_vs_orin=None,
            psnr_delta_db=None, ssim_delta=None, lpips_delta=None,
            config_sha256=str(trace.metadata.get("config_sha256", "")),
            status="passed",
        ) for run in runs]
        write_ablation_csv(rows, args.output)
        if quick_scope:
            write_json({
                "result_scope": "quick_cycle_validation",
                "formal_performance_eligible": False,
                "trace_sample": sample_metadata,
                "trace_window": window_metadata,
            }, args.output.with_suffix(args.output.suffix + ".manifest.json"))
        print(json.dumps({"variants": len(runs), "status": "passed"}, sort_keys=True))
        return 0
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        print(f"gala-sim: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
