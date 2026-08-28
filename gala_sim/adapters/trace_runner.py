"""Execute an official training script with the trace-only hooks installed."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import runpy
import sys

from .trace_capture import TraceSession


def _iteration_range(value: str) -> tuple[int, int]:
    start_text, separator, end_text = value.partition(":")
    if not separator:
        raise argparse.ArgumentTypeError("iteration range must use START:END")
    try:
        start, end = int(start_text), int(end_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("iteration range bounds must be integers") from error
    if start <= 0 or end < start:
        raise argparse.ArgumentTypeError(
            "iteration range must have 1 <= START <= END"
        )
    return start, end


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gala-r2-trace-runner")
    parser.add_argument("--trace-output", type=Path, required=True)
    parser.add_argument(
        "--stream-only", action="store_true",
        help="keep validated capture columns in bounded chunks without final mmap merge",
    )
    parser.add_argument(
        "--virtual-capture", action="store_true",
        help="capture exact bounded CUDA work-buffer packets without raw event columns",
    )
    parser.add_argument(
        "--virtual-capture-audit-only", action="store_true",
        help="capture and validate virtual packet ledgers without online cycle replay",
    )
    parser.add_argument(
        "--packet-archive-root", type=Path, default=None,
        help="persist compact virtual packets for independent later replays",
    )
    parser.add_argument(
        "--capture-config", type=Path, default=None,
        help="Gala config providing compact capture software parameters",
    )
    parser.add_argument(
        "--capture-iteration-range", type=_iteration_range, default=None,
        metavar="START:END",
        help="capture an inclusive validation window while executing all training iterations",
    )
    parser.add_argument(
        "--online-cycle-config", type=Path, default=None,
        help="attach bounded virtual capture to an online cycle replay using this Gala config",
    )
    parser.add_argument(
        "--online-ramulator-build-manifest", type=Path, default=None,
        help="Ramulator 2 bridge build manifest for online virtual replay",
    )
    parser.add_argument(
        "--online-ramulator-config", type=Path, default=None,
        help="Ramulator 2 configuration for online virtual replay",
    )
    parser.add_argument(
        "--online-cycle-policy", default="base",
        choices=("base", "query_oracle", "residency_oracle", "full"),
        help="cycle policy used by the bounded online virtual replay",
    )
    parser.add_argument("train_script", type=Path)
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    state_record_bytes = int(os.environ.get("GALA_TRACE_STATE_RECORD_BYTES", "128"))
    chunk_events = int(os.environ.get("GALA_TRACE_CHUNK_EVENTS", "65536"))
    inactivity_timeout_seconds = float(
        os.environ.get("GALA_TRACE_INACTIVITY_TIMEOUT_SECONDS", "300")
    )
    progress_interval_seconds = float(
        os.environ.get("GALA_TRACE_PROGRESS_INTERVAL_SECONDS", "30")
    )
    if state_record_bytes <= 0:
        raise ValueError("GALA_TRACE_STATE_RECORD_BYTES must be positive")
    if chunk_events <= 0:
        raise ValueError("GALA_TRACE_CHUNK_EVENTS must be positive")
    if inactivity_timeout_seconds <= 0:
        raise ValueError("GALA_TRACE_INACTIVITY_TIMEOUT_SECONDS must be positive")
    if progress_interval_seconds <= 0:
        raise ValueError("GALA_TRACE_PROGRESS_INTERVAL_SECONDS must be positive")
    if args.virtual_capture and args.stream_only:
        raise ValueError("--virtual-capture and --stream-only are mutually exclusive")
    if args.packet_archive_root is not None and not args.virtual_capture:
        raise ValueError("--packet-archive-root requires --virtual-capture")
    if args.capture_config is not None and not args.virtual_capture:
        raise ValueError("--capture-config requires --virtual-capture")
    online_options = (
        args.online_cycle_config,
        args.online_ramulator_build_manifest,
        args.online_ramulator_config,
    )
    if args.virtual_capture_audit_only and not args.virtual_capture:
        raise ValueError("--virtual-capture-audit-only requires --virtual-capture")
    if not args.virtual_capture and any(value is not None for value in online_options):
        raise ValueError("online cycle replay requires --virtual-capture")
    if args.virtual_capture_audit_only and any(value is not None for value in online_options):
        raise ValueError("audit-only virtual capture cannot enable online cycle replay")
    if args.virtual_capture and not args.virtual_capture_audit_only:
        if any(value is None for value in online_options):
            raise ValueError(
                "online cycle replay requires cycle config, Ramulator build manifest, and Ramulator config"
            )
        from gala_sim.timing import BufferedVirtualCycleConsumer, CycleConfig, CycleEngine
        from gala_sim.timing.memory import NativeRamulator2Binding, Ramulator2Backend
        from gala_sim.config import load_config

        online_config = load_config(args.online_cycle_config)
        online_config.require_ready()
        configured_chunk_events = int(online_config.value("trace.chunk_events"))
        max_inflight_chunks = int(online_config.value("trace.max_inflight_chunks"))
        if configured_chunk_events <= 0 or max_inflight_chunks <= 0:
            raise ValueError("online trace frontier parameters must be positive")
        max_frontier_events = configured_chunk_events * max_inflight_chunks
        for required_path in (
            args.online_ramulator_build_manifest,
            args.online_ramulator_config,
        ):
            if not required_path.is_file():
                raise ValueError(f"online cycle input does not exist: {required_path}")
        online_sinks: list[BufferedVirtualCycleConsumer] = []

        def online_progress(progress) -> None:
            elapsed = max(progress.elapsed_seconds, sys.float_info.epsilon)
            print(json.dumps({
                "phase": progress.phase,
                "completed_events": progress.completed_events,
                "accepted_events": progress.total_events,
                "simulated_cycles": progress.simulated_cycles,
                "elapsed_seconds": progress.elapsed_seconds,
                "completed_events_per_second": progress.completed_events / elapsed,
            }, ensure_ascii=True, sort_keys=True), file=sys.stderr, flush=True)

        def consumer_factory(initial_gaussian_count: int) -> BufferedVirtualCycleConsumer:
            binding = NativeRamulator2Binding.from_build_manifest(
                args.online_ramulator_build_manifest, args.online_ramulator_config,
            )
            cycle_config = CycleConfig.from_gala(
                online_config, Ramulator2Backend(binding),
            )
            sink = BufferedVirtualCycleConsumer(
                CycleEngine(cycle_config, policy=args.online_cycle_policy).online_session(
                    max_events=chunk_events,
                    max_frontier_events=max_frontier_events,
                    initial_gaussian_count=initial_gaussian_count,
                    progress=online_progress,
                    progress_interval_seconds=progress_interval_seconds,
                )
            )
            online_sinks.append(sink)
            return sink
    else:
        consumer_factory = None
        online_sinks = []
    archive_chunk_bytes = None
    if args.packet_archive_root is not None:
        config_path = args.capture_config or args.online_cycle_config
        if config_path is None:
            raise ValueError(
                "--packet-archive-root requires --capture-config or --online-cycle-config"
            )
        from gala_sim.config import load_config

        capture_config = load_config(config_path)
        capture_config.require_ready()
        archive_chunk_bytes = int(capture_config.value("trace.archive_chunk_bytes"))
        if archive_chunk_bytes <= 0:
            raise ValueError("trace.archive_chunk_bytes must be positive")
    session = TraceSession(
        args.trace_output, state_record_bytes=state_record_bytes,
        chunk_events=chunk_events, stream_only=args.stream_only,
        capture_iteration_range=args.capture_iteration_range,
        virtual_capture=args.virtual_capture,
        virtual_packet_consumer_factory=consumer_factory,
        virtual_packet_archive_root=args.packet_archive_root,
        virtual_packet_archive_chunk_bytes=archive_chunk_bytes,
        inactivity_timeout_seconds=inactivity_timeout_seconds,
        progress_interval_seconds=progress_interval_seconds,
    )
    session.install()
    original_argv = sys.argv
    completed = False
    try:
        sys.argv = [str(args.train_script), *args.train_args]
        runpy.run_path(str(args.train_script), run_name="__main__")
        completed = True
    finally:
        session.restore()
        try:
            if completed:
                session.finish()
                if online_sinks and online_sinks[0].result is not None:
                    online_result = {
                        "schema_version": "gala-online-virtual-cycle-v1",
                        "result_scope": "online_virtual_cycle_replay",
                        "formal_performance_eligible": False,
                        "limits": {
                            "packet_event_batch": chunk_events,
                            "configured_trace_chunk_events": configured_chunk_events,
                            "max_inflight_chunks": max_inflight_chunks,
                            "max_frontier_events": max_frontier_events,
                            "inactivity_timeout_seconds": inactivity_timeout_seconds,
                            "progress_interval_seconds": progress_interval_seconds,
                        },
                        "frontier": {
                            "peak_frontier_events": online_sinks[0].session.peak_frontier_events,
                            "resident_completion_markers": (
                                online_sinks[0].session.resident_completion_markers
                            ),
                            "quiescent": online_sinks[0].session.quiescent,
                        },
                        "progress": {
                            "accepted_events": online_sinks[0].session.accepted_event_count,
                            "completed_events": online_sinks[0].session.completed_event_count,
                            "simulated_cycles": online_sinks[0].session.simulated_cycles,
                            "source_packets": online_sinks[0].session.source_packet_count,
                            "query_packets": online_sinks[0].session.query_packet_count,
                            "closed_iterations": online_sinks[0].session.closed_iteration_count,
                        },
                        "semantic_workset": {
                            "exact": True,
                            "retained_keys_at_finish": len(
                                online_sinks[0].session.semantic_workset_totals
                            ),
                        },
                        "cycle": asdict(online_sinks[0].result),
                    }
                    args.trace_output.mkdir(parents=True, exist_ok=True)
                    (args.trace_output / "online_cycle_result.json").write_text(
                        json.dumps(online_result, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
                        encoding="utf-8",
                    )
        finally:
            sys.argv = original_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
