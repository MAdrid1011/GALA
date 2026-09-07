"""Execute an official training script with the trace-only hooks installed."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict
import json
import os
from pathlib import Path
import runpy
import sys
from typing import Any

from gala_sim.mechanisms import ONLINE_POLICY_NAMES

from .fact_low_memory import (
    install_exact_low_memory_overlay,
    install_fact_low_memory_overlay,
    install_r2_low_memory_overlay,
)
from .trace_capture import TRACE_HOOK_PROFILES, TraceCaptureComplete, TraceSession


def _install_virtual_capture_eval_guard() -> object:
    """Skip upstream image evaluation that is outside the capture contract.

    R2-Gaussian's training script unconditionally appends iteration 1 to its
    evaluation schedule.  Its evaluator concatenates every train/test image on
    the GPU, which can exceed the remaining device headroom even though the
    selected training iteration itself fits.  A profile hook changes only the
    local ``testing_iterations`` argument at ``training_report`` entry; the
    training, backward, and trace hooks remain untouched.
    """

    previous = sys.getprofile()

    def profile(frame: Any, event: str, _arg: Any) -> None:
        if event != "call" or frame.f_code.co_name != "training_report":
            return
        frame.f_locals["testing_iterations"] = ()
        # CPython keeps function locals in fast slots; synchronize the mapping
        # update so the callee observes the empty schedule immediately.
        ctypes.pythonapi.PyFrame_LocalsToFast(
            ctypes.py_object(frame), ctypes.c_int(1),
        )

    sys.setprofile(profile)
    return previous


def _restore_profile(previous: object) -> None:
    sys.setprofile(previous if callable(previous) else None)


def _configure_cuda_memory_environment() -> None:
    """Use bounded CUDA allocator settings before an upstream import.

    Official model entrypoints import torch lazily.  Setting these defaults at
    the runner boundary therefore also covers direct trace-runner use, where
    the parent adapter cannot inject an environment.  Callers may override
    them explicitly for profiling or a model-specific runtime.
    """

    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True,max_split_size_mb:128",
    )
    # CUDA module loading and cuDNN's plan cache can otherwise consume the
    # small headroom left after model initialization on 12 GiB devices.
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    os.environ.setdefault("TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT", "0")


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
    parser = argparse.ArgumentParser(prog="gala-trace-runner")
    parser.add_argument("--trace-output", type=Path, required=True)
    parser.add_argument(
        "--model-id", choices=tuple(TRACE_HOOK_PROFILES), default="r2_gaussian",
    )
    parser.add_argument("--dataset-id", default="Chest")
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
        "--archive-max-inflight-chunks", type=int, default=None,
        help=(
            "override concurrent compact-archive compression chunks; this "
            "software bound is independent of the online cycle frontier"
        ),
    )
    parser.add_argument(
        "--capture-iteration-range", type=_iteration_range, default=None,
        metavar="START:END",
        help="capture an inclusive validation window while executing all training iterations",
    )
    parser.add_argument(
        "--stop-after-capture-range", action="store_true",
        help="stop official execution after the selected validation window closes",
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
        choices=ONLINE_POLICY_NAMES,
        help="cycle policy used by the bounded online virtual replay",
    )
    parser.add_argument("train_script", type=Path)
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _configure_cuda_memory_environment()
    state_record_bytes = int(os.environ.get("GALA_TRACE_STATE_RECORD_BYTES", "128"))
    # Larger bounded chunks reduce filesystem calls while keeping the capture
    # memory bounded.  The official adapter can override this per run.
    chunk_events = int(os.environ.get("GALA_TRACE_CHUNK_EVENTS", "262144"))
    fact_raster_record_chunk = int(
        os.environ.get("GALA_TRACE_FACT_RASTER_CHUNK", "4096")
    )
    fact_voxel_record_chunk = int(
        os.environ.get("GALA_TRACE_FACT_VOXEL_CHUNK", "4096")
    )
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
    if fact_raster_record_chunk <= 0 or fact_voxel_record_chunk <= 0:
        raise ValueError("Fact-GS record chunk sizes must be positive")
    if inactivity_timeout_seconds <= 0:
        raise ValueError("GALA_TRACE_INACTIVITY_TIMEOUT_SECONDS must be positive")
    if progress_interval_seconds <= 0:
        raise ValueError("GALA_TRACE_PROGRESS_INTERVAL_SECONDS must be positive")
    if args.virtual_capture and args.stream_only:
        raise ValueError("--virtual-capture and --stream-only are mutually exclusive")
    if args.stop_after_capture_range and args.capture_iteration_range is None:
        raise ValueError("--stop-after-capture-range requires --capture-iteration-range")
    if args.packet_archive_root is not None and not args.virtual_capture:
        raise ValueError("--packet-archive-root requires --virtual-capture")
    if args.capture_config is not None and not args.virtual_capture:
        raise ValueError("--capture-config requires --virtual-capture")
    if (
        args.archive_max_inflight_chunks is not None
        and args.packet_archive_root is None
    ):
        raise ValueError(
            "--archive-max-inflight-chunks requires --packet-archive-root"
        )
    if (
        args.archive_max_inflight_chunks is not None
        and args.archive_max_inflight_chunks <= 0
    ):
        raise ValueError("--archive-max-inflight-chunks must be positive")
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
                    max_atomic_packet_events=max_frontier_events,
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
    archive_max_inflight_chunks = 1
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
        configured_archive_chunks = int(
            capture_config.value("trace.max_inflight_chunks")
        )
        archive_max_inflight_chunks = (
            configured_archive_chunks
            if args.archive_max_inflight_chunks is None
            else int(args.archive_max_inflight_chunks)
        )
        if archive_chunk_bytes <= 0 or archive_max_inflight_chunks <= 0:
            raise ValueError("trace archive chunk capacities must be positive")
    session = TraceSession(
        args.trace_output, model_id=args.model_id, dataset_name=args.dataset_id,
        state_record_bytes=state_record_bytes,
        chunk_events=chunk_events, stream_only=args.stream_only,
        fact_raster_record_chunk=fact_raster_record_chunk,
        fact_voxel_record_chunk=fact_voxel_record_chunk,
        capture_iteration_range=args.capture_iteration_range,
        stop_after_capture_range=args.stop_after_capture_range,
        virtual_capture=args.virtual_capture,
        virtual_packet_consumer_factory=consumer_factory,
        virtual_packet_archive_root=args.packet_archive_root,
        virtual_packet_archive_chunk_bytes=archive_chunk_bytes,
        virtual_packet_archive_max_inflight_chunks=archive_max_inflight_chunks,
        inactivity_timeout_seconds=inactivity_timeout_seconds,
        progress_interval_seconds=progress_interval_seconds,
    )
    low_memory_overlay = None
    exact_runtime_overlay = None
    previous_profile = None
    original_argv = sys.argv
    completed = False
    installed = False
    try:
        session.install()
        installed = True
        if args.model_id == "r2_gaussian":
            low_memory_overlay = install_r2_low_memory_overlay()
        elif args.model_id == "fact_gs":
            # Both upstream loaders eagerly upload all projections.  Keep the
            # selected camera path while materializing only its image.
            low_memory_overlay = install_fact_low_memory_overlay()
        elif args.model_id == "exact_gs":
            from .exact_runtime import install_exact_runtime_overlay

            exact_runtime_overlay = install_exact_runtime_overlay()
            low_memory_overlay = install_exact_low_memory_overlay()
        # R² unconditionally calls training_report at iteration one.  That
        # routine concatenates every train/test projection on CUDA and can
        # exceed the capture process's remaining headroom.  Evaluation is
        # outside the trace contract, so skip it for every R² trace capture,
        # including stream-only captures (not only virtual archives).
        if args.model_id == "r2_gaussian" and (args.virtual_capture or args.stream_only):
            previous_profile = _install_virtual_capture_eval_guard()
        sys.argv = [str(args.train_script), *args.train_args]
        try:
            runpy.run_path(str(args.train_script), run_name="__main__")
        except TraceCaptureComplete:
            pass
        completed = True
    finally:
        if previous_profile is not None:
            _restore_profile(previous_profile)
        if low_memory_overlay is not None:
            low_memory_overlay.restore()
        if exact_runtime_overlay is not None:
            exact_runtime_overlay.restore()
        if installed:
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
