"""Execute an official training script with the trace-only hooks installed."""

from __future__ import annotations

import argparse
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
        "--capture-iteration-range", type=_iteration_range, default=None,
        metavar="START:END",
        help="capture an inclusive validation window while executing all training iterations",
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
    session = TraceSession(
        args.trace_output, state_record_bytes=state_record_bytes,
        chunk_events=chunk_events, stream_only=args.stream_only,
        capture_iteration_range=args.capture_iteration_range,
        virtual_capture=args.virtual_capture,
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
        finally:
            sys.argv = original_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
