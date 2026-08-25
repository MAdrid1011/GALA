"""Execute an official training script with the trace-only hooks installed."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys

from .trace_capture import TraceSession


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gala-r2-trace-runner")
    parser.add_argument("--trace-output", type=Path, required=True)
    parser.add_argument("train_script", type=Path)
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    state_record_bytes = int(os.environ.get("GALA_TRACE_STATE_RECORD_BYTES", "128"))
    chunk_events = int(os.environ.get("GALA_TRACE_CHUNK_EVENTS", "65536"))
    if state_record_bytes <= 0:
        raise ValueError("GALA_TRACE_STATE_RECORD_BYTES must be positive")
    if chunk_events <= 0:
        raise ValueError("GALA_TRACE_CHUNK_EVENTS must be positive")
    session = TraceSession(
        args.trace_output, state_record_bytes=state_record_bytes,
        chunk_events=chunk_events,
    )
    session.install()
    original_argv = sys.argv
    try:
        sys.argv = [str(args.train_script), *args.train_args]
        runpy.run_path(str(args.train_script), run_name="__main__")
    finally:
        session.restore()
        session.finish()
        sys.argv = original_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
