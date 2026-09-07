"""Run an upstream training entrypoint with GALA's memory-safe input hooks."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gala-reference-runner")
    parser.add_argument(
        "--model-id", choices=("r2_gaussian", "fact_gs", "exact_gs"), required=True,
    )
    parser.add_argument("train_script", type=Path)
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.train_script.is_file():
        raise SystemExit(f"training script does not exist: {args.train_script}")
    from .fact_low_memory import (
        configure_cuda_memory_environment,
        install_exact_low_memory_overlay,
        install_fact_low_memory_overlay,
        install_r2_low_memory_overlay,
    )
    configure_cuda_memory_environment()

    if args.model_id == "r2_gaussian":
        overlay = install_r2_low_memory_overlay()
        runtime_overlay = None
    elif args.model_id == "fact_gs":
        overlay = install_fact_low_memory_overlay()
        runtime_overlay = None
    else:
        overlay = install_exact_low_memory_overlay()
        from .exact_runtime import install_exact_runtime_overlay

        runtime_overlay = install_exact_runtime_overlay()
    original_argv = sys.argv
    try:
        sys.argv = [str(args.train_script), *args.train_args]
        runpy.run_path(str(args.train_script), run_name="__main__")
    finally:
        sys.argv = original_argv
        overlay.restore()
        if runtime_overlay is not None:
            runtime_overlay.restore()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
