"""Machine-readable simulator command line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from gala_sim.ablation import run_matrix
from gala_sim.config import load_config
from gala_sim.results import AblationRow, write_ablation_csv
from gala_sim.results.run import RunOutputWriter
from gala_sim.timing import CycleConfig, CycleEngine, ModuleTiming
from gala_sim.timing.memory import RecordedMemoryBackend
from gala_sim.trace import TraceReader, validate_trace


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gala-sim")
    commands = parser.add_subparsers(dest="command", required=True)
    config = commands.add_parser("config-check")
    config.add_argument("--config", type=Path, required=True)
    trace = commands.add_parser("trace-validate")
    trace.add_argument("--trace", type=Path, required=True)
    replay = commands.add_parser("cycle-replay")
    replay.add_argument("--trace", type=Path, required=True)
    replay.add_argument("--timing", type=Path, required=True)
    replay.add_argument("--policy", default="base")
    replay.add_argument("--output", type=Path, required=True)
    ablation = commands.add_parser("ablation")
    ablation.add_argument("--trace", type=Path, required=True)
    ablation.add_argument("--timing", type=Path, required=True)
    ablation.add_argument("--output", type=Path, required=True)
    return parser


def _load_timing(path: Path) -> CycleConfig:
    document = json.loads(path.read_text(encoding="utf-8"))
    modules = {
        name: ModuleTiming(**values)
        for name, values in document["modules"].items()
    }
    completions = {
        (int(item["address"]), int(item["size_bytes"]), bool(item["is_write"]), int(item["arrival_cycle"])):
        int(item["completion_cycle"])
        for item in document["memory_completions"]
    }
    return CycleConfig(
        modules=modules,
        memory=RecordedMemoryBackend(completions),
        clock_frequency_hz=int(document["clock_frequency_hz"]),
        relation_seed_fifo_entries=int(document["relation_seed_fifo_entries"]),
        candidate_lanes=int(document["candidate_lanes"]),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "config-check":
            config = load_config(args.config)
            print(json.dumps({"sha256": config.sha256, "ready": config.ready,
                              "pending": [] if config.ready else "configuration_pending"},
                             sort_keys=True))
            return 0 if config.ready else 2
        trace = TraceReader().read(args.trace, mmap_mode="r")
        validate_trace(trace)
        if args.command == "trace-validate":
            print(json.dumps({"events": trace.event_count, "status": "passed"}, sort_keys=True))
            return 0
        config = _load_timing(args.timing)
        if args.command == "cycle-replay":
            result = CycleEngine(config, policy=args.policy).run(trace)
            writer = RunOutputWriter(args.output)
            writer.write_cycles(result)
            writer.write_status("passed")
            print(json.dumps({"cycles": result.total_cycles, "policy": result.policy}, sort_keys=True))
            return 0
        runs = run_matrix(trace, config)
        rows = [AblationRow(
            model=str(trace.metadata.get("model", "unknown")),
            dataset=str(trace.metadata.get("dataset", "unknown")), bits=run.variant.bits,
            cycles=run.result.total_cycles,
            speedup_vs_base_asic=runs[0].result.total_cycles / run.result.total_cycles,
            local_gpu_seconds=None, orin_seconds=None, speedup_vs_orin=None,
            psnr_delta_db=None, ssim_delta=None, lpips_delta=None,
            config_sha256=str(trace.metadata.get("config_sha256", "")), status="passed",
        ) for run in runs]
        write_ablation_csv(rows, args.output)
        print(json.dumps({"variants": len(runs), "status": "passed"}, sort_keys=True))
        return 0
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        print(f"gala-sim: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
