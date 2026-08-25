"""Machine-readable simulator command line."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

from gala_sim.ablation import run_matrix
from gala_sim.config import load_config, pending_parameters
from gala_sim.results import AblationRow, write_ablation_csv
from gala_sim.results.run import RunOutputWriter
from gala_sim.timing import CycleConfig, CycleEngine
from gala_sim.timing.memory import NativeRamulator2Binding, Ramulator2Backend
from gala_sim.timing.resources import ResourceUsage
from gala_sim.tools.cycle_preflight import run_cycle_preflight, write_cycle_preflight
from gala_sim.trace import TraceReader, validate_trace


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
    trace = commands.add_parser("trace-validate")
    trace.add_argument("--trace", type=Path, required=True)
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
    return parser


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
        trace = TraceReader().read(args.trace, validate=False, mmap_mode="r")
        if args.command == "trace-validate":
            validate_trace(trace)
            print(json.dumps({"events": trace.event_count, "status": "passed"}, sort_keys=True))
            return 0
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
