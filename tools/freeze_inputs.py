#!/usr/bin/env python3
"""Create an auditable input-freeze record for a catalogued campaign.

This command only freezes identities.  It does not download data, train a
model, or infer missing hardware parameters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Running a tool by path puts ``tools/`` first on ``sys.path``.  Add the
# repository root explicitly so the command works without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gala_sim.config import ConfigError
from gala_sim.manifest import build_campaign_freeze_record, write_freeze_record
from gala_sim.workspace import WorkspacePaths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--config", default="configs/architecture/gala.yaml", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--model-output", type=Path)
    parser.add_argument("--python-executable", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        paths = WorkspacePaths.discover(args.repository, args.workspace)
        output = args.output or paths.cache / "input-freeze" / f"{args.model}-{args.dataset}.json"
        record = build_campaign_freeze_record(
            workspace=paths, model_id=args.model, dataset_id=args.dataset,
            config_path=args.config, source_root=args.source_root,
            dataset_root=args.dataset_root, output_root=args.model_output,
            python_executable=args.python_executable,
        )
        write_freeze_record(record, output)
    except (ConfigError, OSError, ValueError) as error:
        print(f"freeze_inputs: {error}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
