# GALA Simulator

GALA Simulator is an architecture research framework for evaluating GALA, a
hardware/software co-design for Gaussian-based tomographic reconstruction. It
combines a numerical reference path, trace capture, an event-driven cycle
model, and compiler/architecture ablations behind reproducible interfaces.

The repository contains source code, configuration, tests, and design
documentation. Model repositories, datasets, generated traces, checkpoints,
profiles, and result artifacts are stored under an ignored local workspace.

## Features

- Numerical reconstruction and quality validation interfaces
- Dynamic event and dependency trace capture
- Physical relation packet construction and validation
- Query scheduling, semantic worksets, and residency modeling
- Banked SRAM, bounded queues, backpressure, and memory timing
- Native Ramulator 2 integration
- Composable model and dataset adapters
- Compiler and architecture ablation generation

## Requirements

- Python 3.10 or newer
- A C++17 compiler for the optional Ramulator 2 bridge
- CUDA for model execution and GPU profiling workflows
- Ramulator 2 v2.1.0 for native LPDDR5 timing

Create an environment and install the simulator:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Optional dependencies are grouped by use:

```bash
python -m pip install -e '.[quality,parquet,assets]'
```

## Quick Start

Validate the repository configuration and local workspace:

```bash
gala-sim config-check --config configs/architecture/gala.yaml
gala-sim workspace-check
pytest -q
```

Inspect the available commands:

```bash
gala-sim --help
gala-sim cycle-replay --help
gala-sim archive-ablation --help
```

## Local Workspace

All non-versioned assets live under `workspace/` by default:

```text
workspace/
  upstream/     pinned model and third-party source checkouts
  datasets/     downloaded and prepared datasets
  build/        native extensions and external builds
  cache/        resumable downloads and converted data
  traces/       captured traces and packet archives
  profiles/     profiler output
  results/      generated simulator and quality results
  archive/      retained local artifacts
```

The workspace location is selected in this order:

1. The command-line `--workspace` option
2. The `GALA_WORKSPACE` environment variable
3. The `workspace/` directory in the repository root

The complete workspace is ignored by Git and excluded from package builds.
See [Workspace and Assets](docs/WORKSPACE_AND_ASSETS.md) for portability and
data provenance rules.

An installed wheel can be invoked outside the clone by identifying the
configuration root explicitly:

```bash
gala-sim --repository /path/to/GALA config-check \
  --config configs/architecture/gala.yaml
```

## Acquire Models and Datasets

Preview or acquire the catalogued public assets:

```bash
gala-sim acquire --all --dry-run
gala-sim acquire --models r2_gaussian,fact_gs,exact_gs
gala-sim acquire --datasets chest,walnut,hdtomo_usb
```

Acquisition is manifest-driven. Downloads are resumable, publisher checksums
are verified when supplied, and a local SHA-256 identity is written under the
workspace. Dataset licenses remain those of their publishers and are not
changed by this repository's license.

## Supported Integrations

Model adapters expose a shared preparation, reference execution, trace capture,
and reduction replay contract:

- R2-Gaussian, using its pinned author repository
- FaCT-GS, using its pinned author repository
- Exact-GS, using its pinned author repository
- GR-Gaussian, as an independent implementation of the published method

Dataset adapters normalize projections, scanner geometry, view partitions,
and reference volumes for Chest, FIPS Walnut, and HDTomo-USB. Adapters and
datasets are selected independently through catalog identifiers. Generated
campaigns carry the exact model, data, and configuration identity.

## Reproduction Workflow

A complete workflow uses the following order:

1. Acquire and validate model and dataset assets.
2. Freeze source, data, environment, training, and configuration identities.
3. Execute the model reference path and quality checks.
4. Capture and validate event traces.
5. Run relation-capacity and cycle preflight checks.
6. Replay the cycle model and produce ablation outputs.

Example cycle replay:

```bash
gala-sim cycle-replay \
  --trace workspace/traces/example \
  --config configs/architecture/gala.yaml \
  --ramulator-build-manifest workspace/build/ramulator2-build.json \
  --ramulator-config configs/memory/ramulator2-lpddr5-6400-external.yaml \
  --resource-usage configs/architecture/gala-resource-usage.json \
  --policy variant:1111 \
  --output workspace/results/example-full
```

Every generated result identifies its inputs, configuration, memory backend,
resource snapshot, event coverage, and comparison baseline. The repository
does not ship generated performance tables or local run histories.

## Ramulator 2 Bridge

Build the bridge from a verified Ramulator 2 checkout:

```bash
python tools/build_ramulator2_bridge.py \
  --ramulator-source workspace/upstream/ramulator2 \
  --ramulator-library workspace/build/ramulator2/libramulator.so \
  --output workspace/build/libgala_ramulator2_bridge.so \
  --manifest workspace/build/ramulator2-build.json
```

The native bridge is optional for unit tests but required for native memory
timing runs.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `gala_sim/` | Simulator, adapter, trace, timing, and output packages |
| `configs/` | Architecture, model, dataset, memory, and profiling manifests |
| `native/` | Native Ramulator 2 bridge source |
| `tools/` | Repository utilities |
| `tests/` | Unit and integration tests |
| `docs/` | Design contracts and reproduction documentation |

Start with the [documentation index](docs/README.md).

## Development

Run `pytest -q` before submitting a change. New integrations should implement
the documented model or dataset protocol, add manifest-driven fixtures, and
avoid embedding local paths or generated artifacts. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

The GALA Simulator source is licensed under the Apache License 2.0. See
[LICENSE](LICENSE). Upstream models and datasets retain their own licenses.
