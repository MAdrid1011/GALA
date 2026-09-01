# Contributing

## Development Setup

Create a Python 3.10 or newer environment and install the project in editable
mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[quality,parquet,assets]'
```

Run `pytest -q` before submitting a change.

## Repository Rules

- Keep source, tests, configuration, and documentation in English.
- Do not commit datasets, model checkouts, traces, profiles, checkpoints,
  generated results, environment snapshots, or workspace contents.
- Do not add machine-specific absolute paths to tracked files.
- Keep model and dataset provenance in declarative manifests.
- Do not describe an independent implementation as upstream author code.
- Update interface documentation and tests with behavioral changes.

## Adding an Integration

A model integration provides a model manifest, a registered `ModelAdapter`,
and tests for preparation and command construction. A dataset integration
provides a source manifest, a registered `DatasetAdapter`, validation rules,
and a small parser fixture.

Large source repositories and datasets must be acquired through the asset
catalog and stored in `workspace/`. Generated performance results are not
checked into this repository.

## Licensing

Contributions to this repository are accepted under Apache-2.0. Verify that
third-party source and data licenses permit the intended use, and retain their
attribution and notices separately from this project's license.
