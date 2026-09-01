# freeze_inputs.py

## External Interface

The command creates a local input identity document for any registered model
and dataset pair. It validates the prepared dataset, pinned source checkout,
architecture configuration, interpreter, and official command binding. The
default output is under `workspace/cache/input-freeze/`; it does not download
assets or execute training.

## Internal Helpers

Manifest helpers serialize repository and workspace paths as portable
references while preserving explicit external paths. Source, dataset,
configuration, environment, and command identities are content-addressed.
