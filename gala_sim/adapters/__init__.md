# adapters/__init__.py

## External Interfaces

The package exports model execution artifacts and protocols, concrete
R2-Gaussian integration, dataset manifests and geometry types, and the public
model and dataset registry functions.

Dataset callers use `get_dataset_adapter()` and `dataset_descriptors()`.
Model callers use `get_model_adapter()` and `model_descriptors()`.
Campaign callers use `prepare_campaign()` to compose registered model and
dataset identifiers through the prepared-data cache.

## Internal Helpers

Model-specific execution, tracing, and profiling live in separate adapter
modules. Dataset parsing and normalization live in `datasets.py`; the legacy
Chest structures remain exported for compatibility with existing callers.
