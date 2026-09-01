# manifest.py

## External Interfaces

`source_record()` validates a pinned upstream checkout and license identity.
`dataset_record()` inventories prepared dataset files, geometry, and reference
volume. `training_record()` binds and validates an upstream training profile.

`environment_snapshot()` records interpreter, compiler, CUDA, and dependency
versions. `build_freeze_record()` creates the input identity document, and
`verify_freeze_record()` validates its schema and required audit fields.
`build_campaign_freeze_record()` provides the model- and dataset-independent
public workflow and emits portable repository and workspace references.

## Internal Helpers

AST helpers extract declared upstream defaults, schedules, and random-state
behavior without importing the training package. Quality checks validate the
reference range and configured metric inputs.
