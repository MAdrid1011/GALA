# registry.py

## External Interfaces

`ModelDescriptor` records a model identifier, provenance type, immutable source
identity, and entry point. `model_descriptors()` loads these values from the
asset catalog.

`get_model_adapter()` returns a workspace-bound official command adapter or the
independent GR-Gaussian adapter. `CommandModelAdapter` prepares and executes a
pinned author command without relying on the process working directory.
`prepare_campaign()` converts and caches a dataset through its registered
adapter, then binds the selected model's reference command.

## Internal Helpers

Reconstruction discovery recognizes the standard volume filenames used by the
integrated projects. Trace capture remains an explicit adapter-hook boundary;
it never fabricates events from a completed output.
