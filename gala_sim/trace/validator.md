# validator.py

## External Interfaces

`TraceValidationConfig` selects scan chunk size and an optional temporary index
directory. `validate_trace()` returns `TraceValidationReport` or raises
`TraceValidationError` for structural, identity, dependency, lineage, version,
or lifecycle violations.

## Internal Helpers

Vectorized structural passes check ranges and typed domains. The lifecycle
state machine reconstructs active Gaussians, cache fills, consumers, gradients,
updates, clone/split/prune lineage, and cross-iteration barriers without
assuming one legal backward completion order.
