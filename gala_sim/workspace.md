# workspace.py

## External Interfaces

`WorkspacePaths.discover(repository=None, workspace=None)` locates a clone by
its `pyproject.toml` and `configs/` markers. Workspace precedence is an explicit
argument, `GALA_WORKSPACE`, then `<repository>/workspace`.

`ensure()` creates the standard ignored directories. `reference()` serializes
internal paths as `repo://` or `workspace://`; `resolve_reference()` reverses
those references and rejects parent traversal.

## Internal Helpers

`_repository_from()` searches a path and its parents for repository markers.
`_safe_join()` ensures a portable reference cannot escape its selected root.
