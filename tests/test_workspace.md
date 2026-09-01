# Workspace Tests

The tests verify repository discovery from nested paths, workspace precedence,
managed-directory creation, and portable `repo://` and `workspace://` path
references. All filesystem state is isolated under pytest temporary paths.
