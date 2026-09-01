# migrate_workspace.py

## External Interface

The command atomically renames a legacy runtime directory to the repository's
ignored `workspace/` directory. It refuses to run across filesystems, when the
destination exists, or while a process has an open file or working directory
inside the source. `--dry-run` performs all checks without changing files.

After the rename, known source, dataset, build, profile, and cache directories
are normalized. Other legacy artifacts are retained under
`workspace/archive/legacy/`. The command neither copies the runtime nor creates
a compatibility symbolic link.

## Internal Helpers

Process inspection combines recursive open-file discovery with `/proc` working
directory inspection. Every relocation uses a same-filesystem rename and
rejects destination collisions.
