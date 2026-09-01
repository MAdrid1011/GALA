# main.py

## External Interface

`gala-sim workspace-check [--workspace PATH]` discovers the repository,
creates the standard ignored workspace directories, and prints portable path
references.

`gala-sim acquire --all|--models IDS|--datasets IDS [--workspace PATH]
[--dry-run] [--no-resume]` validates the asset catalog and acquires selected
public sources. Comma-separated identifiers are accepted for model and dataset
selections.

Configuration and input validation commands include `config-check`,
`native-preflight`, `cycle-preflight`, `relation-capacity-preflight`,
`trace-validate`, and `trace-archive-validate`.

Trace transformation commands include `trace-sample`, `trace-query-packets`,
`trace-packetize`, `trace-captured-packets`, `trace-archive-snapshot`, and the
representative-packet planning commands. They preserve event identities and
mark any intentionally bounded output in its generated manifest.

Cycle commands include `cycle-replay`, `cycle-bounds`, `ablation`, and
`archive-ablation`. Archive ablation requires explicit `--model` and
`--dataset` identifiers. Variant order and compiler/hardware prerequisites are
validated by the ablation layer.

Every command prints machine-readable JSON on success where practical and
returns exit code 2 with an explanatory stderr message for validation or
runtime errors.

## Internal Helpers

Argument parsers normalize query ranges, query domains, asset identifier lists,
Ramulator bindings, and resource snapshots. Output helpers keep module
breakdowns and variant run directories relative to their owning manifest.
