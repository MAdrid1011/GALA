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

`representative-ablation` runs the complete seven-variant matrix over the
calibrated phase-stratified representative windows. Apply the same strategy to
all workload documents and verify their common provenance with
`representative-matrix-audit`. The audit requires all twelve documents to
embed the exact calibrated strategy contract and marks them as complete
representative-window experiments; it does not relabel them as a quick path.

`campaign-ablation` composes trace capture, necessary-bound analysis, and the
seven canonical ablations. Its default and `--representative-archive` modes are
bounded CPU engineering checks. `--official-trace` uses the registered model's
official entrypoint and retains the model-specific trace in the ignored
workspace; capture wall time is not treated as GPU Base performance.

Cycle commands include `cycle-replay`, `cycle-bounds`, `ablation`, and
`archive-ablation`. Archive ablation requires explicit `--model` and
`--dataset` identifiers. Variant order and compiler/hardware prerequisites are
validated by the ablation layer. `archive-ablation --adaptive-stop` (with the
legacy alias `--stop-when-speedup-stable`) runs the same end-to-end replay while
stopping at a common stable iteration boundary and emits an explicit full-run
stability certificate; it never silently upgrades an incomplete source archive
to formal performance eligibility.

Every command prints machine-readable JSON on success where practical and
returns exit code 2 with an explanatory stderr message for validation or
runtime errors.

## Internal Helpers

Argument parsers normalize query ranges, query domains, asset identifier lists,
Ramulator bindings, and resource snapshots. Output helpers keep module
breakdowns and variant run directories relative to their owning manifest.
