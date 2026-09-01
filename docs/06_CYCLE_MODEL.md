# Cycle Model and Event Semantics

## Event Contract

The cycle model consumes validated CLAMP events with global IDs, explicit
dependencies, state versions, relation ownership, template IDs, field masks,
and deterministic address tokens. Events are immutable after validation.

Candidate, relation, forward, reduction, consumer, adjoint, gradient, update,
and set-mutation events form one lifecycle. An update cannot commit until all
reads and gradients of the old state version have drained.

## Engine Semantics

The timing engine is event-driven. Each module exposes its next wakeup,
acceptance decision, state transition, and counters. The engine advances to the
earliest observable transition and preserves backpressure when a downstream
resource cannot accept work.

A module may inspect its configured queues, tables, ports, and local metadata.
It may not inspect arbitrary future trace events, expand capacity, or bypass a
dependency. Deterministic tie-breaking uses physical arrival order and stable
event IDs.

## Real-Trace Burst Capacity

Trace packets are admitted through bounded frontiers. A packet that exceeds a
physical window is divided into continuation windows while retaining global
event and dependency identities. The engine reports peak in-flight events,
relations, packet bytes, and queue occupancy.

Streaming validation carries lifecycle state across chunk boundaries. End of
stream is accepted only after every relation, reduction, update, and mutation
has closed.

## Module Flow

1. The relation constructor admits candidates and emits valid relations.
2. Fusion queues hold physical task heads until dependencies and resources are ready.
3. The issue unit selects legal work under policy A/C and port constraints.
4. The semantic cache resolves state reads under policy B/D and SRAM capacity.
5. Compute Pods execute the declared arithmetic template.
6. Query and gradient units apply reductions and consumer dependencies.
7. The update unit commits state versions and set mutations.
8. Memory requests complete only after all Ramulator transactions return.

## Base and Optimized Policies

Base ASIC uses the same modules, templates, ports, banks, queues, interconnect,
and memory configuration as optimized variants. It issues legal queue heads in
arrival order and uses the ordinary state-fetch path.

Compiler metadata can expose bounded query-load and semantic-workset
information. Matching hardware policies consume that metadata for issue and
residency. Disabled policies do not remove tasks or hardware needed for
correctness.

## Bounds

Oracle policies may choose the best legal candidate within their declared
scope but retain dependencies, issue width, ports, banks, queues, capacity,
compute resources, and memory timing. Oracle output is diagnostic and remains
separate from implemented variants.

## End-to-End Cycles

Cycle zero is the first cycle in which the relation constructor can accept
work. The final cycle is the completion of the last update or set mutation and
all hardware work that contributes to it. Asset acquisition and offline quality
metric calculation are outside this interval.

Every run reports end-to-end cycles, per-module active and blocked cycles,
stall causes, memory transactions, and peak resource occupancy.
