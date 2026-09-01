# sample.py

## External Interfaces

`QueryRange` and `TraceSampleConfig` define dependency-closed query selection
and resource bounds. `dependency_closed_query_sample()` retains selected
terminals and every transitive dependency, then rebases event, dependency, and
payload offsets into a validated trace.

`QueryPacketSampleConfig` and `real_query_packet_sample()` construct bounded
physical query packets with real candidates, relations, state versions,
consumers, adjoints, and gradients. CPU and CUDA scan backends have identical
selection semantics.

## Internal Helpers

Chunked scanners bound host memory. Dense rebasing preserves source identity,
lineage, and payload order. Sample metadata identifies its bounded scope and
prevents use where a complete trace is required.
