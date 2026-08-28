# Representative NCU sampling

`python -m gala_sim.tools.gpu_ncu_sampling` builds a bounded performance estimate from a representative subset of measured NCU reports. It matches launches by iteration, stage, call, ordinal, and kernel name without enforcing plan, repository, or artifact hashes.

Primary reports take precedence over supplemental historical reports. Missing launches use, in order, the closest same-stage kernel samples, same-kernel samples, or same-stage samples. The output records exact coverage, extrapolated launch fractions, fallback modes, and per-stage dispersion. The R²-Gaussian + Chest campaign fixes the primary sample count at 16 groups and records `sampling_group_count`; it is eligible for sampled performance analysis, but it is not labeled as exhaustive launch coverage.
