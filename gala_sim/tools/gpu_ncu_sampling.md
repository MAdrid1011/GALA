# Representative NCU sampling

`python -m gala_sim.tools.gpu_ncu_sampling` builds a bounded performance estimate from a representative subset of measured NCU reports. It matches launches by iteration, stage, call, ordinal, and kernel name without enforcing plan, repository, or artifact hashes.

Primary reports take precedence over supplemental reports. Missing launches use,
in order, the closest same-stage kernel samples, same-kernel samples, or
same-stage samples. The output records exact coverage, extrapolated launch
fractions, fallback modes, per-stage dispersion, and the configured
`sampling_group_count`. Representative sampling is identified separately from
exhaustive launch coverage.
