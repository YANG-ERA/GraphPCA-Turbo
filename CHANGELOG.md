# Changelog

## [2.0.0] - Unreleased

### Added
- Hierarchical multisample GraphPCA.
- Optional `execution_mode="out_of_core"` for disk-backed, section-separable
  hierarchical fitting, with a reusable section-store writer, resumable state,
  and on-demand embedding access.
- Global and grouped partial-pooling formulations.
- Rotation-aware gauge alignment and convergence diagnostics.
- Reusable shared loading coordinates for new-section projection workflows.
- Formal runtime-core provenance and reproducibility checks.

### Preserved
- Existing `Run_GPCA` single-sample API.
- Exact, iterative, and accelerated single-sample engines.

### Notes
- This release does not claim unseen-gene generation or general-purpose imputation.
