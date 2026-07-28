# Changelog

## [2.1.0] - 2026-07-29

### Added
- Optional `execution_mode="out_of_core"` for disk-backed, section-separable
  hierarchical fitting, with a reusable section-store writer, resumable state,
  on-demand embedding access, and a runnable example.

### Notes
- The default `execution_mode="in_memory"` and existing tutorials are unchanged.
- The MERFISH Aging engineering benchmark supports a memory-saving option; it
  does not claim a universal runtime improvement.

## [2.0.0]

### Added
- Hierarchical multisample GraphPCA.
- Global and grouped partial-pooling formulations.
- Rotation-aware gauge alignment and convergence diagnostics.
- Reusable shared loading coordinates for new-section projection workflows.
- Formal runtime-core provenance and reproducibility checks.

### Preserved
- Existing `Run_GPCA` single-sample API.
- Exact, iterative, and accelerated single-sample engines.

### Notes
- This release does not claim unseen-gene generation or general-purpose imputation.
