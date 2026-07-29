# Changelog

## [2.2.0] - 2026-07-30

### Added
- Dense-aware disk-backed section stores: `expression_storage="auto"` keeps
  high-density expression matrices as float32 `.npy` arrays rather than
  coercing them to CSR.
- Optional `out_of_core_group_labels` for stage-, donor-, or other
  group-specific loading centres while preserving a common rotational gauge.
- Complete-snapshot checkpoint publication with an atomic current-checkpoint
  pointer, including safe recovery after an interrupted staged update.

### Improved
- Disk-backed checkpoint cleanup tolerates transient macOS/exFAT `._*`
  sidecars after the new checkpoint has been published.
- Documentation now describes the global `O(SpK)` loading/cross-product state
  honestly and recommends benchmarking the OpenMP thread limit.

### Notes
- The default in-memory API and existing tutorials remain unchanged.
- Disk-backed mode is a memory-saving option, not a universal runtime
  improvement.

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
