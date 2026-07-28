"""Hierarchical multi-sample GraphPCA-Turbo.

This module implements a partially pooled extension of GraphPCA-Turbo. Each
sample has a sample-specific orthonormal loading matrix ``W_s`` and a
sample-specific spatial embedding ``Z_s``. The sample loadings are shrunk
towards a common orthonormal loading matrix ``W_0``.

The implemented per-spot-normalized objective is

    sum_s q_s [
        (1 / n_s) * ||X_s - Z_s W_s.T||_F^2
        + (lambda_s / n_s) * tr(Z_s.T L_s Z_s)
        + rho_s * ||W_s - W_0||_F^2
    ],

subject to

    W_s.T W_s = I,  W_0.T W_0 = I.

Here ``q_s`` controls how samples contribute to the cohort objective:

* ``sample_weights='size_proportional'`` uses
  q_s = n_s / sum_t n_t. Combined with the 1/n_s-normalized slice loss, this
  gives equal contribution to individual spatial locations;
* ``sample_weights='equal_slice'`` uses q_s = 1 / S, giving every tissue
  section equal total influence regardless of its number of cells.
Legacy aliases ``'spot'`` and ``'balanced'`` remain supported.

The block updates are:

    (I + lambda_s L_s) Z_s = X_s W_s,

    W_s = polar((X_s.T Z_s) / n_s + rho_s W_0),

    W_0 = polar(sum_s q_s rho_s W_s).

The Z-step remains a sparse symmetric positive-definite system and is solved
with PCG. Both loading updates are thin-SVD orthogonal Procrustes solutions.

Sample-wise centering is implemented implicitly and therefore does not densify
sparse expression matrices.
"""

from __future__ import annotations

import logging
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix, issparse

from .multigpca import (
    HAS_CPP_MULTI,
    ArrayLike,
    ScalarOrSequence,
    _as_list,
    _build_phi,
    _centered_matmul,
    _centered_t_matmul,
    _expression_matrix,
    _gene_mean,
    _solve_embedding,
    _streaming_initial_loading,
    _validate_samples,
)

logger = logging.getLogger(__name__)


def _expression_matrix(adata: Any):
    """Return an efficient in-memory expression representation.

    Sparse matrices remain CSR. Dense standardized MERFISH expression remains
    dense float32, avoiding conversion of an all-nonzero matrix to a much larger
    CSR representation.
    """
    if not hasattr(adata, "X"):
        raise TypeError("Each sample must provide an '.X' expression matrix.")
    x = adata.X
    if issparse(x):
        return x.tocsr().astype(np.float64, copy=False)
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError("Each expression matrix must be two-dimensional.")
    return np.ascontiguousarray(x.astype(np.float32, copy=False))



@dataclass
class HierarchicalMultiGPCAInfo:
    """Diagnostics returned by :func:`Run_Hierarchical_Multi_GPCA`."""

    converged: bool
    n_iter: int
    relative_changes: List[float]
    z_changes: List[float]
    sample_loading_changes: List[float]
    global_loading_changes: List[float]
    objective_values: List[float]
    objective_relative_changes: List[float]
    sample_contributions: np.ndarray
    lambdas: np.ndarray
    rhos: np.ndarray
    gene_means: List[np.ndarray]
    pcg_status: List[List[int]]
    loading_deviations: np.ndarray
    loading_subspace_distances: np.ndarray
    loading_column_cosines: np.ndarray
    initial_cross_covariance_scales: np.ndarray
    convergence_reason: str
    objective_stable_iterations: int
    stationarity_residuals: List[float]
    z_stationarity_residuals: List[float]
    sample_loading_stationarity_residuals: List[float]
    global_loading_stationarity_residuals: List[float]
    loading_inner_iterations: List[int]


def _resolve_sample_contributions(
    adatas: Sequence[Any],
    sample_weights: Optional[Union[str, Sequence[float]]],
) -> np.ndarray:
    """Return q_s coefficients for the per-spot-normalized objective."""
    n_obs = np.asarray([adata.shape[0] for adata in adatas], dtype=np.float64)
    if sample_weights is None or (
        isinstance(sample_weights, str)
        and sample_weights in {"balanced", "equal_slice"}
    ):
        # Equal contribution from each tissue slice: q_s = 1 / S.
        contributions = np.ones_like(n_obs)
    elif (
        isinstance(sample_weights, str)
        and sample_weights in {"spot", "size_proportional"}
    ):
        # Size-proportional slice weights: q_s = n_s / sum_t n_t.
        # Combined with the 1/n_s-normalized slice loss, this gives equal
        # contribution to each spatial location in the pooled cohort.
        contributions = n_obs.copy()
    elif isinstance(sample_weights, str):
        raise ValueError(
            "sample_weights must be 'equal_slice', 'size_proportional' "
            "(legacy aliases: 'balanced', 'spot'), or a positive sequence."
        )
    else:
        contributions = np.asarray(sample_weights, dtype=np.float64)
        if contributions.shape != n_obs.shape:
            raise ValueError(f"sample_weights must have length {len(adatas)}.")

    if np.any(~np.isfinite(contributions)) or np.any(contributions <= 0):
        raise ValueError("All sample weights must be finite and strictly positive.")
    return contributions / contributions.sum()


def _resolve_rhos(rhos: ScalarOrSequence, n_samples: int) -> np.ndarray:
    values = np.asarray(_as_list(rhos, n_samples, "rhos"), dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("All loading-shrinkage parameters must be finite and non-negative.")
    return values


def _polar_factor(matrix: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    """Return the orthonormal polar factor U V^T of a p x k matrix."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("The Procrustes matrix must be two-dimensional.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("The Procrustes matrix contains non-finite values.")

    norm = float(np.linalg.norm(matrix, ord="fro"))
    if norm <= np.finfo(np.float64).eps:
        if fallback is None:
            raise ValueError("Cannot compute a polar factor from a numerically zero matrix.")
        return np.asarray(fallback, dtype=np.float64).copy()

    u, _, vt = np.linalg.svd(matrix, full_matrices=False)
    return u @ vt


def _align_loading_to_reference(loading: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Rotate an orthonormal loading to the closest orientation to reference."""
    cross = loading.T @ reference
    rotation = _polar_factor(cross, fallback=np.eye(cross.shape[0]))
    return loading @ rotation


def _initial_loadings(
    expressions: Sequence[csr_matrix],
    means: Sequence[np.ndarray],
    contributions: np.ndarray,
    n_components: int,
    random_seed: int,
    strategy: str,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    n_obs = np.asarray([x.shape[0] for x in expressions], dtype=np.float64)

    # The cohort initialization follows the normalized covariance objective:
    # sum_s q_s * X_s^T X_s / n_s.
    covariance_weights = contributions / n_obs
    covariance_weights /= covariance_weights.sum()
    w0 = _streaming_initial_loading(
        expressions,
        means,
        covariance_weights,
        n_components,
        random_seed,
    )

    if strategy == "shared":
        return w0, [w0.copy() for _ in expressions]

    individual: List[np.ndarray] = []
    for index, (x, mean) in enumerate(zip(expressions, means)):
        ws = _streaming_initial_loading(
            [x],
            [mean],
            np.ones(1, dtype=np.float64),
            n_components,
            random_seed + index + 1,
        )
        ws = _align_loading_to_reference(ws, w0)
        individual.append(ws)

    if strategy == "individual":
        return w0, individual
    if strategy == "hybrid":
        hybrid = [
            _polar_factor(ws + w0, fallback=w0)
            for ws in individual
        ]
        return w0, hybrid
    raise ValueError("init_strategy must be 'shared', 'individual', or 'hybrid'.")


def _sample_data_cross(
    expression: csr_matrix,
    mean: np.ndarray,
    embedding: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Return X_s^T Z_s / n_s and its spectral norm."""
    n_obs = expression.shape[0]
    data_cross = _centered_t_matmul(expression, mean, embedding) / float(n_obs)
    scale = float(np.linalg.norm(data_cross, ord=2))
    return data_cross, scale


def _update_sample_loading_from_cross(
    data_cross: np.ndarray,
    global_loading: np.ndarray,
    rho: float,
) -> np.ndarray:
    procrustes_matrix = data_cross + float(rho) * global_loading
    return _polar_factor(procrustes_matrix, fallback=global_loading)


def _update_sample_loading(
    expression: csr_matrix,
    mean: np.ndarray,
    embedding: np.ndarray,
    global_loading: np.ndarray,
    rho: float,
) -> Tuple[np.ndarray, float]:
    """Compatibility wrapper used by unseen-section projection."""
    data_cross, scale = _sample_data_cross(expression, mean, embedding)
    ws = _update_sample_loading_from_cross(data_cross, global_loading, rho)
    return ws, scale


def _update_global_loading(
    sample_loadings: Sequence[np.ndarray],
    contributions: np.ndarray,
    rhos: np.ndarray,
    fallback: np.ndarray,
) -> np.ndarray:
    matrix = np.zeros_like(fallback)
    effective = contributions * rhos
    if float(effective.sum()) <= np.finfo(np.float64).eps:
        # W0 is not identifiable when every rho_s is zero. We still return a
        # weighted Procrustes mean for diagnostics and a coherent orientation.
        effective = contributions
    for ws, weight in zip(sample_loadings, effective):
        matrix += float(weight) * ws
    return _polar_factor(matrix, fallback=fallback)


def _optimize_loading_blocks(
    data_crosses: Sequence[np.ndarray],
    sample_loadings: Sequence[np.ndarray],
    global_loading: np.ndarray,
    contributions: np.ndarray,
    rhos: np.ndarray,
    max_iter: int,
    tol: float,
) -> Tuple[List[np.ndarray], np.ndarray, int]:
    """Jointly optimize {W_s} and W_0 for fixed embeddings.

    The loading-only subproblem is inexpensive because it involves only
    p-by-k thin SVDs. Performing several inner Procrustes sweeps substantially
    reduces the block fixed-point residual, especially under strong shrinkage.
    """
    ws_list = [np.asarray(ws, dtype=np.float64).copy() for ws in sample_loadings]
    w0 = np.asarray(global_loading, dtype=np.float64).copy()
    iterations = 0

    for inner in range(max(1, int(max_iter))):
        previous_ws = [ws.copy() for ws in ws_list]
        previous_w0 = w0.copy()

        ws_list = [
            _update_sample_loading_from_cross(cross, w0, rho)
            for cross, rho in zip(data_crosses, rhos)
        ]
        w0 = _update_global_loading(
            ws_list,
            contributions,
            rhos,
            fallback=w0,
        )
        iterations = inner + 1

        ws_change = _relative_loading_change(
            ws_list,
            previous_ws,
            contributions,
        )
        if np.all(rhos == 0):
            joint_change = ws_change
        else:
            w0_change = _relative_matrix_change(w0, previous_w0)
            joint_change = max(ws_change, w0_change)

        if joint_change < tol:
            break

    return ws_list, w0, iterations


def _block_stationarity_residual(
    expressions: Sequence[csr_matrix],
    means: Sequence[np.ndarray],
    phis: Sequence[csr_matrix],
    embeddings: Sequence[np.ndarray],
    sample_loadings: Sequence[np.ndarray],
    global_loading: np.ndarray,
    data_crosses: Sequence[np.ndarray],
    contributions: np.ndarray,
    rhos: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Return joint, Z, W_s, and W_0 block fixed-point residuals."""
    z_residual = 0.0
    for x, mean, phi, z, ws in zip(
        expressions,
        means,
        phis,
        embeddings,
        sample_loadings,
    ):
        rhs = _centered_matmul(x, mean, ws)
        residual = np.asarray(phi @ z, dtype=np.float64) - rhs
        value = float(
            np.linalg.norm(residual, ord="fro")
            / (np.linalg.norm(rhs, ord="fro") + 1e-12)
        )
        z_residual = max(z_residual, value)

    ws_residual = 0.0
    for cross, ws, rho in zip(data_crosses, sample_loadings, rhos):
        target = _update_sample_loading_from_cross(
            cross,
            global_loading,
            rho,
        )
        value = float(
            np.linalg.norm(ws - target, ord="fro")
            / (np.sqrt(ws.shape[1]) + 1e-12)
        )
        ws_residual = max(ws_residual, value)

    if np.all(rhos == 0):
        w0_residual = 0.0
        joint = max(z_residual, ws_residual)
    else:
        target_w0 = _update_global_loading(
            sample_loadings,
            contributions,
            rhos,
            fallback=global_loading,
        )
        w0_residual = float(
            np.linalg.norm(global_loading - target_w0, ord="fro")
            / (np.sqrt(global_loading.shape[1]) + 1e-12)
        )
        joint = max(z_residual, ws_residual, w0_residual)

    return joint, z_residual, ws_residual, w0_residual


def _centered_squared_norm(expression, mean: np.ndarray) -> float:
    if issparse(expression):
        raw = float(expression.multiply(expression).sum())
    else:
        dense = np.asarray(expression)
        raw = float(
            np.einsum("ij,ij->", dense, dense, dtype=np.float64, optimize=True)
        )
    centered = raw - float(expression.shape[0]) * float(mean @ mean)
    return max(centered, 0.0)


def _objective_value(
    expressions: Sequence[csr_matrix],
    means: Sequence[np.ndarray],
    phis: Sequence[csr_matrix],
    embeddings: Sequence[np.ndarray],
    sample_loadings: Sequence[np.ndarray],
    global_loading: np.ndarray,
    contributions: np.ndarray,
    rhos: np.ndarray,
    centered_norms: np.ndarray,
) -> float:
    total = 0.0
    for x, mean, phi, z, ws, q, rho, xnorm in zip(
        expressions,
        means,
        phis,
        embeddings,
        sample_loadings,
        contributions,
        rhos,
        centered_norms,
    ):
        n_obs = float(x.shape[0])
        cross = _centered_t_matmul(x, mean, z)
        reconstruction = (
            float(xnorm)
            + float(np.sum(z * z))
            - 2.0 * float(np.sum(ws * cross))
        )
        # phi = I + lambda L, hence z^T(lambda L)z = z^T phi z - ||z||^2.
        graph_term = float(np.sum(z * (phi @ z))) - float(np.sum(z * z))
        shrinkage = float(np.sum((ws - global_loading) ** 2))
        sample_objective = (
            (reconstruction + graph_term) / n_obs
            + float(rho) * shrinkage
        )
        total += float(q) * sample_objective
    return float(total)


def _align_state_to_reference(
    embeddings: Sequence[np.ndarray],
    sample_loadings: Sequence[np.ndarray],
    global_loading: np.ndarray,
    reference_sample_loadings: Sequence[np.ndarray],
    reference_global_loading: np.ndarray,
    contributions: np.ndarray,
    rhos: np.ndarray,
) -> Tuple[
    List[np.ndarray],
    List[np.ndarray],
    np.ndarray,
    List[np.ndarray],
    str,
]:
    """Remove the orthogonal gauge using the positive-coupling components.

    For positive shrinkage, all sections and W0 form one connected component
    and therefore receive one common right rotation.  For rho=0, every section
    is an independent component and receives its own rotation.  The returned
    rotations must also be applied to cached X.T @ Z cross-products.
    """
    if np.all(rhos == 0):
        rotations: List[np.ndarray] = []
        aligned_z: List[np.ndarray] = []
        aligned_ws: List[np.ndarray] = []
        for z, ws, old_ws in zip(
            embeddings, sample_loadings, reference_sample_loadings
        ):
            cross = ws.T @ old_ws
            u, _, vt = np.linalg.svd(cross, full_matrices=False)
            rotation = u @ vt
            rotations.append(rotation)
            aligned_z.append(z @ rotation)
            aligned_ws.append(ws @ rotation)

        # W0 is reporting-only in the independent limit.  Recompute it after
        # the section-specific gauges have been fixed.
        matrix = np.zeros_like(global_loading)
        for ws, weight in zip(aligned_ws, contributions):
            matrix += float(weight) * ws
        aligned_w0 = _polar_factor(matrix, fallback=reference_global_loading)
        return (
            aligned_z,
            aligned_ws,
            aligned_w0,
            rotations,
            "independent_sections",
        )

    cross = global_loading.T @ reference_global_loading
    u, _, vt = np.linalg.svd(cross, full_matrices=False)
    rotation = u @ vt
    rotations = [rotation for _ in embeddings]
    aligned_z = [z @ rotation for z in embeddings]
    aligned_ws = [ws @ rotation for ws in sample_loadings]
    aligned_w0 = global_loading @ rotation
    return aligned_z, aligned_ws, aligned_w0, rotations, "global_common"


def _relative_embedding_change(
    current: Sequence[np.ndarray],
    previous: Sequence[np.ndarray],
    contributions: np.ndarray,
) -> float:
    numerator = 0.0
    denominator = 0.0
    for z, old, q in zip(current, previous, contributions):
        numerator += float(q) * float(np.mean((z - old) ** 2))
        denominator += float(q) * float(np.mean(z**2))
    return float(np.sqrt(numerator) / (np.sqrt(denominator) + 1e-12))


def _relative_loading_change(
    current: Sequence[np.ndarray],
    previous: Sequence[np.ndarray],
    contributions: np.ndarray,
) -> float:
    numerator = 0.0
    denominator = 0.0
    for ws, old, q in zip(current, previous, contributions):
        numerator += float(q) * float(np.sum((ws - old) ** 2))
        denominator += float(q) * float(np.sum(ws**2))
    return float(np.sqrt(numerator) / (np.sqrt(denominator) + 1e-12))


def _relative_matrix_change(current: np.ndarray, previous: np.ndarray) -> float:
    return float(
        np.linalg.norm(current - previous, ord="fro")
        / (np.linalg.norm(current, ord="fro") + 1e-12)
    )


def _joint_align(
    embeddings: Sequence[np.ndarray],
    sample_loadings: Sequence[np.ndarray],
    global_loading: np.ndarray,
    contributions: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """Apply one common rotation, preserving reconstruction and shrinkage."""
    gram = np.zeros(
        (global_loading.shape[1], global_loading.shape[1]),
        dtype=np.float64,
    )
    for z, q in zip(embeddings, contributions):
        gram += float(q) * (z.T @ z) / float(z.shape[0])
    _, rotation = np.linalg.eigh(0.5 * (gram + gram.T))
    rotation = rotation[:, ::-1]
    return (
        [z @ rotation for z in embeddings],
        [ws @ rotation for ws in sample_loadings],
        global_loading @ rotation,
    )


def _loading_diagnostics(
    sample_loadings: Sequence[np.ndarray],
    global_loading: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    deviations = []
    subspace = []
    cosines = []
    k = global_loading.shape[1]
    p0 = global_loading @ global_loading.T
    for ws in sample_loadings:
        deviations.append(float(np.linalg.norm(ws - global_loading, ord="fro")))
        ps = ws @ ws.T
        subspace.append(
            float(np.linalg.norm(ps - p0, ord="fro") / np.sqrt(2.0 * k))
        )
        cosines.append(np.diag(ws.T @ global_loading))
    return (
        np.asarray(deviations, dtype=np.float64),
        np.asarray(subspace, dtype=np.float64),
        np.asarray(cosines, dtype=np.float64),
    )


def Run_Hierarchical_Multi_GPCA(
    adatas: Optional[Sequence[Any]] = None,
    locations: Optional[Sequence[Optional[np.ndarray]]] = None,
    n_components: int = 50,
    networks: Optional[Sequence[Optional[ArrayLike]]] = None,
    platforms: Optional[Union[str, Sequence[Optional[str]]]] = None,
    lambdas: ScalarOrSequence = 0.5,
    rhos: ScalarOrSequence = 1.0,
    n_neighbors: Optional[Union[int, Sequence[Optional[int]]]] = None,
    sample_weights: Optional[Union[str, Sequence[float]]] = "equal_slice",
    center: bool = True,
    pcg_tol: float = 1e-6,
    pcg_max_iter: int = 500,
    outer_tol: float = 1e-6,
    objective_tol: float = 1e-8,
    objective_patience: int = 3,
    relaxed_outer_factor: float = 10.0,
    align_each_iteration: bool = True,
    loading_inner_max_iter: int = 10,
    loading_inner_tol: float = 1e-6,
    stationarity_tol: float = 1e-3,
    stationarity_patience: int = 2,
    check_stationarity_every: int = 1,
    max_iter: int = 20,
    random_seed: int = 666,
    init_strategy: str = "hybrid",
    align: bool = True,
    n_jobs: int = 1,
    mode: str = "iterative",
    save_embedding: bool = True,
    embedding_key: str = "X_GraphPCA_HMS",
    save_loadings: bool = True,
    sample_loading_key: str = "GraphPCA_HMS_Ws",
    global_loading_key: str = "GraphPCA_HMS_W0",
    save_reconstruction: bool = False,
    reconstruction_key: str = "GraphPCA_HMS_ReX",
    return_log: bool = False,
    return_info: bool = False,
    execution_mode: str = "in_memory",
    section_store: Optional[Any] = None,
    out_of_core_output_dir: Optional[Any] = None,
    out_of_core_resume: bool = False,
):
    """Run partially pooled hierarchical multi-sample GraphPCA-Turbo.

    Parameters
    ----------
    adatas
        AnnData-like objects containing identical genes in identical order.
    locations, networks
        Sample-specific coordinates or adjacency matrices.
    n_components
        Number of latent dimensions.
    lambdas
        Scalar or sample-specific spatial graph regularization strengths.
    rhos
        Scalar or sample-specific loading-shrinkage strengths. ``rho=0``
        yields independent sample loadings; increasing rho moves the model
        towards a fully shared loading basis.
    sample_weights
        Slice-level weighting scheme. ``'size_proportional'`` uses
        q_s=n_s/sum(n), which yields equal contribution per spatial location
        after the 1/n_s slice normalization. ``'equal_slice'`` uses q_s=1/S.
        Legacy aliases ``'spot'`` and ``'balanced'`` remain supported.
    objective_patience
        Number of consecutive iterations for which the relative objective
        change must remain below ``objective_tol`` before the plateau-based
        stopping rule can trigger.
    relaxed_outer_factor
        The plateau-based rule additionally requires the rotation-aligned
        parameter change to be below ``relaxed_outer_factor * outer_tol``.
    align_each_iteration
        Remove the non-identifiable common orthogonal rotation before
        convergence diagnostics.
    loading_inner_max_iter, loading_inner_tol
        Number and tolerance of inexpensive inner Procrustes sweeps used to
        jointly optimize the section-specific and global loading blocks for
        fixed embeddings.
    stationarity_tol, stationarity_patience
        Tolerance and required consecutive iterations for the block
        fixed-point residual. This is the primary rotation-invariant
        convergence criterion.
    check_stationarity_every
        Evaluate the block residual every this many outer iterations.
    center
        Implicitly center genes within each sample. Set to ``False`` when the
        input has already been centered by preprocessing such as sc.pp.scale.
    mode
        ``'iterative'`` uses SciPy PCG. ``'accelerated'`` uses the patched C++
        Z-only PCG kernel when available and otherwise falls back to Python.
    init_strategy
        ``'shared'``, ``'individual'``, or ``'hybrid'``. The default hybrid
        initialization averages cohort and aligned sample-specific loading
        estimates on the Stiefel manifold.
    execution_mode
        ``'in_memory'`` (default) preserves the established AnnData API.
        ``'out_of_core'`` processes a prepared disk-backed section store one
        section at a time. It requires ``section_store`` and
        ``out_of_core_output_dir``; its returned embedding collection loads
        one section on demand rather than retaining all embeddings in RAM.
        Spatial graphs must be written into the section store in advance;
        this mode does not accept ``locations`` or ``networks``. It uses the
        objective-plus-aligned-change stopping rule; stationarity diagnostics
        are reported as unavailable (``NaN``).

    Returns
    -------
    Z_list, W0, Ws_list
        Sample embeddings, global loading, and sample-specific loadings.
        Optional logs and diagnostics are appended according to flags.
    """
    if execution_mode == "out_of_core":
        if adatas is not None:
            raise ValueError(
                "Pass adatas=None in out_of_core mode. Creating a full AnnData "
                "list first would not provide end-to-end memory savings."
            )
        if any(value is not None for value in (locations, networks, platforms, n_neighbors)):
            raise ValueError(
                "out_of_core mode reads its spatial graphs from section_store; "
                "do not also provide locations, networks, platforms, or n_neighbors."
            )
        if not align or not align_each_iteration:
            raise ValueError(
                "out_of_core mode requires rotation alignment for stable disk-backed updates."
            )
        if section_store is None or out_of_core_output_dir is None:
            raise ValueError(
                "out_of_core mode requires section_store and out_of_core_output_dir."
            )
        if return_log or save_reconstruction:
            raise ValueError(
                "return_log and save_reconstruction are unavailable in out_of_core mode."
            )
        if init_strategy != "shared":
            warnings.warn(
                "out_of_core mode uses a streaming shared initialization; "
                "init_strategy is ignored.",
                RuntimeWarning,
                stacklevel=2,
            )
        if n_jobs != 1:
            warnings.warn(
                "out_of_core mode intentionally processes one section at a time; "
                "n_jobs is ignored.",
                RuntimeWarning,
                stacklevel=2,
            )
        from .out_of_core_hierarchical import run_hierarchical_multi_gpca_out_of_core
        return run_hierarchical_multi_gpca_out_of_core(
            section_store=section_store,
            output_dir=out_of_core_output_dir,
            n_components=n_components,
            lambdas=lambdas,
            rhos=rhos,
            sample_weights=sample_weights,
            center=center,
            pcg_tol=pcg_tol,
            pcg_max_iter=pcg_max_iter,
            outer_tol=outer_tol,
            objective_tol=objective_tol,
            objective_patience=objective_patience,
            relaxed_outer_factor=relaxed_outer_factor,
            loading_inner_max_iter=loading_inner_max_iter,
            loading_inner_tol=loading_inner_tol,
            max_iter=max_iter,
            random_seed=random_seed,
            mode=mode,
            resume=out_of_core_resume,
            return_info=return_info,
        )
    if execution_mode != "in_memory":
        raise ValueError("execution_mode must be 'in_memory' or 'out_of_core'.")
    if adatas is None:
        raise ValueError("adatas is required for execution_mode='in_memory'.")
    adatas = list(adatas)
    p = _validate_samples(adatas, n_components)
    n_samples = len(adatas)

    if mode not in {"iterative", "accelerated"}:
        raise ValueError("mode must be 'iterative' or 'accelerated'.")
    use_cpp = mode == "accelerated" and HAS_CPP_MULTI
    if mode == "accelerated" and not HAS_CPP_MULTI:
        warnings.warn(
            "The installed gpca_cpp extension does not expose the Z-only PCG "
            "kernel. Falling back to the Python iterative solver.",
            RuntimeWarning,
            stacklevel=2,
        )
        use_cpp = False

    if pcg_tol <= 0 or outer_tol <= 0 or objective_tol < 0:
        raise ValueError("Solver and convergence tolerances must be valid and positive.")
    if (
        pcg_max_iter < 1
        or max_iter < 1
        or n_jobs < 1
        or objective_patience < 1
        or loading_inner_max_iter < 1
        or stationarity_patience < 1
        or check_stationarity_every < 1
    ):
        raise ValueError(
            "Iteration limits, patience parameters, check frequency, and "
            "n_jobs must be positive integers."
        )
    if relaxed_outer_factor < 1:
        raise ValueError("relaxed_outer_factor must be at least 1.")
    if loading_inner_tol <= 0 or stationarity_tol <= 0:
        raise ValueError("loading_inner_tol and stationarity_tol must be positive.")

    locations_list = (
        [None] * n_samples
        if locations is None
        else _as_list(locations, n_samples, "locations")
    )
    networks_list = (
        [None] * n_samples
        if networks is None
        else _as_list(networks, n_samples, "networks")
    )
    platforms_list = _as_list(platforms, n_samples, "platforms")
    lambdas_array = np.asarray(
        _as_list(lambdas, n_samples, "lambdas"),
        dtype=np.float64,
    )
    if np.any(~np.isfinite(lambdas_array)) or np.any(lambdas_array < 0):
        raise ValueError("All lambdas must be finite and non-negative.")
    rhos_array = _resolve_rhos(rhos, n_samples)
    neighbors_list = _as_list(n_neighbors, n_samples, "n_neighbors")
    contributions = _resolve_sample_contributions(adatas, sample_weights)

    expressions = [_expression_matrix(adata) for adata in adatas]
    means = [_gene_mean(x, center) for x in expressions]
    phis = [
        _build_phi(
            adata.shape[0],
            location,
            network,
            platform,
            neighbors,
            graph_lambda,
        )
        for adata, location, network, platform, neighbors, graph_lambda in zip(
            adatas,
            locations_list,
            networks_list,
            platforms_list,
            neighbors_list,
            lambdas_array,
        )
    ]
    centered_norms = np.asarray(
        [_centered_squared_norm(x, mean) for x, mean in zip(expressions, means)],
        dtype=np.float64,
    )

    logger.info(
        "Initializing hierarchical multi-sample GraphPCA: %d samples, %d spots, "
        "%d genes, %d components.",
        n_samples,
        sum(adata.shape[0] for adata in adatas),
        p,
        n_components,
    )
    logger.info("rhos=%s | sample contributions=%s", rhos_array, contributions)

    w0, ws_list = _initial_loadings(
        expressions,
        means,
        contributions,
        n_components,
        random_seed,
        init_strategy,
    )
    z_list = [
        np.zeros((adata.shape[0], n_components), dtype=np.float64)
        for adata in adatas
    ]

    relative_changes: List[float] = []
    z_changes: List[float] = []
    ws_changes: List[float] = []
    w0_changes: List[float] = []
    objective_values: List[float] = []
    objective_relative_changes: List[float] = []
    stationarity_residuals: List[float] = []
    z_stationarity_residuals: List[float] = []
    ws_stationarity_residuals: List[float] = []
    w0_stationarity_residuals: List[float] = []
    loading_inner_iterations: List[int] = []
    logs: List[Tuple[List[np.ndarray], np.ndarray, List[np.ndarray]]] = []
    pcg_status: List[List[int]] = [[] for _ in adatas]
    initial_scales = np.zeros(n_samples, dtype=np.float64)
    converged = False
    convergence_reason = "maximum_iterations"
    objective_stable_count = 0
    stationarity_stable_count = 0
    quotient_stable_count = 0
    iterations_run = max_iter

    for iteration in range(max_iter):
        start = time.time()
        previous_z = [z.copy() for z in z_list]
        previous_ws = [ws.copy() for ws in ws_list]
        previous_w0 = w0.copy()

        def solve_one(index: int):
            return _solve_embedding(
                expressions[index],
                means[index],
                phis[index],
                ws_list[index],
                previous_z[index],
                pcg_tol,
                pcg_max_iter,
                use_cpp=use_cpp,
            )

        if n_jobs == 1:
            results = [solve_one(i) for i in range(n_samples)]
        else:
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                futures = [executor.submit(solve_one, i) for i in range(n_samples)]
                results = [future.result() for future in futures]

        z_list = [result[0] for result in results]
        pcg_status = [result[1] for result in results]
        for sample_index, status in enumerate(pcg_status):
            failed = [i for i, code in enumerate(status) if code != 0]
            if failed:
                logger.warning(
                    "Sample %d: PCG did not fully converge for %d/%d components; "
                    "status codes=%s.",
                    sample_index,
                    len(failed),
                    n_components,
                    sorted(set(status[i] for i in failed)),
                )

        data_crosses: List[np.ndarray] = []
        current_scales = np.zeros(n_samples, dtype=np.float64)
        for index in range(n_samples):
            data_cross, scale = _sample_data_cross(
                expressions[index],
                means[index],
                z_list[index],
            )
            data_crosses.append(data_cross)
            current_scales[index] = scale
        if iteration == 0:
            initial_scales = current_scales.copy()

        ws_list, w0, loading_inner_count = _optimize_loading_blocks(
            data_crosses,
            ws_list,
            w0,
            contributions,
            rhos_array,
            loading_inner_max_iter,
            loading_inner_tol,
        )
        loading_inner_iterations.append(int(loading_inner_count))

        if align_each_iteration:
            (
                z_list,
                ws_list,
                w0,
                gauge_rotations,
                gauge_mode,
            ) = _align_state_to_reference(
                z_list,
                ws_list,
                w0,
                previous_ws,
                previous_w0,
                contributions,
                rhos_array,
            )
            # Cached cross-products must use the same coordinate system as the
            # aligned Z/W variables.  Without this step, block-stationarity
            # diagnostics are computed in mismatched gauges.
            data_crosses = [
                cross @ rotation
                for cross, rotation in zip(data_crosses, gauge_rotations)
            ]
        else:
            gauge_mode = (
                "independent_sections"
                if np.all(rhos_array == 0)
                else "global_common"
            )

        objective = _objective_value(
            expressions,
            means,
            phis,
            z_list,
            ws_list,
            w0,
            contributions,
            rhos_array,
            centered_norms,
        )
        objective_values.append(objective)
        if len(objective_values) == 1:
            objective_change = np.inf
        else:
            previous_objective = objective_values[-2]
            objective_change = abs(objective - previous_objective) / (
                abs(previous_objective) + 1e-12
            )
            if objective > previous_objective * (1.0 + 1e-7):
                logger.warning(
                    "Objective increased from %.8e to %.8e. This can occur when "
                    "PCG solves are too loose; consider decreasing pcg_tol.",
                    previous_objective,
                    objective,
                )
        objective_relative_changes.append(float(objective_change))

        z_change = _relative_embedding_change(z_list, previous_z, contributions)
        ws_change = _relative_loading_change(ws_list, previous_ws, contributions)
        w0_change = _relative_matrix_change(w0, previous_w0)

        # W0 is not identifiable when every rho_s is zero and must not prevent
        # convergence of the independent-slice limit.
        if np.all(rhos_array == 0):
            joint_change = max(z_change, ws_change)
        else:
            joint_change = max(z_change, ws_change, w0_change)

        z_changes.append(z_change)
        ws_changes.append(ws_change)
        w0_changes.append(w0_change)
        relative_changes.append(joint_change)

        if (
            (iteration + 1) % check_stationarity_every == 0
            or iteration == 0
            or iteration + 1 == max_iter
        ):
            (
                stationarity,
                z_stationarity,
                ws_stationarity,
                w0_stationarity,
            ) = _block_stationarity_residual(
                expressions,
                means,
                phis,
                z_list,
                ws_list,
                w0,
                data_crosses,
                contributions,
                rhos_array,
            )
        else:
            stationarity = np.inf
            z_stationarity = np.inf
            ws_stationarity = np.inf
            w0_stationarity = np.inf

        stationarity_residuals.append(float(stationarity))
        z_stationarity_residuals.append(float(z_stationarity))
        ws_stationarity_residuals.append(float(ws_stationarity))
        w0_stationarity_residuals.append(float(w0_stationarity))

        logger.info(
            "Hierarchical iteration %d/%d completed in %.2fs | joint=%.4e | "
            "Z=%.4e | Ws=%.4e | W0=%.4e | stat=%.4e | objective=%.8e",
            iteration + 1,
            max_iter,
            time.time() - start,
            joint_change,
            z_change,
            ws_change,
            w0_change,
            stationarity,
            objective,
        )

        if return_log:
            logs.append(
                (
                    [z.copy() for z in z_list],
                    w0.copy(),
                    [ws.copy() for ws in ws_list],
                )
            )

        objective_this_iteration = (
            objective_tol > 0
            and len(objective_values) > 1
            and objective_relative_changes[-1] < objective_tol
        )
        if objective_this_iteration:
            objective_stable_count += 1
        else:
            objective_stable_count = 0

        if np.isfinite(stationarity) and stationarity < stationarity_tol:
            stationarity_stable_count += 1
        elif np.isfinite(stationarity):
            stationarity_stable_count = 0

        strict_quotient_stable = (
            objective_this_iteration
            and joint_change < outer_tol
        )
        plateau_quotient_stable = (
            objective_this_iteration
            and joint_change < relaxed_outer_factor * outer_tol
        )
        if strict_quotient_stable or plateau_quotient_stable:
            quotient_stable_count += 1
        else:
            quotient_stable_count = 0

        if quotient_stable_count >= objective_patience:
            converged = True
            iterations_run = iteration + 1
            convergence_reason = (
                "strict_joint"
                if strict_quotient_stable
                else "objective_plateau_with_bounded_aligned_change"
            )
            logger.info(
                "Hierarchical multi-sample GraphPCA converged in %d iterations "
                "(reason=%s; quotient_stable_count=%d; gauge=%s; "
                "stationarity=%.4e).",
                iterations_run,
                convergence_reason,
                quotient_stable_count,
                gauge_mode,
                stationarity,
            )
            break
    else:
        logger.warning("Reached maximum alternating iterations (%d).", max_iter)

    # A final common variance-ordering rotation is valid only for a connected
    # positive-coupling model.  Independent sections retain separate gauges and
    # are aligned to an external reference only for explicit joint sensitivity.
    if align and not np.all(rhos_array == 0):
        z_list, ws_list, w0 = _joint_align(
            z_list,
            ws_list,
            w0,
            contributions,
        )

    deviations, subspace_distances, column_cosines = _loading_diagnostics(
        ws_list,
        w0,
    )

    if save_embedding:
        for adata, z in zip(adatas, z_list):
            if hasattr(adata, "obsm"):
                adata.obsm[embedding_key] = z

    if save_loadings:
        for adata, ws in zip(adatas, ws_list):
            if hasattr(adata, "varm"):
                adata.varm[sample_loading_key] = ws
                adata.varm[global_loading_key] = w0

    if save_reconstruction:
        for adata, z, ws, mean in zip(adatas, z_list, ws_list, means):
            reconstruction = z @ ws.T
            if center:
                reconstruction += mean[None, :]
            if hasattr(adata, "layers"):
                adata.layers[reconstruction_key] = reconstruction

    info = HierarchicalMultiGPCAInfo(
        converged=converged,
        n_iter=iterations_run,
        relative_changes=relative_changes,
        z_changes=z_changes,
        sample_loading_changes=ws_changes,
        global_loading_changes=w0_changes,
        objective_values=objective_values,
        objective_relative_changes=objective_relative_changes,
        sample_contributions=contributions.copy(),
        lambdas=lambdas_array.copy(),
        rhos=rhos_array.copy(),
        gene_means=[mean.copy() for mean in means],
        pcg_status=pcg_status,
        loading_deviations=deviations,
        loading_subspace_distances=subspace_distances,
        loading_column_cosines=column_cosines,
        initial_cross_covariance_scales=initial_scales,
        convergence_reason=convergence_reason,
        objective_stable_iterations=int(objective_stable_count),
        stationarity_residuals=stationarity_residuals,
        z_stationarity_residuals=z_stationarity_residuals,
        sample_loading_stationarity_residuals=ws_stationarity_residuals,
        global_loading_stationarity_residuals=w0_stationarity_residuals,
        loading_inner_iterations=loading_inner_iterations,
    )

    output: List[Any] = [z_list, w0, ws_list]
    if return_log:
        output.append(logs)
    if return_info:
        output.append(info)
    return tuple(output)


def Project_Hierarchical_Multi_GPCA(
    adata: Any,
    global_loading: np.ndarray,
    location: Optional[np.ndarray] = None,
    network: Optional[ArrayLike] = None,
    platform: Optional[str] = None,
    graph_lambda: float = 0.5,
    rho: float = 1.0,
    n_neighbors: Optional[int] = None,
    center: bool = True,
    gene_mean: Optional[np.ndarray] = None,
    pcg_tol: float = 1e-6,
    pcg_max_iter: int = 500,
    outer_tol: float = 1e-6,
    max_iter: int = 20,
    mode: str = "iterative",
    save_embedding: bool = True,
    embedding_key: str = "X_GraphPCA_HMS",
    save_loading: bool = True,
    sample_loading_key: str = "GraphPCA_HMS_Ws",
    global_loading_key: str = "GraphPCA_HMS_W0",
) -> Tuple[np.ndarray, np.ndarray]:
    """Adapt an unseen sample to a fixed cohort loading ``W_0``.

    This alternates a PCG Z-step with

        W_new = polar(X_new.T Z_new / n_new + rho W_0),

    while keeping ``W_0`` fixed.
    """
    if rho < 0 or not np.isfinite(rho):
        raise ValueError("rho must be finite and non-negative.")
    if mode not in {"iterative", "accelerated"}:
        raise ValueError("mode must be 'iterative' or 'accelerated'.")

    x = _expression_matrix(adata)
    w0 = np.asarray(global_loading, dtype=np.float64)
    if w0.ndim != 2 or w0.shape[0] != x.shape[1]:
        raise ValueError(
            f"global_loading must have shape ({x.shape[1]}, k); received {w0.shape}."
        )
    if not np.allclose(w0.T @ w0, np.eye(w0.shape[1]), atol=1e-5):
        raise ValueError("global_loading must have orthonormal columns.")

    if gene_mean is None:
        mean = _gene_mean(x, center)
    else:
        mean = np.asarray(gene_mean, dtype=np.float64)
        if mean.shape != (x.shape[1],):
            raise ValueError(f"gene_mean must have shape ({x.shape[1]},).")

    phi = _build_phi(
        x.shape[0],
        location,
        network,
        platform,
        n_neighbors,
        graph_lambda,
    )
    use_cpp = mode == "accelerated" and HAS_CPP_MULTI
    if mode == "accelerated" and not HAS_CPP_MULTI:
        warnings.warn(
            "The accelerated Z-only kernel is unavailable; using Python PCG.",
            RuntimeWarning,
            stacklevel=2,
        )

    ws = w0.copy()
    z = np.zeros((x.shape[0], w0.shape[1]), dtype=np.float64)
    for _ in range(max_iter):
        old_z = z.copy()
        old_ws = ws.copy()
        z, status = _solve_embedding(
            x,
            mean,
            phi,
            ws,
            old_z,
            pcg_tol,
            pcg_max_iter,
            use_cpp=use_cpp,
        )
        if any(code != 0 for code in status):
            logger.warning("PCG did not fully converge during sample adaptation.")
        ws, _ = _update_sample_loading(x, mean, z, w0, rho)
        change = max(
            float(
                np.linalg.norm(z - old_z, ord="fro")
                / (np.linalg.norm(z, ord="fro") + 1e-12)
            ),
            _relative_matrix_change(ws, old_ws),
        )
        if change < outer_tol:
            break

    if save_embedding and hasattr(adata, "obsm"):
        adata.obsm[embedding_key] = z
    if save_loading and hasattr(adata, "varm"):
        adata.varm[sample_loading_key] = ws
        adata.varm[global_loading_key] = w0
    return z, ws


# PEP-8 aliases while retaining the repository's Run_GPCA naming style.
run_hierarchical_multi_gpca = Run_Hierarchical_Multi_GPCA
project_hierarchical_multi_gpca = Project_Hierarchical_Multi_GPCA
