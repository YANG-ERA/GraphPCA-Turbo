"""Disk-backed execution mode for hierarchical multi-sample GraphPCA.

The helpers in this module are deliberately separate from AnnData I/O.  A
caller may yield one expression matrix and graph at a time while creating a
section store.  Fitting then holds one section's expression, graph, and
embedding in RAM at a time; the shared loading blocks scale with the number
of sections and components, not with the total number of spatial locations.
"""

from __future__ import annotations

import gc
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import (
    Any,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix, load_npz, save_npz

from .hierarchical_multigpca import (
    HierarchicalMultiGPCAInfo,
    _centered_matmul,
    _centered_t_matmul,
    _polar_factor,
    _relative_loading_change,
    _resolve_rhos,
    _solve_embedding,
    _update_global_loading,
    _update_sample_loading_from_cross,
)
from .multigpca import HAS_CPP_MULTI, _build_phi


def _safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "section"


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON encode {type(value)!r}")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=_json_default) + "\n")


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically replace a small JSON control file.

    Checkpoint payloads are written to a new directory first.  This pointer is
    the only mutable control file, so a process interruption can leave at most
    an orphaned checkpoint, never a partially selected one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=_json_default) + "\n")
    os.replace(temporary, path)


def _read_manifest(store_root: Union[str, Path]) -> List[dict]:
    root = Path(store_root)
    path = root / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing section-store manifest: {path}")
    data = json.loads(path.read_text())
    sections = data.get("sections", [])
    if not sections:
        raise ValueError("The section store contains no sections.")
    required = {
        "name",
        "n_cells",
        "n_genes",
        "x_path",
        "graph_path",
        "mean_path",
        "sum_sq",
    }
    normalized: List[dict] = []
    for source_row in sections:
        row = dict(source_row)
        missing = required.difference(row)
        if missing:
            raise ValueError(f"Malformed section-store row; missing {sorted(missing)}")
        for key in ("x_path", "graph_path", "mean_path"):
            stored_path = Path(row[key])
            resolved_path = (
                stored_path if stored_path.is_absolute() else root / stored_path
            )
            if not resolved_path.exists():
                raise FileNotFoundError(f"Missing stored section file: {resolved_path}")
            row[key] = str(resolved_path)
        normalized.append(row)
    names = [row["name"] for row in normalized]
    if len(names) != len(set(names)):
        raise ValueError("Section names must be unique in a section store.")
    return normalized


def create_hierarchical_section_store(
    store_root: Union[str, Path],
    sections: Iterable[Tuple[str, Any, Any]],
    *,
    overwrite: bool = False,
    expression_storage: str = "auto",
    dense_threshold: float = 0.5,
) -> Path:
    """Write a reusable disk-backed section store.

    Parameters
    ----------
    store_root
        New output directory for ``manifest.json`` and per-section arrays.
    sections
        An iterable yielding ``(name, expression, graph)``. It is consumed
        once, making it suitable for generators that load a single section
        from disk. ``expression`` is observations by genes and ``graph`` is a
        square symmetric spatial adjacency matrix.
    overwrite
        Allow writing into an existing empty-or-managed store directory.
    expression_storage
        ``"auto"`` stores matrices with density at least ``dense_threshold``
        as ``.npy`` arrays and otherwise as compressed CSR matrices.  Use
        ``"dense"`` or ``"sparse"`` to force one representation.
    dense_threshold
        Density threshold used by ``expression_storage="auto"``.
    """
    root = Path(store_root)
    manifest_path = root / "manifest.json"
    if root.exists() and any(root.iterdir()) and not manifest_path.exists():
        raise FileExistsError(
            f"Refusing to write a section store into a nonempty unmanaged directory: {root}"
        )
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Section store already exists: {root}")
    if expression_storage not in {"auto", "dense", "sparse"}:
        raise ValueError("expression_storage must be 'auto', 'dense', or 'sparse'.")
    if not 0.0 <= dense_threshold <= 1.0:
        raise ValueError("dense_threshold must be between 0 and 1.")
    root.mkdir(parents=True, exist_ok=True)
    section_root = root / "sections"
    section_root.mkdir(exist_ok=True)
    rows: List[dict] = []
    n_genes: Optional[int] = None
    for order, (name, expression, graph) in enumerate(sections):
        safe = _safe_name(name)
        if any(row["safe_name"] == safe for row in rows):
            raise ValueError(
                f"Section names collide after filesystem sanitization: {name!r}"
            )
        x = (
            expression.tocsr()
            if sparse.issparse(expression)
            else csr_matrix(np.asarray(expression))
        )
        x = x.astype(np.float32, copy=False)
        x.sum_duplicates()
        x.eliminate_zeros()
        adjacency = (
            graph.tocsr() if sparse.issparse(graph) else csr_matrix(np.asarray(graph))
        )
        adjacency = adjacency.astype(np.float32, copy=False)
        adjacency = (0.5 * (adjacency + adjacency.T)).tocsr()
        adjacency.setdiag(0.0)
        adjacency.eliminate_zeros()
        if x.shape[0] != adjacency.shape[0] or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError(f"Expression/graph shape mismatch for section {name!r}.")
        if n_genes is None:
            n_genes = int(x.shape[1])
        elif x.shape[1] != n_genes:
            raise ValueError("Every section must have the same number of genes.")
        directory = section_root / safe
        directory.mkdir(exist_ok=True)
        density = float(x.nnz) / float(max(1, x.shape[0] * x.shape[1]))
        use_dense = expression_storage == "dense" or (
            expression_storage == "auto" and density >= dense_threshold
        )
        x_path = directory / ("X.npy" if use_dense else "X.npz")
        graph_path, mean_path = directory / "graph.npz", directory / "mean.npy"
        if use_dense:
            np.save(x_path, x.toarray().astype(np.float32, copy=False))
        else:
            save_npz(x_path, x, compressed=True)
        save_npz(graph_path, adjacency, compressed=True)
        mean = np.asarray(x.mean(axis=0)).ravel().astype(np.float64)
        np.save(mean_path, mean)
        rows.append(
            {
                "name": str(name),
                "safe_name": safe,
                "order": order,
                "n_cells": int(x.shape[0]),
                "n_genes": int(x.shape[1]),
                "x_path": str(x_path.relative_to(root)),
                "x_format": "dense_npy" if use_dense else "csr_npz",
                "x_density": density,
                "graph_path": str(graph_path.relative_to(root)),
                "mean_path": str(mean_path.relative_to(root)),
                "sum_sq": float(x.multiply(x).sum()),
            }
        )
        del x, adjacency
        gc.collect()
    if not rows:
        raise ValueError("At least one section is required.")
    _write_json(
        manifest_path,
        {
            "format": "graphpca_hierarchical_section_store_v2",
            "n_genes": n_genes,
            "sections": rows,
        },
    )
    return root


class DiskBackedEmbeddings(Sequence[np.ndarray]):
    """Read-only embedding collection that loads one section on demand."""

    def __init__(self, paths: Sequence[Union[str, Path]], names: Sequence[str]):
        self.paths = tuple(Path(path) for path in paths)
        self.names = tuple(str(name) for name in names)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: Union[int, slice]):
        if isinstance(index, slice):
            return [self.load(i) for i in range(*index.indices(len(self)))]
        return self.load(index)

    def load(self, index: int, mmap_mode: Optional[str] = None) -> np.ndarray:
        return np.load(self.paths[index], mmap_mode=mmap_mode)

    def items(self) -> Iterator[Tuple[str, np.ndarray]]:
        for index, name in enumerate(self.names):
            yield name, self.load(index)


def _contributions(
    rows: Sequence[dict], sample_weights: Optional[Union[str, Sequence[float]]]
) -> np.ndarray:
    n = np.asarray([row["n_cells"] for row in rows], dtype=np.float64)
    if sample_weights is None or (
        isinstance(sample_weights, str)
        and sample_weights in {"equal_slice", "balanced"}
    ):
        value = np.ones_like(n)
    elif isinstance(sample_weights, str) and sample_weights in {
        "size_proportional",
        "spot",
    }:
        value = n.copy()
    elif isinstance(sample_weights, str):
        raise ValueError(
            "sample_weights must be 'equal_slice', 'size_proportional', or a positive sequence."
        )
    else:
        value = np.asarray(sample_weights, dtype=np.float64)
        if value.shape != n.shape:
            raise ValueError(f"sample_weights must have length {len(rows)}.")
    if np.any(~np.isfinite(value)) or np.any(value <= 0):
        raise ValueError("Sample weights must be finite and strictly positive.")
    return value / value.sum()


def _load_x(row: dict) -> Union[csr_matrix, np.ndarray]:
    """Load one expression matrix without forcing a dense matrix into CSR."""
    x_format = row.get("x_format")
    if x_format == "dense_npy" or (x_format is None and str(row["x_path"]).endswith(".npy")):
        # Keep float32 dense storage memory-mapped.  Matrix products promote
        # against the float64 solver/loadings as needed, without a standalone
        # full-section float64 expression copy.
        return np.load(row["x_path"], mmap_mode="r")
    return load_npz(row["x_path"]).tocsr().astype(np.float64, copy=False)


def _streaming_shared_initialization(
    rows: Sequence[dict],
    means: Sequence[np.ndarray],
    weights: np.ndarray,
    k: int,
    seed: int,
) -> np.ndarray:
    p = int(rows[0]["n_genes"])
    rank = min(p, k + 10)
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((p, rank)), mode="reduced")
    covariance_weights = weights / np.asarray(
        [row["n_cells"] for row in rows], dtype=np.float64
    )
    covariance_weights /= covariance_weights.sum()
    for _ in range(2):
        b = np.zeros((p, rank), dtype=np.float64)
        for row, mean, weight in zip(rows, means, covariance_weights):
            x = _load_x(row)
            xq = _centered_matmul(x, mean, q)
            b += float(weight) * _centered_t_matmul(x, mean, xq)
            del x, xq
        q, _ = np.linalg.qr(b, mode="reduced")
    small = np.zeros((rank, rank), dtype=np.float64)
    for row, mean, weight in zip(rows, means, covariance_weights):
        x = _load_x(row)
        xq = _centered_matmul(x, mean, q)
        small += float(weight) * (xq.T @ xq)
        del x, xq
    _, vectors = np.linalg.eigh(small)
    w, _ = np.linalg.qr(q @ vectors[:, ::-1][:, :k], mode="reduced")
    return w[:, :k]


def _resolve_group_labels(
    group_labels: Optional[Sequence[Any]], n_sections: int
) -> Tuple[np.ndarray, List[str]]:
    """Encode section groups in first-occurrence order.

    ``None`` is the original one-global-centre model.  Explicit labels enable
    partial pooling: each section shrinks toward the loading centre of its
    declared group, while all centres retain one common rotational gauge.
    """
    if group_labels is None:
        return np.zeros(n_sections, dtype=np.int64), ["global"]
    if isinstance(group_labels, (str, bytes)) or len(group_labels) != n_sections:
        raise ValueError(f"group_labels must have length {n_sections}.")
    labels: List[str] = []
    lookup: dict[str, int] = {}
    codes = np.empty(n_sections, dtype=np.int64)
    for index, label in enumerate(group_labels):
        key = str(label)
        if key not in lookup:
            lookup[key] = len(labels)
            labels.append(key)
        codes[index] = lookup[key]
    return codes, labels


def _update_group_loadings(
    sample_loadings: np.ndarray,
    contributions: np.ndarray,
    rhos: np.ndarray,
    group_codes: np.ndarray,
    fallback: np.ndarray,
) -> np.ndarray:
    centres = np.empty_like(fallback)
    for group in range(fallback.shape[0]):
        members = np.flatnonzero(group_codes == group)
        centres[group] = _update_global_loading(
            sample_loadings[members],
            contributions[members],
            rhos[members],
            fallback=fallback[group],
        )
    return centres


def _relative_array_change(current: np.ndarray, previous: np.ndarray) -> float:
    """Frobenius-relative change for a loading matrix or a stack of centres."""
    return float(
        np.linalg.norm(np.asarray(current) - np.asarray(previous))
        / (np.linalg.norm(np.asarray(previous)) + 1e-12)
    )


def _checkpoint_paths(checkpoint: Path, rows: Sequence[dict]) -> List[Path]:
    return [checkpoint / "embeddings" / f"{row['safe_name']}_Z.npy" for row in rows]


def _activate_checkpoint(
    checkpoint_root: Path, stage: Path, label: str
) -> Path:
    """Publish a complete staged checkpoint by atomically replacing a pointer."""
    pointer = checkpoint_root / "current.json"
    previous = None
    if pointer.exists():
        previous = json.loads(pointer.read_text()).get("checkpoint")
    destination = checkpoint_root / label
    if destination.exists():
        raise FileExistsError(f"Checkpoint already exists: {destination}")
    os.replace(stage, destination)
    _write_json_atomic(pointer, {"format": "graphpca_atomic_checkpoint_v1", "checkpoint": label})
    if previous and previous != label:
        old = checkpoint_root / str(previous)
        if old.exists():
            # Cleanup is non-critical. Some removable/exFAT volumes expose
            # transient macOS ``._*`` sidecars while directories are removed.
            # The new checkpoint is already selected by the atomic pointer.
            shutil.rmtree(old, ignore_errors=True)
    return destination


def _checkpoint_state(
    *,
    iteration: int,
    objective_values: Sequence[float],
    stable: int,
    z_changes: Sequence[float],
    ws_changes: Sequence[float],
    w0_changes: Sequence[float],
    rel_changes: Sequence[float],
    objective_rel: Sequence[float],
    inner_used: Sequence[int],
    pcg_status: Sequence[Sequence[int]],
    model_signature: Mapping[str, Any],
    converged: Optional[bool] = None,
    convergence_reason: Optional[str] = None,
) -> dict:
    state: dict[str, Any] = {
        "completed_iteration": int(iteration),
        "objective_values": list(objective_values),
        "stable_count": int(stable),
        "z_changes": list(z_changes),
        "ws_changes": list(ws_changes),
        "w0_changes": list(w0_changes),
        "relative_changes": list(rel_changes),
        "objective_relative_changes": list(objective_rel),
        "loading_inner_iterations": list(inner_used),
        "pcg_status": list(pcg_status),
        "model_signature": dict(model_signature),
    }
    if converged is not None:
        state["converged"] = bool(converged)
    if convergence_reason is not None:
        state["convergence_reason"] = convergence_reason
    return state


def run_hierarchical_multi_gpca_out_of_core(
    section_store: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    n_components: int = 50,
    lambdas: Union[float, Sequence[float]] = 0.5,
    rhos: Union[float, Sequence[float]] = 1.0,
    sample_weights: Optional[Union[str, Sequence[float]]] = "equal_slice",
    group_labels: Optional[Sequence[Any]] = None,
    center: bool = True,
    pcg_tol: float = 1e-6,
    pcg_max_iter: int = 500,
    outer_tol: float = 1e-6,
    objective_tol: float = 1e-8,
    objective_patience: int = 3,
    relaxed_outer_factor: float = 10.0,
    loading_inner_max_iter: int = 10,
    loading_inner_tol: float = 1e-6,
    max_iter: int = 20,
    random_seed: int = 666,
    mode: str = "iterative",
    resume: bool = False,
    return_info: bool = False,
):
    """Fit hierarchical GraphPCA while keeping only one section in RAM.

    ``output_dir`` persists embeddings and restart state. The returned first
    object is :class:`DiskBackedEmbeddings`; call ``.load(i)`` to materialize a
    single section. ``return_log`` and dense reconstruction are intentionally
    unavailable in this mode because they defeat its memory contract.  When
    ``group_labels`` is supplied, sections shrink toward a group-specific
    shared loading; the returned loading is then an array of shape
    ``(n_groups, n_genes, n_components)`` in first-occurrence group order.
    """
    rows = _read_manifest(section_store)
    if len({int(row["n_genes"]) for row in rows}) != 1:
        raise ValueError("All stored sections must have the same gene count.")
    p, s = int(rows[0]["n_genes"]), len(rows)
    if n_components < 1 or n_components > p:
        raise ValueError("n_components must be between 1 and the gene count.")
    if mode not in {"iterative", "accelerated"}:
        raise ValueError("mode must be 'iterative' or 'accelerated'.")
    if min(pcg_tol, outer_tol, loading_inner_tol) <= 0 or objective_tol < 0:
        raise ValueError("Solver and convergence tolerances must be valid.")
    if relaxed_outer_factor < 1:
        raise ValueError("relaxed_outer_factor must be at least 1.")
    if min(pcg_max_iter, max_iter, objective_patience, loading_inner_max_iter) < 1:
        raise ValueError("Iteration limits and patience must be positive.")
    use_cpp = mode == "accelerated" and HAS_CPP_MULTI
    if mode == "accelerated" and not use_cpp:
        import warnings

        warnings.warn(
            "The C++ Z-only PCG kernel is unavailable; using SciPy PCG.",
            RuntimeWarning,
            stacklevel=2,
        )
    contributions = _contributions(rows, sample_weights)
    lambdas_array = np.broadcast_to(np.asarray(lambdas, dtype=np.float64), (s,)).copy()
    if np.any(~np.isfinite(lambdas_array)) or np.any(lambdas_array < 0):
        raise ValueError("All lambdas must be finite and non-negative.")
    rhos_array = _resolve_rhos(rhos, s)
    group_codes, group_names = _resolve_group_labels(group_labels, s)
    model_signature = {
        "section_names": [row["name"] for row in rows],
        "section_shapes": [[int(row["n_cells"]), int(row["n_genes"])] for row in rows],
        "n_components": int(n_components),
        "lambdas": lambdas_array,
        "rhos": rhos_array,
        "sample_contributions": contributions,
        "group_labels": group_names if group_labels is not None else None,
        "section_group_codes": group_codes,
        "center": bool(center),
    }
    means = [
        np.load(row["mean_path"]).astype(np.float64) if center else np.zeros(p)
        for row in rows
    ]
    centered_norms = np.asarray(
        [
            float(row["sum_sq"]) - (float(row["n_cells"]) * float(mean @ mean))
            if center
            else float(row["sum_sq"])
            for row, mean in zip(rows, means)
        ]
    )
    run_started = time.time()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_root = root / "checkpoints"
    checkpoint_root.mkdir(exist_ok=True)
    pointer_path = checkpoint_root / "current.json"
    # An unselected stage is never part of a published run and can safely be
    # discarded before a resume.  The selected checkpoint is identified only
    # by ``current.json``.
    for orphan in checkpoint_root.glob(".stage_*"):
        shutil.rmtree(orphan, ignore_errors=True)
    if resume and (root / "state" / "state.json").exists() and not pointer_path.exists():
        raise ValueError(
            "This output directory uses the pre-v2.2 checkpoint layout and cannot "
            "be resumed safely. Please start a new output_dir."
        )
    has_resume = resume and pointer_path.exists()
    if has_resume:
        pointer = json.loads(pointer_path.read_text())
        checkpoint = checkpoint_root / str(pointer.get("checkpoint", ""))
        state_path = checkpoint / "state" / "state.json"
        if not state_path.exists():
            raise FileNotFoundError("The selected checkpoint has no state.json.")
        state = json.loads(state_path.read_text())
        if state.get("model_signature") != json.loads(
            json.dumps(model_signature, default=_json_default)
        ):
            raise ValueError(
                "The requested model configuration differs from the saved state; "
                "use a new output_dir rather than resume."
            )
        paths = _checkpoint_paths(checkpoint, rows)
        if not all(path.exists() for path in paths):
            raise FileNotFoundError(
                "Cannot resume because one or more disk-backed embedding files are missing."
            )
        centres = np.load(checkpoint / "state" / "centres.npy")
        ws = np.load(checkpoint / "state" / "Ws.npy")
        if centres.shape != (len(group_names), p, n_components):
            raise ValueError("Saved group loading centres have an incompatible shape.")
        start = int(state["completed_iteration"]) + 1
        objective_values = [float(x) for x in state["objective_values"]]
        stable = int(state["stable_count"])
    else:
        if pointer_path.exists():
            raise FileExistsError(
                f"Checkpoint state exists in {checkpoint_root}; choose a new output_dir or resume=True."
            )
        initial = _streaming_shared_initialization(
            rows, means, contributions, n_components, random_seed
        )
        centres = np.repeat(initial[None, :, :], len(group_names), axis=0)
        ws = np.asarray([centres[group] for group in group_codes])
        stage = checkpoint_root / ".stage_initial"
        stage.mkdir(exist_ok=False)
        (stage / "embeddings").mkdir()
        for row, path in zip(rows, _checkpoint_paths(stage, rows)):
            np.save(
                path, np.zeros((int(row["n_cells"]), n_components), dtype=np.float32)
            )
        state_stage = stage / "state"
        state_stage.mkdir()
        np.save(state_stage / "centres.npy", centres)
        np.save(state_stage / "Ws.npy", ws)
        _write_json(
            state_stage / "state.json",
            _checkpoint_state(
                iteration=0,
                objective_values=[],
                stable=0,
                z_changes=[],
                ws_changes=[],
                w0_changes=[],
                rel_changes=[],
                objective_rel=[],
                inner_used=[],
                pcg_status=[[] for _ in rows],
                model_signature=model_signature,
            ),
        )
        checkpoint = _activate_checkpoint(checkpoint_root, stage, "iteration_0000")
        paths = _checkpoint_paths(checkpoint, rows)
        start, objective_values, stable = 1, [], 0
    if has_resume:
        z_changes = [float(x) for x in state.get("z_changes", [])]
        ws_changes = [float(x) for x in state.get("ws_changes", [])]
        w0_changes = [float(x) for x in state.get("w0_changes", [])]
        rel_changes = [float(x) for x in state.get("relative_changes", [])]
        objective_rel = [float(x) for x in state.get("objective_relative_changes", [])]
        inner_used = [int(x) for x in state.get("loading_inner_iterations", [])]
        pcg_status = state.get("pcg_status", [[] for _ in rows])
    else:
        z_changes = []
        ws_changes = []
        w0_changes = []
        rel_changes = []
        objective_rel = []
        inner_used = []
        pcg_status = [[] for _ in rows]
    converged = bool(state.get("converged", False)) if has_resume else False
    reason = (
        str(state.get("convergence_reason", "maximum_iterations"))
        if converged
        else "maximum_iterations"
    )
    iterations = int(state["completed_iteration"]) if converged else max_iter
    if converged:
        # A published converged final checkpoint is immutable; reopening it is
        # a read-only operation rather than an extra optimisation iteration.
        start = max_iter + 1
    for iteration in range(start, max_iter + 1):
        old_ws, old_centres = ws.copy(), centres.copy()
        stage = checkpoint_root / f".stage_{iteration:04d}"
        stage.mkdir(exist_ok=False)
        (stage / "embeddings").mkdir()
        stage_paths = _checkpoint_paths(stage, rows)
        crosses: List[np.ndarray] = []
        z_norms = np.zeros(s)
        graph_terms = np.zeros(s)
        for index, (row, mean, lam, path, stage_path) in enumerate(
            zip(rows, means, lambdas_array, paths, stage_paths)
        ):
            x, graph = _load_x(row), load_npz(row["graph_path"]).tocsr().astype(
                np.float64
            )
            phi = _build_phi(int(row["n_cells"]), None, graph, None, None, float(lam))
            old_z = np.load(path).astype(np.float64)
            new_z, status = _solve_embedding(
                x, mean, phi, ws[index], old_z, pcg_tol, pcg_max_iter, use_cpp=use_cpp
            )
            pcg_status[index] = status
            cross = _centered_t_matmul(x, mean, new_z) / float(row["n_cells"])
            crosses.append(cross)
            z_norms[index] = float(np.sum(new_z * new_z))
            graph_terms[index] = max(
                float(np.sum(new_z * (phi @ new_z))) - z_norms[index], 0.0
            )
            np.save(stage_path, new_z.astype(np.float32))
            del x, graph, phi, old_z, new_z
        for inner in range(loading_inner_max_iter):
            next_ws = np.asarray(
                [
                    _update_sample_loading_from_cross(cross, current_centre, rho)
                    for cross, rho, current_centre in zip(
                        crosses, rhos_array, centres[group_codes]
                    )
                ]
            )
            next_centres = _update_group_loadings(
                next_ws, contributions, rhos_array, group_codes, fallback=centres
            )
            change = max(
                float(np.linalg.norm(next_ws - ws) / (np.linalg.norm(ws) + 1e-12)),
                _relative_array_change(next_centres, centres),
            )
            ws, centres = next_ws, next_centres
            if change < loading_inner_tol:
                break
        inner_used.append(inner + 1)
        rotation = (
            _polar_factor(
                sum(
                    centre.T @ old_centre
                    for centre, old_centre in zip(centres, old_centres)
                ),
                fallback=np.eye(n_components),
            )
            if np.any(rhos_array > 0)
            else np.eye(n_components)
        )
        ws, centres = ws @ rotation, centres @ rotation
        crosses = [cross @ rotation for cross in crosses]
        z_num = z_den = 0.0
        for row, path, stage_path in zip(rows, paths, stage_paths):
            old_z = np.load(path, mmap_mode="r")
            current = np.load(stage_path)
            aligned = np.asarray(current, dtype=np.float64) @ rotation
            z_num += float(np.sum((aligned - old_z) ** 2))
            z_den += float(np.sum(old_z**2)) + 1e-30
            np.save(stage_path, aligned.astype(np.float32))
        objective = 0.0
        for index, row in enumerate(rows):
            reconstruction = (
                centered_norms[index]
                + z_norms[index]
                - 2.0
                * float(row["n_cells"])
                * float(np.trace(ws[index].T @ crosses[index]))
            )
            shrinkage = float(rhos_array[index]) * float(
                np.sum((ws[index] - centres[group_codes[index]]) ** 2)
            )
            objective += float(contributions[index]) * (
                (reconstruction + graph_terms[index]) / float(row["n_cells"])
                + shrinkage
            )
        prior = objective_values[-1] if objective_values else np.nan
        obj_change = (
            np.inf
            if not np.isfinite(prior)
            else abs(objective - prior) / max(abs(prior), 1e-30)
        )
        objective_values.append(float(objective))
        objective_rel.append(float(obj_change))
        z_change = float(np.sqrt(z_num / z_den))
        ws_change = _relative_loading_change(list(ws), list(old_ws), contributions)
        w0_change = _relative_array_change(centres, old_centres)
        joint = (
            max(z_change, ws_change, w0_change)
            if np.any(rhos_array > 0)
            else max(z_change, ws_change)
        )
        z_changes.append(z_change)
        ws_changes.append(ws_change)
        w0_changes.append(w0_change)
        rel_changes.append(joint)
        stable = (
            stable + 1
            if obj_change < objective_tol and joint < relaxed_outer_factor * outer_tol
            else 0
        )
        state_stage = stage / "state"
        state_stage.mkdir()
        np.save(state_stage / "centres.npy", centres)
        np.save(state_stage / "Ws.npy", ws)
        _write_json(
            state_stage / "state.json",
            _checkpoint_state(
                iteration=iteration,
                objective_values=objective_values,
                stable=stable,
                z_changes=z_changes,
                ws_changes=ws_changes,
                w0_changes=w0_changes,
                rel_changes=rel_changes,
                objective_rel=objective_rel,
                inner_used=inner_used,
                pcg_status=pcg_status,
                model_signature=model_signature,
            ),
        )
        checkpoint = _activate_checkpoint(
            checkpoint_root, stage, f"iteration_{iteration:04d}"
        )
        paths = _checkpoint_paths(checkpoint, rows)
        if stable >= objective_patience:
            converged, iterations, reason = (
                True,
                iteration,
                "objective_plateau_with_bounded_aligned_change",
            )
            break
        gc.collect()
    needs_final_orientation = not (
        has_resume and checkpoint.name.startswith("final_") and start > max_iter
    )
    if needs_final_orientation:
        rotation = np.eye(n_components)
        if np.any(rhos_array > 0):
            # Preserve a deterministic common orientation without loading all Z.
            gram = np.zeros((n_components, n_components), dtype=np.float64)
            for row, path, weight in zip(rows, paths, contributions):
                z = np.load(path, mmap_mode="r")
                gram += float(weight) * (z.T @ z) / float(row["n_cells"])
            _, rotation = np.linalg.eigh(0.5 * (gram + gram.T))
            rotation = rotation[:, ::-1]
            ws, centres = ws @ rotation, centres @ rotation
        stage = checkpoint_root / ".stage_final"
        stage.mkdir(exist_ok=False)
        (stage / "embeddings").mkdir()
        final_paths = _checkpoint_paths(stage, rows)
        for path, final_path in zip(paths, final_paths):
            z = np.load(path, mmap_mode="r")
            np.save(
                final_path, (np.asarray(z, dtype=np.float64) @ rotation).astype(np.float32)
            )
        state_stage = stage / "state"
        state_stage.mkdir()
        np.save(state_stage / "centres.npy", centres)
        np.save(state_stage / "Ws.npy", ws)
        _write_json(
            state_stage / "state.json",
            _checkpoint_state(
                iteration=iterations,
                objective_values=objective_values,
                stable=stable,
                z_changes=z_changes,
                ws_changes=ws_changes,
                w0_changes=w0_changes,
                rel_changes=rel_changes,
                objective_rel=objective_rel,
                inner_used=inner_used,
                pcg_status=pcg_status,
                model_signature=model_signature,
                converged=converged,
                convergence_reason=reason,
            ),
        )
        checkpoint = _activate_checkpoint(
            checkpoint_root, stage, f"final_{iterations:04d}"
        )
        paths = _checkpoint_paths(checkpoint, rows)
    deviations = np.asarray(
        [np.linalg.norm(item - centres[group]) for item, group in zip(ws, group_codes)],
        dtype=np.float64,
    )
    subspace = np.asarray(
        [
            np.linalg.norm(
                item @ item.T - centres[group] @ centres[group].T
            )
            / np.sqrt(2 * n_components)
            for item, group in zip(ws, group_codes)
        ]
    )
    cosines = np.asarray(
        [np.diag(item.T @ centres[group]) for item, group in zip(ws, group_codes)]
    )
    info = HierarchicalMultiGPCAInfo(
        converged,
        iterations,
        rel_changes,
        z_changes,
        ws_changes,
        w0_changes,
        objective_values,
        objective_rel,
        contributions,
        lambdas_array,
        rhos_array,
        means,
        pcg_status,
        deviations,
        subspace,
        cosines,
        np.full(s, np.nan),
        reason,
        stable,
        [np.nan] * len(rel_changes),
        [np.nan] * len(rel_changes),
        [np.nan] * len(rel_changes),
        [np.nan] * len(rel_changes),
        inner_used,
    )
    _write_json(
        root / "run_summary.json",
        {
            "converged": converged,
            "iterations": iterations,
            "final_objective": objective_values[-1],
            "convergence_reason": reason,
            "section_store": str(section_store),
            "group_names": group_names,
            "checkpoint": str(checkpoint),
            "elapsed_seconds": time.time() - run_started,
        },
    )
    returned_centres = centres[0] if group_labels is None else centres
    output: List[Any] = [
        DiskBackedEmbeddings(paths, [row["name"] for row in rows]),
        returned_centres,
        list(ws),
    ]
    if return_info:
        output.append(info)
    return tuple(output)


Run_Hierarchical_Multi_GPCA_OutOfCore = run_hierarchical_multi_gpca_out_of_core
