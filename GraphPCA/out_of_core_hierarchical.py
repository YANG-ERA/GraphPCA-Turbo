"""Disk-backed execution mode for hierarchical multi-sample GraphPCA.

The helpers in this module are deliberately separate from AnnData I/O.  A
caller may yield one expression matrix and graph at a time while creating a
section store, so peak RAM does not scale with the number of sections.
"""

from __future__ import annotations

import gc
import json
import re
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
    _relative_matrix_change,
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
    """
    root = Path(store_root)
    manifest_path = root / "manifest.json"
    if root.exists() and any(root.iterdir()) and not manifest_path.exists():
        raise FileExistsError(
            f"Refusing to write a section store into a nonempty unmanaged directory: {root}"
        )
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"Section store already exists: {root}")
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
        x_path, graph_path, mean_path = (
            directory / "X.npz",
            directory / "graph.npz",
            directory / "mean.npy",
        )
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
            "format": "graphpca_hierarchical_section_store_v1",
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


def _load_x(row: dict) -> csr_matrix:
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


def run_hierarchical_multi_gpca_out_of_core(
    section_store: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    n_components: int = 50,
    lambdas: Union[float, Sequence[float]] = 0.5,
    rhos: Union[float, Sequence[float]] = 1.0,
    sample_weights: Optional[Union[str, Sequence[float]]] = "equal_slice",
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
    unavailable in this mode because they defeat its memory contract.
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
    model_signature = {
        "section_names": [row["name"] for row in rows],
        "section_shapes": [[int(row["n_cells"]), int(row["n_genes"])] for row in rows],
        "n_components": int(n_components),
        "lambdas": lambdas_array,
        "rhos": rhos_array,
        "sample_contributions": contributions,
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
    state_dir, z_dir = root / "state", root / "embeddings"
    state_path = state_dir / "state.json"
    root.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(exist_ok=True)
    z_dir.mkdir(exist_ok=True)
    paths = [z_dir / f"{row['safe_name']}_Z.npy" for row in rows]
    if resume and state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("model_signature") != json.loads(
            json.dumps(model_signature, default=_json_default)
        ):
            raise ValueError(
                "The requested model configuration differs from the saved state; "
                "use a new output_dir rather than resume."
            )
        if not all(path.exists() for path in paths):
            raise FileNotFoundError(
                "Cannot resume because one or more disk-backed embedding files are missing."
            )
        w0, ws = np.load(state_dir / "W0.npy"), np.load(state_dir / "Ws.npy")
        start = int(state["completed_iteration"]) + 1
        objective_values = [float(x) for x in state["objective_values"]]
        stable = int(state["stable_count"])
    else:
        if any(path.exists() for path in paths):
            raise FileExistsError(
                f"Embedding state exists in {z_dir}; choose a new output_dir or resume=True."
            )
        w0 = _streaming_shared_initialization(
            rows, means, contributions, n_components, random_seed
        )
        ws = np.repeat(w0[None, :, :], s, axis=0)
        for row, path in zip(rows, paths):
            np.save(
                path, np.zeros((int(row["n_cells"]), n_components), dtype=np.float32)
            )
        start, objective_values, stable = 1, [], 0
    if resume and state_path.exists():
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
    converged, reason, iterations = False, "maximum_iterations", max_iter
    for iteration in range(start, max_iter + 1):
        old_ws, old_w0 = ws.copy(), w0.copy()
        stage = root / f".stage_{iteration:04d}"
        stage.mkdir(exist_ok=False)
        crosses: List[np.ndarray] = []
        z_norms = np.zeros(s)
        graph_terms = np.zeros(s)
        for index, (row, mean, lam, path) in enumerate(
            zip(rows, means, lambdas_array, paths)
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
            np.save(stage / path.name, new_z.astype(np.float32))
            del x, graph, phi, old_z, new_z
        for inner in range(loading_inner_max_iter):
            next_ws = np.asarray(
                [
                    _update_sample_loading_from_cross(cross, w0, rho)
                    for cross, rho in zip(crosses, rhos_array)
                ]
            )
            next_w0 = _update_global_loading(
                next_ws, contributions, rhos_array, fallback=w0
            )
            change = max(
                float(np.linalg.norm(next_ws - ws) / (np.linalg.norm(ws) + 1e-12)),
                _relative_matrix_change(next_w0, w0),
            )
            ws, w0 = next_ws, next_w0
            if change < loading_inner_tol:
                break
        inner_used.append(inner + 1)
        rotation = (
            _polar_factor(w0.T @ old_w0, fallback=np.eye(n_components))
            if np.any(rhos_array > 0)
            else np.eye(n_components)
        )
        ws, w0 = ws @ rotation, w0 @ rotation
        crosses = [cross @ rotation for cross in crosses]
        z_num = z_den = 0.0
        for row, path in zip(rows, paths):
            old_z = np.load(path, mmap_mode="r")
            current = np.load(stage / path.name)
            aligned = np.asarray(current, dtype=np.float64) @ rotation
            z_num += float(np.sum((aligned - old_z) ** 2))
            z_den += float(np.sum(old_z**2)) + 1e-30
            np.save(stage / path.name, aligned.astype(np.float32))
        for path in paths:
            (stage / path.name).replace(path)
        stage.rmdir()
        objective = 0.0
        for index, row in enumerate(rows):
            reconstruction = (
                centered_norms[index]
                + z_norms[index]
                - 2.0
                * float(row["n_cells"])
                * float(np.trace(ws[index].T @ crosses[index]))
            )
            shrinkage = float(rhos_array[index]) * float(np.sum((ws[index] - w0) ** 2))
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
        w0_change = _relative_matrix_change(w0, old_w0)
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
        _write_json(
            state_path,
            {
                "completed_iteration": iteration,
                "objective_values": objective_values,
                "stable_count": stable,
                "z_changes": z_changes,
                "ws_changes": ws_changes,
                "w0_changes": w0_changes,
                "relative_changes": rel_changes,
                "objective_relative_changes": objective_rel,
                "loading_inner_iterations": inner_used,
                "pcg_status": pcg_status,
                "model_signature": model_signature,
            },
        )
        np.save(state_dir / "W0.npy", w0)
        np.save(state_dir / "Ws.npy", ws)
        if stable >= objective_patience:
            converged, iterations, reason = (
                True,
                iteration,
                "objective_plateau_with_bounded_aligned_change",
            )
            break
        gc.collect()
    if np.any(rhos_array > 0):
        # Preserve a deterministic common orientation without loading all Z.
        gram = np.zeros((n_components, n_components), dtype=np.float64)
        for row, path, weight in zip(rows, paths, contributions):
            z = np.load(path, mmap_mode="r")
            gram += float(weight) * (z.T @ z) / float(row["n_cells"])
        _, rotation = np.linalg.eigh(0.5 * (gram + gram.T))
        rotation = rotation[:, ::-1]
        ws, w0 = ws @ rotation, w0 @ rotation
        for path in paths:
            z = np.load(path, mmap_mode="r")
            np.save(
                path, (np.asarray(z, dtype=np.float64) @ rotation).astype(np.float32)
            )
        np.save(state_dir / "W0.npy", w0)
        np.save(state_dir / "Ws.npy", ws)
    deviations = np.asarray(
        [np.linalg.norm(item - w0) for item in ws], dtype=np.float64
    )
    subspace = np.asarray(
        [
            np.linalg.norm(item @ item.T - w0 @ w0.T) / np.sqrt(2 * n_components)
            for item in ws
        ]
    )
    cosines = np.asarray([np.diag(item.T @ w0) for item in ws])
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
        state_path,
        {
            "completed_iteration": iterations,
            "objective_values": objective_values,
            "stable_count": stable,
            "z_changes": z_changes,
            "ws_changes": ws_changes,
            "w0_changes": w0_changes,
            "relative_changes": rel_changes,
            "objective_relative_changes": objective_rel,
            "loading_inner_iterations": inner_used,
            "pcg_status": pcg_status,
            "model_signature": model_signature,
            "converged": converged,
            "convergence_reason": reason,
        },
    )
    _write_json(
        root / "run_summary.json",
        {
            "converged": converged,
            "iterations": iterations,
            "final_objective": objective_values[-1],
            "convergence_reason": reason,
            "section_store": str(section_store),
            "elapsed_seconds": time.time() - run_started,
        },
    )
    output: List[Any] = [
        DiskBackedEmbeddings(paths, [row["name"] for row in rows]),
        w0,
        list(ws),
    ]
    if return_info:
        output.append(info)
    return tuple(output)


Run_Hierarchical_Multi_GPCA_OutOfCore = run_hierarchical_multi_gpca_out_of_core
