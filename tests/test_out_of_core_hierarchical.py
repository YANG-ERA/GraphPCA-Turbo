from pathlib import Path

import numpy as np
import pytest
from scipy import sparse


def _sections():
    rng = np.random.default_rng(17)
    for index, n_obs in enumerate((9, 11, 8)):
        expression = sparse.csr_matrix(rng.poisson(1.5, (n_obs, 6)).astype(np.float32))
        graph = sparse.diags(
            [np.ones(n_obs - 1), np.ones(n_obs - 1)],
            [-1, 1],
            shape=(n_obs, n_obs),
            format="csr",
        )
        yield f"section {index}", expression, graph


def test_out_of_core_hierarchical_public_api(tmp_path):
    from GraphPCA import (
        DiskBackedEmbeddings,
        Run_Hierarchical_Multi_GPCA,
        create_hierarchical_section_store,
    )

    store = create_hierarchical_section_store(tmp_path / "store", _sections())
    embeddings, w0, sample_loadings, info = Run_Hierarchical_Multi_GPCA(
        adatas=None,
        execution_mode="out_of_core",
        section_store=store,
        out_of_core_output_dir=tmp_path / "fit",
        n_components=2,
        lambdas=0.2,
        rhos=0.5,
        init_strategy="shared",
        mode="iterative",
        max_iter=4,
        objective_tol=1e9,
        outer_tol=1e9,
        objective_patience=2,
        return_info=True,
    )

    assert isinstance(embeddings, DiskBackedEmbeddings)
    assert embeddings.names == ("section 0", "section 1", "section 2")
    assert embeddings[1].shape == (11, 2)
    assert embeddings.load(2, mmap_mode="r").shape == (8, 2)
    assert w0.shape == (6, 2)
    assert sample_loadings[0].shape == (6, 2)
    assert info.converged
    assert info.n_iter == 3
    assert (tmp_path / "fit" / "run_summary.json").exists()


def test_out_of_core_rejects_preloaded_adatas(tmp_path):
    from GraphPCA import Run_Hierarchical_Multi_GPCA, create_hierarchical_section_store

    store = create_hierarchical_section_store(tmp_path / "store", _sections())
    with pytest.raises(ValueError, match="adatas=None"):
        Run_Hierarchical_Multi_GPCA(
            adatas=[object()],
            execution_mode="out_of_core",
            section_store=store,
            out_of_core_output_dir=tmp_path / "fit",
        )


def test_section_store_paths_are_portable(tmp_path):
    from GraphPCA import create_hierarchical_section_store
    from GraphPCA.out_of_core_hierarchical import _read_manifest

    store = create_hierarchical_section_store(tmp_path / "store", _sections())
    rows = _read_manifest(Path(store))
    assert all(Path(row["x_path"]).is_absolute() for row in rows)
    manifest_text = (Path(store) / "manifest.json").read_text()
    assert str(Path(store)) not in manifest_text


def test_out_of_core_resume_validates_configuration(tmp_path):
    from GraphPCA import Run_Hierarchical_Multi_GPCA, create_hierarchical_section_store

    store = create_hierarchical_section_store(tmp_path / "store", _sections())
    options = dict(
        adatas=None,
        execution_mode="out_of_core",
        section_store=store,
        out_of_core_output_dir=tmp_path / "fit",
        n_components=2,
        lambdas=0.2,
        rhos=0.5,
        init_strategy="shared",
        mode="iterative",
        outer_tol=1e9,
        objective_tol=1e9,
        objective_patience=2,
        return_info=True,
    )
    _, _, _, first_info = Run_Hierarchical_Multi_GPCA(max_iter=1, **options)
    assert not first_info.converged
    _, _, _, resumed_info = Run_Hierarchical_Multi_GPCA(
        max_iter=4, out_of_core_resume=True, **options
    )
    assert resumed_info.converged
    assert resumed_info.n_iter == 3
    with pytest.raises(ValueError, match="differs from the saved state"):
        Run_Hierarchical_Multi_GPCA(
            max_iter=4,
            out_of_core_resume=True,
            lambdas=0.3,
            **{key: value for key, value in options.items() if key != "lambdas"},
        )


def test_dense_expression_store_and_group_centres(tmp_path):
    from GraphPCA import Run_Hierarchical_Multi_GPCA, create_hierarchical_section_store

    dense_sections = [
        (name, expression.toarray(), graph) for name, expression, graph in _sections()
    ]
    store = create_hierarchical_section_store(
        tmp_path / "dense_store",
        dense_sections,
        expression_storage="auto",
        dense_threshold=0.0,
    )
    manifest = (store / "manifest.json").read_text()
    assert '"x_format": "dense_npy"' in manifest
    embeddings, centres, ws, info = Run_Hierarchical_Multi_GPCA(
        adatas=None,
        execution_mode="out_of_core",
        section_store=store,
        out_of_core_output_dir=tmp_path / "fit",
        out_of_core_group_labels=["early", "early", "late"],
        n_components=2,
        lambdas=0.2,
        rhos=0.5,
        init_strategy="shared",
        max_iter=3,
        objective_tol=1e9,
        outer_tol=1e9,
        objective_patience=2,
        return_info=True,
    )
    assert centres.shape == (2, 6, 2)
    assert len(ws) == len(embeddings) == 3
    assert np.isfinite(embeddings[0]).all()
    assert np.isfinite(info.objective_values).all()
    pointer = (tmp_path / "fit" / "checkpoints" / "current.json").read_text()
    assert '"checkpoint": "final_' in pointer


def test_resume_discards_unpublished_stage_and_reuses_atomic_checkpoint(tmp_path):
    from GraphPCA import Run_Hierarchical_Multi_GPCA, create_hierarchical_section_store

    store = create_hierarchical_section_store(tmp_path / "store", _sections())
    options = dict(
        adatas=None,
        execution_mode="out_of_core",
        section_store=store,
        out_of_core_output_dir=tmp_path / "fit",
        n_components=2,
        lambdas=0.2,
        rhos=0.5,
        init_strategy="shared",
        outer_tol=1e9,
        objective_tol=1e9,
        objective_patience=2,
        return_info=True,
    )
    Run_Hierarchical_Multi_GPCA(max_iter=1, **options)
    interrupted = tmp_path / "fit" / "checkpoints" / ".stage_interrupted"
    interrupted.mkdir()
    (interrupted / "partial.bin").write_bytes(b"incomplete")
    _, _, _, info = Run_Hierarchical_Multi_GPCA(
        max_iter=4, out_of_core_resume=True, **options
    )
    assert info.converged
    assert not interrupted.exists()
    # A completed final snapshot can also be reopened without rewriting it.
    _, _, _, repeated = Run_Hierarchical_Multi_GPCA(
        max_iter=4, out_of_core_resume=True, **options
    )
    assert repeated.n_iter == info.n_iter


def test_independent_mode_also_publishes_a_final_checkpoint(tmp_path):
    from GraphPCA import Run_Hierarchical_Multi_GPCA, create_hierarchical_section_store

    store = create_hierarchical_section_store(tmp_path / "store", _sections())
    options = dict(
        adatas=None,
        execution_mode="out_of_core",
        section_store=store,
        out_of_core_output_dir=tmp_path / "fit",
        n_components=2,
        lambdas=0.2,
        rhos=0.0,
        init_strategy="shared",
        max_iter=1,
        return_info=True,
    )
    _, w0, _, first = Run_Hierarchical_Multi_GPCA(**options)
    assert w0.shape == (6, 2)
    _, _, _, resumed = Run_Hierarchical_Multi_GPCA(
        out_of_core_resume=True, **options
    )
    assert resumed.n_iter == first.n_iter
    assert '"checkpoint": "final_0001"' in (
        tmp_path / "fit" / "checkpoints" / "current.json"
    ).read_text()
