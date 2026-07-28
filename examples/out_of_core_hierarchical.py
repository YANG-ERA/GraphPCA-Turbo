"""Minimal runnable example for disk-backed hierarchical GraphPCA.

Replace ``synthetic_sections`` with a generator that loads one prepared
cells-by-genes matrix and its spatial graph from disk at a time.  The section
store is reusable across fits with different model hyperparameters.
"""

from pathlib import Path
import tempfile

import numpy as np
from scipy import sparse

from GraphPCA import Run_Hierarchical_Multi_GPCA, create_hierarchical_section_store


def synthetic_sections():
    rng = np.random.default_rng(666)
    for index, n_cells in enumerate((80, 95, 75)):
        expression = sparse.csr_matrix(
            rng.poisson(1.5, size=(n_cells, 20)).astype(np.float32)
        )
        graph = sparse.diags(
            [np.ones(n_cells - 1), np.ones(n_cells - 1)],
            [-1, 1],
            shape=(n_cells, n_cells),
            format="csr",
        )
        yield f"section_{index}", expression, graph


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="graphpca_out_of_core_") as directory:
        root = Path(directory)
        store = create_hierarchical_section_store(
            root / "section_store", synthetic_sections()
        )
        embeddings, global_loading, sample_loadings, info = Run_Hierarchical_Multi_GPCA(
            adatas=None,
            execution_mode="out_of_core",
            section_store=store,
            out_of_core_output_dir=root / "fit",
            n_components=5,
            lambdas=0.5,
            rhos=0.5,
            init_strategy="shared",
            outer_tol=1e-3,
            objective_tol=1e-5,
            max_iter=100,
            return_info=True,
        )
        print("converged:", info.converged, "iterations:", info.n_iter)
        print("section 0 embedding shape:", embeddings.load(0, mmap_mode="r").shape)
        print("global loading shape:", global_loading.shape)
        print("sample loading shapes:", [loading.shape for loading in sample_loadings])
