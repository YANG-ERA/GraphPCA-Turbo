# GraphPCA-Turbo

[![CI](https://github.com/YANG-ERA/GraphPCA-Turbo/actions/workflows/ci.yml/badge.svg)](https://github.com/YANG-ERA/GraphPCA-Turbo/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-ReadTheDocs-blue.svg)](https://graphpca-analyses.readthedocs.io/en/latest/)

GraphPCA-Turbo is a scalable and interpretable graph-regularized dimension-reduction framework for spatial transcriptomics. It preserves the original single-sample GraphPCA interface and extends it with iterative and optional C++-accelerated solvers, hierarchical multi-sample partial pooling, cohort-level loading coordinates, rotation-aware convergence diagnostics, and projection of unseen spatial sections.

The Python distribution remains named `st-graphpca`, and the import namespace remains `GraphPCA`.

## Main capabilities

- **Single-sample GraphPCA** through `Run_GPCA`
- **Exact, iterative, and optional accelerated solvers**
- **Hierarchical multi-sample GraphPCA** through `Run_Hierarchical_Multi_GPCA`
- **Sample-specific loadings** $W_s$ and a **cohort-level loading** $W_0$
- **Partial pooling** controlled by sample-specific or shared $\rho$
- **Equal-slice or size-proportional sample weighting**
- **Rotation-aware convergence and stationarity diagnostics**
- **Projection of unseen samples** through `Project_Hierarchical_Multi_GPCA`
- **Sparse expression-matrix support**
- **Optional disk-backed, section-separable hierarchical fitting** for atlases that do not fit comfortably in RAM

## Method overview

### Single-sample GraphPCA

For one spatial sample, GraphPCA estimates a low-dimensional embedding $Z$ and an orthonormal gene loading matrix $W$:

$$\min_{Z,W} \Vert{}X-ZW^\top\Vert{}_F^2 + \lambda\,\mathrm{tr}(Z^\top LZ), \qquad W^\top W=I.$$

Here, $X$ is the expression matrix, $L$ is a spatial graph Laplacian, and $\lambda$ controls graph regularization.

### Hierarchical multi-sample GraphPCA

For samples $s=1,\ldots,S$, GraphPCA-Turbo v2 estimates sample-specific embeddings $Z_s$, sample-specific loadings $W_s$, and a shared cohort loading $W_0$:

$$\min_{\{Z_s,W_s\},W_0} \sum_{s=1}^{S}q_s \left[ \frac{1}{n_s}\Vert{}X_s-Z_sW_s^\top\Vert{}_F^2 + \frac{\lambda_s}{n_s}\mathrm{tr}(Z_s^\top L_sZ_s) + \rho_s\Vert{}W_s-W_0\Vert{}_F^2 \right],$$

subject to

$$W_s^\top W_s=I, \qquad W_0^\top W_0=I.$$

The shrinkage parameter $\rho_s$ controls information sharing:

- `rho = 0`: independent sample-specific loading spaces
- moderate positive `rho`: partial pooling
- larger `rho`: stronger alignment toward the shared loading basis

Sample weighting options are:

- `sample_weights="equal_slice"`: every section has equal total influence
- `sample_weights="size_proportional"`: influence is proportional to sample size

Legacy aliases `"balanced"` and `"spot"` remain supported.

## Installation

### Standard installation

```bash
python -m pip install st-graphpca
```

### Installation from source

```bash
git clone [https://github.com/YANG-ERA/GraphPCA-Turbo.git](https://github.com/YANG-ERA/GraphPCA-Turbo.git)
cd GraphPCA-Turbo
python -m pip install .
```

For development:

```bash
python -m pip install -e .
```

### Optional C++ acceleration

The optional C++ backend requires Eigen3 and pybind11.

```bash
conda install -c conda-forge eigen pybind11
python -m pip install --no-build-isolation .
```

To force a source build from PyPI:

```bash
conda install -c conda-forge eigen pybind11

python -m pip install \
  --no-binary st-graphpca \
  --no-build-isolation \
  st-graphpca
```

If the compiled extension is unavailable, the Python iterative solvers remain available.

## Quick start

### Single spatial sample

```python
import numpy as np
from GraphPCA import Run_GPCA

location = np.asarray(adata.obsm["spatial"])

Z, W = Run_GPCA(
    adata,
    location=location,
    n_components=30,
    _lambda=0.5,
    n_neighbors=6,
    mode="iterative",
    random_seed=666,
)

adata.obsm["X_GraphPCA"] = Z
```

Available modes are:

- `mode="exact"` for smaller datasets
- `mode="iterative"` for the Python PCG implementation
- `mode="accelerated"` for the optional C++ backend

### Hierarchical multi-sample analysis

All samples must contain the same genes in the same order.

```python
import numpy as np
from GraphPCA import Run_Hierarchical_Multi_GPCA

adatas = [adata_1, adata_2, adata_3, adata_4]
locations = [np.asarray(adata.obsm["spatial"]) for adata in adatas]

Z_list, W0, Ws_list, info = Run_Hierarchical_Multi_GPCA(
    adatas=adatas,
    locations=locations,
    n_components=30,
    lambdas=0.5,
    rhos=2.0,
    n_neighbors=6,
    sample_weights="equal_slice",
    center=True,
    pcg_tol=1e-6,
    pcg_max_iter=500,
    outer_tol=1e-6,
    max_iter=50,
    init_strategy="hybrid",
    n_jobs=1,
    mode="iterative",
    return_info=True,
)

print("Converged:", info.converged)
print("Iterations:", info.n_iter)
print("Reason:", info.convergence_reason)
```

With the default storage options, the fitted objects are also written to:

```text
adata.obsm["X_GraphPCA_HMS"]
adata.varm["GraphPCA_HMS_Ws"]
adata.varm["GraphPCA_HMS_W0"]
```

### Disk-backed hierarchical analysis for large atlases

`execution_mode="in_memory"` is the default and preserves the established
AnnData-based API, including all existing tutorials. For a cohort with too
many sections to keep in memory, create a reusable section store by yielding
one preprocessed expression matrix and its spatial graph at a time, then opt
in to `execution_mode="out_of_core"`.

```python
from pathlib import Path
from scipy import sparse
from GraphPCA import (
    Run_Hierarchical_Multi_GPCA,
    create_hierarchical_section_store,
)

# Each tuple is (section name, cells-by-genes sparse matrix, spatial graph).
# This generator can load one section from disk, yield it, and then release it.
def section_generator(section_records):
    for name, expression_file, graph_file in section_records:
        yield name, sparse.load_npz(expression_file), sparse.load_npz(graph_file)

section_records = [
    ("section_01", "prepared/section_01_X.npz", "prepared/section_01_graph.npz"),
    ("section_02", "prepared/section_02_X.npz", "prepared/section_02_graph.npz"),
]
store = create_hierarchical_section_store(
    "prepared/graphpca_section_store",
    section_generator(section_records),
)

Z_disk, W0, Ws_list, info = Run_Hierarchical_Multi_GPCA(
    adatas=None,                       # required: do not pre-load all sections
    execution_mode="out_of_core",
    section_store=store,
    out_of_core_output_dir="results/graphpca_disk_backed",
    n_components=30,
    lambdas=0.5,
    rhos=0.15,
    sample_weights="equal_slice",
    init_strategy="shared",
    mode="iterative",                 # or "accelerated" when available
    max_iter=50,
    return_info=True,
)

# Materialize only the section currently needed.
Z_section_01 = Z_disk.load(0, mmap_mode="r")
print(info.converged, info.n_iter)
```

The section store includes the spatial graph, so do not also pass `locations`,
`networks`, `platforms`, or `n_neighbors` in this mode. `Z_disk` is a
`DiskBackedEmbeddings` collection rather than an in-memory list; it has
`len()`, indexing, `.load(i)`, and `.items()` methods. Embeddings, model
state, and a `run_summary.json` are saved under `out_of_core_output_dir`; set
`out_of_core_resume=True` to continue an interrupted fit. `return_log` and
`save_reconstruction` are intentionally unavailable because they would break
the low-memory contract.

This mode keeps the expression matrix, graph operator, and embedding state of
only the current section in working memory (plus small global loading arrays).
It avoids a global cell-level graph, but it is not a promise that memory is
limited to a single array: the largest section and a staged embedding update
still determine the working-memory peak. Use the default in-memory mode when
the cohort already fits comfortably in RAM, since it can be faster.

#### MERFISH Aging engineering benchmark

On the selected MERFISH Aging input (376,107 cells, 31 sections, 374 genes;
`K=30`, `lambda=0.5`, equal-section weights, `rho=0.1487261502079812`, C++
PCG), both implementations converged and gave essentially identical final
objectives and primary section-wise clustering quality. Peak process-tree RSS
fell by about 47% with the disk-backed mode. This is an engineering benchmark,
not a manuscript result.

| Measure | Default in-memory mode | Disk-backed mode |
|---|---:|---:|
| Converged | yes (51 iterations) | yes (38 iterations) |
| Final objective | 332.440564 | 332.440345 |
| Solver/fit time | 7.45 min solver time | 8.40 min full fit time |
| Peak process-tree RSS | 2.154 GiB | 1.140 GiB |
| Primary section-wise refined ARI | 0.626373 | 0.626901 |

The two modes use different valid initializations (`hybrid` in memory,
streaming `shared` on disk), so bitwise-identical embeddings are not expected.
The result supports the disk-backed mode as a memory-saving option; it does
not establish a universal runtime improvement.

### Projection of an unseen sample

```python
import numpy as np
from GraphPCA import Project_Hierarchical_Multi_GPCA

Z_new, W_new = Project_Hierarchical_Multi_GPCA(
    adata=new_adata,
    global_loading=W0,
    location=np.asarray(new_adata.obsm["spatial"]),
    graph_lambda=0.5,
    rho=2.0,
    n_neighbors=6,
    mode="iterative",
)
```

Projection keeps the learned cohort loading $W_0$ fixed while adapting the unseen sample embedding and sample-specific loading.

## Input requirements

For single-sample analysis, the input must provide:

- `adata.X`: observations by genes
- spatial coordinates through `location`, or an adjacency matrix through `network`

For multi-sample analysis, every object in `adatas` must have:

- the same genes
- identical gene order
- a valid matrix in `.X`
- spatial coordinates or a supplied graph

Sparse expression matrices are supported.

## Outputs

- `Z` or `Z_s`: spatially regularized low-dimensional embeddings
- `W` or `W_s`: sample-specific gene loading matrices
- `W_0`: cohort-level loading matrix
- `info`: convergence, objective, PCG, stationarity, and loading-deviation diagnostics

## Testing and packaging

Run the tests:

```bash
pytest -q
```

Build and validate the distributions:

```bash
python -m build
python -m twine check dist/*
```

## Tutorials and documentation

The public [GraphPCA documentation](https://graphpca-analyses.readthedocs.io/en/latest/) provides executable tutorials with saved outputs, including analyses based on datasets used in the GraphPCA-Turbo study:

- [Tutorial 4: Visium HD mouse small intestine](https://graphpca-analyses.readthedocs.io/en/latest/Tutorial4_MSI.html)
- [Tutorial 5: multi-section ABC-WMB analysis](https://graphpca-analyses.readthedocs.io/en/latest/Tutorial5_ABC_WMB.html)

The ABC-WMB tutorial covers the input contract, section-wise preprocessing, loading precomputed section embeddings and shared loadings, rotation-aware program maps, leading loading genes, anatomical associations, section-specific loading deviations, and serial-section continuity. Because the complete atlas contains millions of cells, the raw atlas and full model outputs are not bundled with the package. The tutorial instead includes a compact result-derived sample with visible executed figures and placeholder paths for local full-data files.

Package-level examples for single-section fitting, hierarchical multi-section fitting, and projection of unseen sections are provided above and in [`examples/README_multisample.md`](examples/README_multisample.md). Paper-scale scripts and result archives are distributed separately from the installable package.

## Citation

GraphPCA-Turbo extends the original GraphPCA method. Please cite the work corresponding to the functionality used in your analysis.

### GraphPCA-Turbo

Yang, J., Qi, J., Jiang, X., Chen, X., Liu, L., and Zheng, X.  
**Ultra-Scalable Dimension Reduction for High-Resolution Spatial Transcriptomics via GraphPCA-Turbo.** Manuscript in preparation, 2026.

### Original GraphPCA

Yang, J., Wang, L., Liu, L., and Zheng, X.  
**GraphPCA: a fast and interpretable dimension reduction algorithm for spatial transcriptomics data.** *Genome Biology* 25, 287 (2024).  
DOI: `10.1186/s13059-024-03429-x`

Machine-readable citation metadata are provided in [`CITATION.cff`](CITATION.cff).

## Version history

### v2.1.0

- Added optional disk-backed, section-separable hierarchical fitting
- Added restartable fitting and on-demand disk-backed embeddings
- Preserved the established in-memory API and tutorials

### v2.0.0

- Added hierarchical multi-sample GraphPCA
- Added sample-specific and cohort-level loading matrices
- Added partial pooling controlled by `rho`
- Added sample weighting options
- Added rotation-aware convergence diagnostics
- Added unseen-section projection
- Retained the original `Run_GPCA` interface

### v1.0.0

- Added exact, iterative, and optional accelerated single-sample engines

## License

GraphPCA-Turbo is distributed under the MIT License. See [`LICENSE`](LICENSE).

## Repository

```text
[https://github.com/YANG-ERA/GraphPCA-Turbo](https://github.com/YANG-ERA/GraphPCA-Turbo)
```
