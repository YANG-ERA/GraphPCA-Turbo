# GraphPCA-Turbo

[![CI](https://github.com/YANG-ERA/GraphPCA-Turbo/actions/workflows/ci.yml/badge.svg)](https://github.com/YANG-ERA/GraphPCA-Turbo/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

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
