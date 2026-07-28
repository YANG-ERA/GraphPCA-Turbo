# Multisample GraphPCA examples

The ordinary hierarchical API remains unchanged:

```python
from GraphPCA import Run_Hierarchical_Multi_GPCA

Z_list, W0, Ws_list = Run_Hierarchical_Multi_GPCA(
    adatas=[adata_1, adata_2],
    locations=[adata_1.obsm["spatial"], adata_2.obsm["spatial"]],
    n_components=30,
    lambdas=0.5,
    rhos=1.0,
    n_neighbors=6,
)
```

For datasets with many sections, see
[`out_of_core_hierarchical.py`](out_of_core_hierarchical.py). It is a complete,
runnable synthetic example of the optional disk-backed workflow. That workflow
requires `adatas=None` and a section store created by
`create_hierarchical_section_store`; it does not change the default tutorial
workflow.
