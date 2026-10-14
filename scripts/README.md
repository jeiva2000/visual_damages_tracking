# scripts/

Secondary scripts: analyses, figures and the comparison with prior work. The core of the project (gap metric,
k-fold comparison, trained heads) stays in the repository root, with its library modules in `reid_lib/`, and the live pipeline in `live_tracking/`.

| Folder | Contents |
|---|---|
| `kfold/` | Variants of the k-fold comparison (pooled folds, inverted calibration, pooling of existing results) |
| `embedding_analysis/` | Embedding analysis: compression (PCA / t-SNE / dendrogram), isotropy, C4 confusion cases |
| `paper_figures/` | Figures of the paper (qualitative native vs. C4, module diagram) |
| `paper7_comparison/` | Comparison with prior work [7] on its five test sequences |

Run them **from the repository root**, for example
`python scripts/paper_figures/paper_fig_reid_module_diagram_pptx.py --help`. Each script adds the repository root
to its import path, so they can import the root modules (`analyze_dinov2_nn_baseline`, ...) and `reid_lib/` (`kfold_utils`, `p4_backbone_encoder`, ...). Several of
them read data that is not in the repository (the dataset, feature caches, live-run outputs); their default
paths point to the layout of the server where the experiments were run.
