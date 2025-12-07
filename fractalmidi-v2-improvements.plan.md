# FractalMIDI v2.0 Improvements & Restructuring Plan

This plan incorporates the project restructuring requested by the user alongside the functional improvements.

## 1. Project Restructuring

We will reorganize the project to separate code, data, and results, and split logic into focused modules.

**New Directory Structure:**
```
FractalMIDI/
|-- README.md
|-- pyproject.toml
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- lists/ (train.txt, etc.)
|-- code/
|   |-- models/           # Model definitions
|   |-- dataset/          # Dataset loading & processing
|   |-- training/         # Trainer & utils
|   |-- inference/        # Inference scripts & metrics
|   |-- visualization/    # Visualizer
|   `-- preprocessing/    # Preprocessing scripts
|-- results/
|   |-- figures/
|   `-- tables/
`-- web/                  # WebUI (keep distinct or move to code/web)
```

**Tasks:**
1.  Create directories.
2.  Move `models/` -> `code/models/`.
3.  Move `dataset.py`, `dataset_utils.py` -> `code/dataset/`.
4.  Move `trainer.py`, `train_utils.py` -> `code/training/`.
5.  Move `inference.py` -> `code/inference/`.
6.  Move `visualizer.py` -> `code/visualization/`.
7.  Move `preprocess.py` -> `code/preprocessing/`.
8.  **Fix Imports**: Update all files to use relative imports or adjust `sys.path` in entry points.

## 2. Flexible Bar Position Embedding

-   **[code/dataset/dataset.py](code/dataset/dataset.py)**:
    -   Complete the refactor to use `dataset_utils.compute_bar_pos`.
    -   Ensure correct import from `.dataset_utils`.
-   **[code/models/model_config.py](code/models/model_config.py)**:
    -   Add `max_bar_len`.
-   **[code/models/temporal_fractal.py](code/models/temporal_fractal.py)**:
    -   Update embeddings.

## 3. Modularize Models

-   **[code/models/components.py](code/models/components.py)** (New File):
    -   Extract `AdaLN`, `TimePositionalEmbedding`, `FractalInputProj` from `temporal_fractal.py`.

## 4. Configurable Harmonic Compression

-   **[code/models/model_config.py](code/models/model_config.py)**:
    -   Add `compressed_dim` and `compression_act`.
-   **[code/models/temporal_fractal.py](code/models/temporal_fractal.py)**:
    -   Implement configurable compression.

## 5. Objective Evaluation Metrics

-   **[code/inference/metrics.py](code/inference/metrics.py)** (New File):
    -   Implement metrics with `MetricsConfig`.
-   **[code/inference/inference.py](code/inference/inference.py)**:
    -   Add `--evaluate` flag.

## 6. WebUI and Visualization

-   **[code/models/temporal_fractal.py](code/models/temporal_fractal.py)**:
    -   Return condition influence data.
-   **[code/inference/inference.py](code/inference/inference.py)**:
    -   Save influence visualization.
-   **[web/backend/app.py](web/backend/app.py)**:
    -   Update for new imports and visualization.

## Verification

-   Run `code/dataset/dataset.py` tests.
-   Run `code/inference/inference.py` with evaluation.

