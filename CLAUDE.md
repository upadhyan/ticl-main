# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ticl** (Tabular In-Context Learning) is a research codebase for training and inference of transformer-based models for tabular data classification. It builds on TabPFN and includes three main models:

- **MotherNet** — a hypernetwork that outputs weights for a small MLP classifier
- **GAMformer** — produces interpretable additive models via in-context learning
- **TabFlex** — uses linear attention to scale beyond TabPFN's limits (supports thousands of features, hundreds of classes, millions of samples)

## Current Research Goal

Modify MotherNet to predict a **decision tree** instead of an MLP as its child model. The tree is encoded using the RADDT "ReLU+Argmin" formulation, where the transformer predicts split weights A and thresholds b, and leaf labels are computed from training data. See `docs/tree-decoder-analysis.md` for the full analysis of why RADDT was chosen over DTSemNet.

## Setup

```bash
conda env create -f environment.yml
conda activate ticl
pip install -e .
```

For TabFlex specifically, use `tabflex_conda.yaml` instead of `environment.yml`.

## Common Commands

### Running Tests
```bash
pytest -sv ticl/tests/ --durations 40       # full test suite
pytest -sv ticl/tests/training/              # training tests only
pytest -sv ticl/tests/prediction/            # prediction tests only
pytest -sv ticl/tests/training/test_train_mothernet.py  # single test file
pytest -sv ticl/tests/training/test_train_mothernet.py::test_name  # single test
```

Tests set `torch.set_num_threads(1)` via a session-scoped fixture in `ticl/conftest.py`.

### Training Models
```bash
python ticl/fit_model.py mothernet           # train MotherNet (default config)
python ticl/fit_model.py mothernet -L 2      # paper config
python ticl/fit_model.py mothernet -g GPU_ID  # specify GPU
python ticl/fit_model.py --help              # see all model types
python ticl/fit_model.py mothernet --help    # model-specific options
```

Multi-GPU via `torchrun`. Experiments tracked with MLFlow (if `MLFLOW_HOSTNAME` is set) and/or Weights & Biases.

### Linting
```bash
flake8  # max-line-length = 160 (see setup.cfg)
```

## Architecture

### Training Pipeline
`fit_model.py` → `cli_parsing.py` (argument parsing with nested namespaces via `GroupedArgParser`) → `model_configs.py` (default configs per model type) → `model_builder.py:get_model()` (constructs model + dataloader) → `train.py:train()` (training loop with mixed precision, LR scheduling, gradient accumulation).

### Model Definitions (`ticl/models/`)
All models are PyTorch `nn.Module` subclasses. Key files:
- `mothernet.py` / `mothernet_additive.py` — MotherNet and its additive variant
- `gamformer.py` — GAMformer
- `tabflex.py` — TabFlex (linear attention)
- `tabpfn.py` / `biattention_tabpfn.py` — TabPFN variants
- `perceiver.py` — Perceiver-based variant
- `encoders.py` / `decoders.py` — shared encoder/decoder components
- `linear_attention.py` — linear attention implementation
- Named configs live in `ticl/configs/` (e.g., `mothernet_configs.py`)

### Prediction / sklearn Interface (`ticl/prediction/`)
Each model has an sklearn-compatible estimator in `ticl/prediction/`:
- `MotherNetClassifier`, `EnsembleMeta` — in `prediction/mothernet.py`
- `TabPFNClassifier` — in `prediction/tabpfn.py`
- `GAMformerClassifier`, `GAMformerRegressor` — in `prediction/gamformer.py`
- `TabFlex` — in `prediction/tabflex.py`

These implement `fit()`/`predict()` and handle preprocessing (normalization, one-hot encoding, feature selection). Models are loaded via `model_builder.py:load_model()` which is cached.

### Synthetic Data / Priors (`ticl/priors/`)
Training data is generated synthetically via prior distributions (MLP prior, boolean conjunctions, step functions, GP). `classification_adapter.py` wraps regression priors for classification. `prior_bag.py` bags multiple priors together. The dataloader in `dataloader.py` wraps these priors.

### Evaluation (`ticl/evaluation/`)
Benchmark evaluation code, baseline implementations (XGBoost, CatBoost, sklearn models, ResNet, MLP), critical difference plots, and demo apps. Baseline wrappers are in `evaluation/baselines/`.

### Config System
Models use nested dict configs (see `model_configs.py` for defaults). CLI parsing in `cli_parsing.py` uses `GroupedArgParser` which creates nested `argparse.Namespace` objects matching config structure. `config_utils.py` provides utilities for merging, flattening, and comparing configs.

### Model Persistence
Models are saved as `(model_state_dict, optimizer_state_dict, scheduler, config_sample)` tuples via `torch.save` with `cloudpickle`. Prediction classes can auto-download pretrained checkpoints via `utils.fetch_model()`.

## Sub-Repositories

This project includes two additional related repositories for differentiable decision tree research.

### RADDT-main/ — RADDT
Differentiable decision trees via "ReLU+Argmin" reformulation with softmin approximation (NeurIPS 2025 Spotlight). Supports classification and regression. Available in single-GPU/CPU and distributed multi-GPU (DDP) versions.

**Requirements:** PyTorch 2.0.1, Python 3.9.6, scikit-learn, numpy, pandas, h5py. No conda env file provided; install dependencies manually.

**Running experiments (single GPU/CPU):**
```bash
cd RADDT-main/singleGPUorCPUVersion
python ./test/test_RADDT.py 3 3 1 1 2 3000 "cuda" 10 5
```
For multi-GPU DDP, see job scripts in `distributedMultiGPUVersion/sh_narval_MultiGPU/`.

**Key source files:** `src/RADDT.py` (main algorithm), `src/treeFunc.py` (tree utilities), `src/warmStart.py` (CART-based initialization), `src/dataset.py` (data loading), `src/ancestorTF_File/` (precomputed tree path routing in h5 format).
