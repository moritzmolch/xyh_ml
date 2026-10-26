# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Machine learning for the X → YH → bbττ analysis (KIT, CMS). The package `xyh_ml` (in `src/xyh_ml`) trains classifiers on preprocessed ntuple "snapshots" produced by the sibling `bbtautau-kit` / CROWN workflow, and evaluates them (predictions, categorization, performance plots). Two decay modes exist: `y2b_h2tau` and `y2tau_h2b`.

## Environment and commands

- Environment is managed by pixi (`pyproject.toml`, `pixi.lock`; linux-64, CUDA 12, pytorch-gpu): `pixi shell` or `pixi run <cmd>`. The README's micromamba/venv + `requirements.txt` route is an alternative.
- Submodule `modules/tayloranalysis` (git@github.com:kit-cms/tayloranalysis): `git submodule update --init`.
- Tests: `pytest` (uses `--import-mode=importlib`); single test: `pytest tests/config/test_network.py::test_name`. Tests currently cover only `xyh_ml.config`.
- Lint/format: `ruff check` / `ruff format` — line length **79**, doc length 79.
- CLI entry points are Hydra apps run as modules from the repo root, e.g.
  `python -m xyh_ml.cli.preprocess_snapshots mode.decay_mode=y2b_h2tau tags.snapshot_tag=... tags.training_tag=...`
  Override any config value on the command line (`mode.fold=1`, `inputs=y2tau_h2b`, ...).

## Architecture

**Pipeline** (each stage is a script in `src/xyh_ml/cli/`, paired with a top-level YAML in `config/`):
1. `preprocess_snapshots` (`config/preprocess_snapshots.yaml`) — reads ROOT snapshots (per campaign/channel from `data_source`), writes HDF5 data frames under `artifacts.data_frame_dir`.
2. `train_classifier` (`config/train_classifier.yaml`) — trains a parametrized MLP (`cli/model.py`, `MultiLayerPerceptron`) separately per `decay_mode`, held-out `fold` (k-fold split on `event % mode.n_folds`; the held-out fold is used for evaluation) and analysis `channel` (et/mt/tt). Outputs go to `artifacts.training_dir`.
3. `evaluate_categorization` / `plot_classification_performance` — categorization over the evaluation fold (predictions in `predictions_dir`) and plots. `cli/evaluate_classifier.py` is obsolete; ignore it.

**Workflow** (run from the repo root; repeat per `mode.decay_mode` ∈ {`y2b_h2tau`, `y2tau_h2b`, also select `inputs=<same>`} per `mode.fold` ∈ {0, …, `mode.n_folds`-1} and per `mode.channel` ∈ {et, mt, tt}; `tags.snapshot_tag` / `tags.training_tag` default to `config/base/tags.yaml`):
```bash
M="mode.decay_mode=y2b_h2tau inputs=y2b_h2tau mode.channel=et"
# 1. ROOT snapshots -> HDF5 data frames (fold-independent, once per decay mode)
python -m xyh_ml.cli.preprocess_snapshots $M
# 2. Train on all folds except the held-out fold f; writes config/transformations.pkl/metrics.json/checkpoints.pt to training_dir
python -m xyh_ml.cli.train_classifier $M mode.fold=0
# 3. Categorization on the *held-out* fold (reads training_dir artifacts): subsample (10k/category) and write predictions_<decay_mode>.h5 with score_<class> columns
python -m xyh_ml.cli.evaluate_categorization $M mode.fold=0
# 4. Efficiency plots (pdf/png) from the h5 of 3 -> data/output/<training_tag>/plots/
python -m xyh_ml.cli.plot_classification_performance $M mode.fold=0
```

Each step consumes the previous step's files, so run in order. `plot_classification_performance` reads the predictions file of the selected tag/mode/fold/channel from `predictions_dir`, so run it per channel with the same overrides.

**Configuration** is Hydra composition: top-level `config/*.yaml` lists defaults from `config/base/*.yaml` (mode, tags, artifacts, outputs, data_source, dataset, network, training) plus `config/inputs/<decay_mode>.yaml` (input variables, parameters, output classes, with per-variable `transformation` and `fallback_value`). Typed dataclasses mirroring this structure live in `src/xyh_ml/config/dataclasses.py` (`Config`, `Variable`, `OutputClass`, `Network`, `Training`, ...); `config/util.py` converts the Hydra config into them. When adding a config field, change the YAML and the dataclass together. Hydra runs write to `output/hydra/<job>/<timestamp>` and do not `chdir`; `PROJECT_DIR` (`xyh_ml/__init__.py`) is the repo root used to locate `config/`.

Paths in `config/base/artifacts.yaml` and `data_source.yaml` point to `/ceph/mmolch/...` storage and are interpolated from `tags.*` and `mode.*`; training outputs land in `data/output/<training_tag>/...`.

**Two generations of code coexist:**
- `cli/train_classifier.py` (large, self-contained, plain PyTorch, Hydra-config-driven) is the current path.
- `cli/train_pnn.py` plus `src/xyh_ml/{data,lightning,models,mlflow_utils.py}` is an older Lightning + MLflow + click implementation. It imports modules that no longer exist (`xyh_ml.config.data`, `xyh_ml.config.network`) and is stale; verify before relying on it.

## Conventions

- Seeds are module constants in the CLI scripts (e.g. `TRAIN_TEST_SPLIT_SEED`); keep them fixed for reproducibility.
- PRs use `.github/PULL_REQUEST_TEMPLATE.md` (Overview, Key Changes, Testing and Verification, Breaking Changes).
- `data/`, `output/`, `notebooks_dev/` hold generated artifacts and exploratory notebooks, not library code.
