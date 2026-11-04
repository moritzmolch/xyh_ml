from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

from xyh_ml.plotting import (
    add_class_indices,
    get_fold_label,
    get_predictions_file,
    load_predictions,
)

N_FOLDS = 3
CLASSES = ["xyh", "t_tt", "z2tau"]


@pytest.fixture
def cfg(tmp_path: Path) -> DictConfig:
    # Mimic the Hydra configuration, including the interpolation of the
    # predictions directory from the selected fold and the struct flag
    cfg = OmegaConf.create(
        {
            "mode": {
                "decay_mode": "y2b_h2tau",
                "n_folds": N_FOLDS,
                "fold": 1,
                "channel": "et",
            },
            "artifacts": {
                "predictions_dir": str(tmp_path)
                + "/${mode.decay_mode}/fold${mode.fold}of${mode.n_folds}"
                + "/${mode.channel}",
            },
            "outputs": {
                "variables": [
                    {"name": "class", "transformation": "ordinal_encoding"}
                ],
                "classes": [{"name": c} for c in CLASSES],
            },
        }
    )
    OmegaConf.set_struct(cfg, True)
    return cfg


def _write_predictions(cfg: DictConfig, folds: list[int]) -> None:
    # Write a small predictions file for each fold, using the event number to
    # mark the fold the events belong to and the fold as reduction rate base
    for fold in folds:
        cfg_fold = deepcopy(cfg)
        cfg_fold.mode.fold = fold
        output_file = get_predictions_file(cfg_fold)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        n = 4
        df = pd.DataFrame(
            {
                "event": np.arange(n) * N_FOLDS + fold,
                "class": CLASSES[:2] * (n // 2),
                "reduction_rate": np.full(n, 0.1 / N_FOLDS),
                "__total_weight": np.ones(n),
            }
        )
        df.to_hdf(output_file, key="predictions", mode="w")


def test_get_predictions_file(cfg, tmp_path):
    assert get_predictions_file(cfg) == (
        tmp_path / "y2b_h2tau/fold1of3/et/predictions_y2b_h2tau.h5"
    )


def test_load_predictions_single_fold(cfg):
    _write_predictions(cfg, list(range(N_FOLDS)))

    df = load_predictions(cfg)

    assert len(df) == 4
    assert (df["eval_fold"] == 1).all()
    assert (df["event"] % N_FOLDS == 1).all()
    np.testing.assert_allclose(df["reduction_rate"], 0.1 / N_FOLDS)


def test_load_predictions_merge_folds(cfg):
    _write_predictions(cfg, list(range(N_FOLDS)))

    df = load_predictions(cfg, merge_folds=True)

    assert len(df) == 4 * N_FOLDS
    assert sorted(df["eval_fold"].unique()) == list(range(N_FOLDS))
    assert (df["event"] % N_FOLDS == df["eval_fold"]).all()
    assert df.index.is_unique

    # Reduction rate corrected for merging, the weighted sum of all folds
    # matches the weighted sum of a single fold
    np.testing.assert_allclose(df["reduction_rate"], 0.1)
    df_single = load_predictions(cfg)
    np.testing.assert_allclose(
        (df["__total_weight"] / df["reduction_rate"]).sum(),
        (df_single["__total_weight"] / df_single["reduction_rate"]).sum(),
    )

    # The configuration passed to the function is not modified
    assert cfg.mode.fold == 1


def test_load_predictions_missing_folds(cfg, tmp_path):
    _write_predictions(cfg, [1])

    with pytest.raises(FileNotFoundError) as exc_info:
        load_predictions(cfg, merge_folds=True)

    msg = str(exc_info.value)
    assert "fold0of3" in msg
    assert "fold2of3" in msg
    assert "fold1of3" not in msg


def test_add_class_indices(cfg):
    df = pd.DataFrame(
        {
            "class": ["xyh", "t_tt", "z2tau", "xyh"],
            "score_xyh": [0.7, 0.1, 0.2, 0.1],
            "score_t_tt": [0.2, 0.8, 0.2, 0.3],
            "score_z2tau": [0.1, 0.1, 0.6, 0.6],
        }
    )

    df = add_class_indices(df, cfg)

    assert df["true_class_index"].tolist() == [0, 1, 2, 0]
    assert df["predicted_class_index"].tolist() == [0, 1, 2, 2]


def test_get_fold_label(cfg):
    assert get_fold_label(cfg) == ("held-out fold 1 of 3", "fold1of3")
    assert get_fold_label(cfg, merge_folds=True) == (
        "all folds of 3",
        "allfoldsof3",
    )
