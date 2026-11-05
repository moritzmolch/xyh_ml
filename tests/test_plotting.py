from copy import deepcopy
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig, OmegaConf

import xyh_ml.plotting
from xyh_ml.config import OutputClass
from xyh_ml.plotting import (
    add_class_indices,
    get_fold_label,
    get_predictions_file,
    get_rlabel,
    load_predictions,
    plot_mass_grid,
    save_figure,
    yields_per_predicted_class,
)

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

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


def _output_classes() -> list[OutputClass]:
    return [
        OutputClass(name=c, label=c, file_basenames=[], is_signal=c == "xyh")
        for c in CLASSES
    ]


def _mass_grid_data() -> pd.DataFrame:
    # Two m_x slices in the upper panel (m_x <= 1500) and one in the lower
    # panel, in shuffled order; the value encodes the class
    mass_points = [
        (300, 60),
        (300, 70),
        (1000, 60),
        (1000, 70),
        (1000, 80),
        (2000, 90),
        (2000, 100),
    ]
    records = [
        {"m_x": m_x, "m_y": m_y, "class": c, "value": 0.1 * (k + 1)}
        for m_x, m_y in mass_points
        for k, c in enumerate(CLASSES)
    ]
    return pd.DataFrame.from_records(records).sample(frac=1, random_state=1)


@pytest.mark.parametrize("log_y", [False, True])
@pytest.mark.filterwarnings("ignore:Tight layout not applied")
def test_plot_mass_grid(log_y):
    fig, ax = plot_mass_grid(
        _mass_grid_data(),
        line_column="class",
        value_column="value",
        output_classes=_output_classes(),
        ylabel="value",
        rlabel="test",
        log_y=log_y,
        ylim=None if log_y else (0.0, 1.05),
    )

    assert len(ax) == 2
    for a, n_points in zip(ax, [5, 2]):
        # One line per class, with one point per mass point of the panel
        lines = {
            line.get_label(): line
            for line in a.get_lines()
            if line.get_label() in CLASSES
        }
        assert list(lines) == CLASSES
        for k, c in enumerate(CLASSES):
            np.testing.assert_allclose(
                lines[c].get_ydata(), [0.1 * (k + 1)] * n_points
            )
        assert a.get_yscale() == ("log" if log_y else "linear")

    # m_y tick labels sorted by m_x, then m_y (every second one is shown)
    labels = [t.get_text() for t in ax[0].get_xticklabels()]
    assert labels == ["60", "60", "80"]

    # m_x slice labels followed by the m_x axis label
    texts = [t.get_text() for t in ax[0].texts]
    assert texts[:2] == ["300", "1000"]
    assert [t.get_text() for t in ax[1].texts][0] == "2000"
    plt.close(fig)


@pytest.mark.filterwarnings("ignore:Tight layout not applied")
def test_plot_mass_grid_log_y_skips_zeros():
    data = _mass_grid_data()
    data.loc[(data["class"] == "t_tt") & (data["m_x"] == 300), "value"] = 0.0

    fig, ax = plot_mass_grid(
        data,
        line_column="class",
        value_column="value",
        output_classes=_output_classes(),
        ylabel="value",
        rlabel="test",
        log_y=True,
        ylim=None,
    )

    line = next(x for x in ax[0].get_lines() if x.get_label() == "t_tt")
    y = np.asarray(line.get_ydata(), dtype=float)
    # The two mass points with m_x = 300 come first and are not drawn
    assert np.isnan(y[:2]).all()
    np.testing.assert_allclose(y[2:], [0.2] * 3)
    plt.close(fig)


def test_save_figure(monkeypatch, tmp_path):
    monkeypatch.setattr(xyh_ml.plotting, "PROJECT_DIR", tmp_path)
    cfg = OmegaConf.create({"tags": {"training_tag": "tag"}})
    fig = plt.figure()

    files = save_figure(fig, cfg, "stem")

    assert files == [
        tmp_path / "data/output/tag/plots/stem.pdf",
        tmp_path / "data/output/tag/plots/stem.png",
    ]
    assert all(f.exists() for f in files)
    assert not plt.fignum_exists(fig.number)


def test_yields_per_predicted_class():
    cfg = OmegaConf.create(
        {
            "outputs": {
                "variables": [{"name": "class"}],
                "classes": [
                    {"name": c, "is_signal": c == "xyh"} for c in CLASSES
                ],
            }
        }
    )
    # Two mass points; background events are repeated for both, signal
    # events only exist for the first mass point. Predicted class indices
    # refer to CLASSES = [xyh, t_tt, z2tau].
    df = pd.DataFrame.from_records(
        [
            # campaign, channel, m_x, m_y, class, weight, rate, predicted
            ("c", "et", 300, 60, "t_tt", 2.0, 0.5, 1),
            ("c", "et", 300, 60, "t_tt", 1.0, 0.5, 0),
            ("c", "et", 300, 60, "z2tau", 3.0, 1.0, 1),
            ("c", "et", 300, 60, "xyh", 1.0, 0.25, 0),
            ("c", "et", 300, 60, "xyh", 0.5, 0.25, 2),
            ("c", "et", 400, 70, "t_tt", 2.0, 0.5, 1),
            ("c", "et", 400, 70, "t_tt", 1.0, 0.5, 1),
            ("c", "et", 400, 70, "z2tau", 3.0, 1.0, 2),
        ],
        columns=[
            "campaign",
            "channel",
            "m_x",
            "m_y",
            "class",
            "__total_weight",
            "reduction_rate",
            "predicted_class_index",
        ],
    )

    background, signal = yields_per_predicted_class(df, cfg)

    def as_dict(yields):
        return {
            (r.m_x, r.predicted_class): r["yield"]
            for _, r in yields.iterrows()
        }

    # Background: weights divided by the reduction rate, per predicted class
    assert as_dict(background) == {
        (300, "xyh"): 2.0,
        (300, "t_tt"): 4.0 + 3.0,
        (300, "z2tau"): 0.0,
        (400, "xyh"): 0.0,
        (400, "t_tt"): 4.0 + 2.0,
        (400, "z2tau"): 3.0,
    }
    # Signal: the second mass point has no signal events and yields 0
    assert as_dict(signal) == {
        (300, "xyh"): 4.0,
        (300, "t_tt"): 0.0,
        (300, "z2tau"): 2.0,
        (400, "xyh"): 0.0,
        (400, "t_tt"): 0.0,
        (400, "z2tau"): 0.0,
    }
    assert list(background.columns) == [
        "campaign",
        "channel",
        "m_x",
        "m_y",
        "predicted_class",
        "yield",
    ]
    # The sum over predicted classes is the total background yield, which is
    # the same for both mass points except for the moved events
    totals = background.groupby("m_x")["yield"].sum()
    assert totals.tolist() == [9.0, 9.0]


def test_get_rlabel():
    rlabel = get_rlabel("y2b_h2tau", "2024_nano_v15", "et", "a", "b")
    parts = rlabel.split(",  ")
    assert parts[1:] == ["2024", r"$\text{e}\tau_{\text{h}}$", "a", "b"]
