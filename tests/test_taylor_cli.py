"""
Tests of the Taylor coefficient plotting CLI, the configuration of the Taylor
analysis and the generalised helpers in `xyh_ml.plotting`.
"""

from copy import deepcopy
from itertools import product
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import xyh_ml.plotting
from xyh_ml import PROJECT_DIR
from xyh_ml.cli import plot_taylor_coefficients
from xyh_ml.config import Taylor
from xyh_ml.plotting import load_predictions, save_figure
from xyh_ml.taylor import get_tc_column, get_tc_index_list, parse_tc_column

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

CONFIG_DIR = str(PROJECT_DIR / "config")
TOP_LEVEL_CONFIGS = [
    "evaluate_taylor_coefficients",
    "plot_taylor_coefficients",
]
INPUT_NAMES = ["a", "b"]
CAMPAIGN = "2022_pre_ee_nano_v12"
CAMPAIGNS = (CAMPAIGN, "2023_post_bpix_nano_v12")
CHANNEL = "et"
DECAY_MODE = "y2b_h2tau"
TAG = "test_tag"


def _compose(config_name: str, overrides: list[str] | None = None):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        return compose(config_name=config_name, overrides=overrides or [])


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config_name", TOP_LEVEL_CONFIGS)
def test_taylor_configs_compose(config_name):
    cfg = _compose(config_name)

    assert len(cfg.taylor.mass_points) == 10
    assert all(len(m) == 2 for m in cfg.taylor.mass_points)
    assert "taylor" in cfg.artifacts.taylor_dir
    assert "${" not in OmegaConf.to_yaml(cfg, resolve=True)
    assert str(cfg.mode.decay_mode) in cfg.artifacts.taylor_dir


def test_taylor_dir_follows_overrides():
    cfg = _compose(
        "plot_taylor_coefficients",
        ["mode.fold=1", "mode.channel=tt", "tags.training_tag=xyz"],
    )
    assert cfg.artifacts.taylor_dir.endswith(
        f"xyz/taylor/{cfg.mode.decay_mode}/fold1of{cfg.mode.n_folds}/tt"
    )


def test_taylor_yaml_matches_dataclass_defaults():
    cfg = _compose("plot_taylor_coefficients")
    default = Taylor(mass_points=[])

    for name in (
        "n_samples_per_category",
        "batch_size",
        "max_order",
        "max_coefficients",
        "per_plot",
    ):
        assert cfg.taylor[name] == getattr(default, name), name


def test_taylor_dataclass_defaults():
    t = Taylor(mass_points=[[300, 60]])
    assert t.n_samples_per_category == 5000
    assert t.batch_size == 4096
    assert t.max_order == 2
    assert t.max_coefficients == 45
    assert t.per_plot == 15


def test_plot_config_has_no_fold_merging_switch():
    # The TC plots always merge all folds and eras
    assert "plotting" not in _compose("plot_taylor_coefficients")


# ---------------------------------------------------------------------------
# load_predictions(get_file, key) and save_figure(subdir)
# ---------------------------------------------------------------------------


def test_load_predictions_custom_file_and_key(tmp_path):
    cfg = OmegaConf.create({"mode": {"fold": 1, "n_folds": 3}})
    paths = []

    def get_file(c):
        paths.append(c.mode.fold)
        return tmp_path / f"custom_fold{c.mode.fold}.h5"

    for fold in range(3):
        df = pd.DataFrame({"reduction_rate": [0.1], "value": [float(fold)]})
        df.to_hdf(tmp_path / f"custom_fold{fold}.h5", key="mykey", mode="w")
        # a second key must not be read
        pd.DataFrame({"value": [-1.0]}).to_hdf(
            tmp_path / f"custom_fold{fold}.h5", key="other", mode="a"
        )

    single = load_predictions(cfg, get_file=get_file, key="mykey")
    merged = load_predictions(
        cfg, merge_folds=True, get_file=get_file, key="mykey"
    )

    assert single["value"].tolist() == [1.0]
    assert sorted(merged["value"]) == [0.0, 1.0, 2.0]
    assert merged["reduction_rate"].tolist() == pytest.approx([0.3] * 3)
    assert cfg.mode.fold == 1


def test_save_figure_subdir(monkeypatch, tmp_path):
    monkeypatch.setattr(xyh_ml.plotting, "PROJECT_DIR", tmp_path)
    cfg = OmegaConf.create({"tags": {"training_tag": "tag"}})

    files = save_figure(plt.figure(), cfg, "stem", subdir="taylor")

    assert files == [
        tmp_path / "data/output/tag/plots/taylor/stem.pdf",
        tmp_path / "data/output/tag/plots/taylor/stem.png",
    ]
    assert all(f.exists() for f in files)


# ---------------------------------------------------------------------------
# Plot CLI
# ---------------------------------------------------------------------------


def _write_taylor_file(
    cfg,
    class_names: list[str],
    n_per_class: int = 6,
    folds: list[int] | None = None,
    campaigns: tuple[str, ...] = CAMPAIGNS,
    input_names: list[str] = INPUT_NAMES,
):
    """
    Write a synthetic TC file for each fold in `folds` (default: all) with
    `n_per_class` events per class, mass point and campaign.
    """
    if folds is None:
        folds = list(range(cfg.mode.n_folds))
    for fold in folds:
        cfg_fold = deepcopy(cfg)
        cfg_fold.mode.fold = fold
        _write_taylor_file_of_fold(
            cfg_fold, class_names, n_per_class, campaigns, input_names
        )


def _write_taylor_file_of_fold(
    cfg, class_names, n_per_class, campaigns, input_names
):
    rng = np.random.default_rng(cfg.mode.fold)
    all_classes = [c.name for c in cfg.outputs.classes]
    tc_columns = [
        get_tc_column(node, idx, input_names)
        for node in class_names
        for idx in get_tc_index_list(len(input_names), 2)
    ]  # 5 coefficients per node
    frames = []
    for campaign, (m_x, m_y), cls in product(
        campaigns, [(300.0, 60.0), (500.0, 100.0)], class_names
    ):
        n = n_per_class
        data = {
            "class": [cls] * n,
            "campaign": [campaign] * n,
            "channel": [CHANNEL] * n,
            "m_x": np.full(n, m_x),
            "m_y": np.full(n, m_y),
            "is_signal": [cls == "xyh"] * n,
            "__total_weight": np.ones(n),
            "reduction_rate": np.full(n, 1 / 3),
            "target": np.zeros(n),
        }
        # Every second event (k = 0, 2, ...) is predicted as its true class,
        # the others as the next class
        k = np.arange(n)
        true_index = all_classes.index(cls)
        predicted = np.where(
            k % 2 == 0, true_index, (true_index + 1) % len(all_classes)
        )
        for j, c in enumerate(all_classes):
            data[f"score_{c}"] = np.where(predicted == j, 0.7, 0.1)
        for c in tc_columns:
            data[c] = rng.normal(size=n).astype(np.float32)
        frames.append(pd.DataFrame(data))
    output = (
        Path(cfg.artifacts.taylor_dir) / f"taylor_coefficients_{DECAY_MODE}.h5"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_hdf(output, key="taylor", mode="w")


def _run_plot_cli(tmp_path, monkeypatch, overrides):
    cfg = _compose(
        "plot_taylor_coefficients",
        [
            # One directory per fold, as in config/base/artifacts.yaml
            f"artifacts.taylor_dir='{tmp_path}/taylor/fold${{mode.fold}}'",
            f"tags.training_tag={TAG}",
            *overrides,
        ],
    )
    monkeypatch.setattr(xyh_ml.plotting, "PROJECT_DIR", tmp_path)
    return cfg


class _Recorder(list):
    """List of saved figure stems, with the sizes of the plotted pages."""

    def __init__(self):
        super().__init__()
        self.page_sizes = []


@pytest.fixture
def saved_stems(monkeypatch):
    """Replace saving by a recorder of the figure stems (fast)."""
    stems = _Recorder()
    page_sizes = stems.page_sizes

    def fake_save_figure(fig, cfg, stem, subdir=None):
        assert subdir == "taylor"
        stems.append(stem)
        plt.close(fig)

    def fake_plot_coefficients(summary, rlabel, title=None, **kwargs):
        # Drawing is covered in test_taylor_plotting, keep the CLI tests fast
        page_sizes.append(len(summary))
        return plt.subplots(figsize=(1, 1))

    monkeypatch.setattr(
        plot_taylor_coefficients, "save_figure", fake_save_figure
    )
    monkeypatch.setattr(
        plot_taylor_coefficients, "plot_coefficients", fake_plot_coefficients
    )
    return stems


def _plot_dir(tmp_path):
    return tmp_path / "data" / "output" / TAG / "plots" / "taylor"


def test_plot_cli_page_count_and_names(tmp_path, monkeypatch, saved_stems):
    # 5 coefficients per node, at most 4 of them in pages of 3 -> 2 pages
    cfg = _run_plot_cli(
        tmp_path,
        monkeypatch,
        ["taylor.max_coefficients=4", "taylor.per_plot=3"],
    )
    classes = [c.name for c in cfg.outputs.classes]
    _write_taylor_file(cfg, classes)

    plot_taylor_coefficients.main.__wrapped__(cfg)

    fold = f"alleras_allfoldsof{cfg.mode.n_folds}"
    # class x mass point x page (only true = predicted = node class), no
    # split by era or fold
    assert len(saved_stems) == len(classes) * 2 * 2
    assert len(set(saved_stems)) == len(saved_stems)
    assert (
        f"taylor__t_tt__{DECAY_MODE}__{CHANNEL}__mx300_my60__{fold}__page2"
    ) in saved_stems
    assert (
        f"taylor__xyh__{DECAY_MODE}__{CHANNEL}__mx500_my100__{fold}__page1"
    ) in saved_stems
    assert not any(s.endswith("__page3") for s in saved_stems)
    # 4 coefficients -> pages of 3 and 1
    assert sorted(set(saved_stems.page_sizes)) == [1, 3]
    assert not plt.get_fignums()


def test_plot_cli_without_coefficient_limit_uses_all(
    tmp_path, monkeypatch, saved_stems
):
    cfg = _run_plot_cli(
        tmp_path,
        monkeypatch,
        ["taylor.max_coefficients=null", "taylor.per_plot=5"],
    )
    classes = [c.name for c in cfg.outputs.classes]
    _write_taylor_file(cfg, classes)

    plot_taylor_coefficients.main.__wrapped__(cfg)

    # 5 coefficients on a single page of 5
    assert len(saved_stems) == len(classes) * 2
    assert all(s.endswith("__page1") for s in saved_stems)


def test_plot_cli_writes_files_and_skips_missing_classes(
    tmp_path, monkeypatch
):
    cfg = _run_plot_cli(tmp_path, monkeypatch, ["taylor.per_plot=15"])
    classes = [c.name for c in cfg.outputs.classes]
    # Only the first two classes have events and coefficients
    _write_taylor_file(cfg, classes[:2])

    plot_taylor_coefficients.main.__wrapped__(cfg)

    files = [p.name for p in _plot_dir(tmp_path).glob("*.png")]
    assert len(files) == 2 * 2  # classes x mass points
    assert not any(f.startswith(f"taylor__{classes[2]}") for f in files)


def test_plot_cli_missing_file_raises(tmp_path, monkeypatch):
    cfg = _run_plot_cli(tmp_path, monkeypatch, [])

    with pytest.raises(FileNotFoundError):
        plot_taylor_coefficients.main.__wrapped__(cfg)


def test_plot_cli_pools_eras_and_folds(tmp_path, monkeypatch, saved_stems):
    cfg = _run_plot_cli(tmp_path, monkeypatch, ["taylor.per_plot=15"])
    classes = [c.name for c in cfg.outputs.classes]
    n_per_class = 4
    _write_taylor_file(cfg, classes, n_per_class=n_per_class)

    # Record the number of events entering each summary
    n_events = []
    summarize = plot_taylor_coefficients.summarize_coefficients

    def recording_summarize(df, columns, **kwargs):
        n_events.append(len(df))
        return summarize(df, columns, **kwargs)

    monkeypatch.setattr(
        plot_taylor_coefficients,
        "summarize_coefficients",
        recording_summarize,
    )

    plot_taylor_coefficients.main.__wrapped__(cfg)

    # One plot per class x mass point, each with the correctly predicted
    # events (half of them) of all eras and all folds
    assert len(saved_stems) == len(classes) * 2
    assert not any(CAMPAIGNS[0] in s or CAMPAIGNS[1] in s for s in saved_stems)
    assert set(n_events) == {
        n_per_class // 2 * len(CAMPAIGNS) * cfg.mode.n_folds
    }


def test_plot_cli_only_diagonal_events(tmp_path, monkeypatch, saved_stems):
    # TCs of node c are summarized only for events with true class c that
    # are also predicted as class c
    cfg = _run_plot_cli(tmp_path, monkeypatch, ["taylor.per_plot=15"])
    classes = [c.name for c in cfg.outputs.classes]
    _write_taylor_file(cfg, classes)

    calls = []
    summarize = plot_taylor_coefficients.summarize_coefficients

    def recording_summarize(df, columns, **kwargs):
        nodes = {parse_tc_column(c)[0] for c in columns}
        assert len(nodes) == 1
        calls.append((nodes.pop(), df))
        return summarize(df, columns, **kwargs)

    monkeypatch.setattr(
        plot_taylor_coefficients,
        "summarize_coefficients",
        recording_summarize,
    )

    plot_taylor_coefficients.main.__wrapped__(cfg)

    assert len(calls) == len(classes) * 2
    for node, df in calls:
        assert (df["class"] == node).all()
        predicted = df[[f"score_{c}" for c in classes]].to_numpy().argmax(1)
        assert (np.asarray(classes)[predicted] == node).all()


def test_plot_cli_missing_fold_raises(tmp_path, monkeypatch):
    cfg = _run_plot_cli(tmp_path, monkeypatch, [])
    classes = [c.name for c in cfg.outputs.classes]
    _write_taylor_file(cfg, classes, folds=list(range(1, cfg.mode.n_folds)))

    with pytest.raises(FileNotFoundError):
        plot_taylor_coefficients.main.__wrapped__(cfg)


@pytest.mark.parametrize("include_parameters", [False, True])
def test_plot_cli_parameter_coefficients(
    tmp_path, monkeypatch, saved_stems, include_parameters
):
    cfg = _run_plot_cli(
        tmp_path,
        monkeypatch,
        [
            "taylor.max_coefficients=null",
            f"taylor.include_parameters={include_parameters}",
        ],
    )
    classes = [c.name for c in cfg.outputs.classes]
    _write_taylor_file(cfg, classes, input_names=INPUT_NAMES + ["m_x"])

    summarized = []
    summarize = plot_taylor_coefficients.summarize_coefficients

    def recording_summarize(df, columns, **kwargs):
        summarized.append(list(columns))
        return summarize(df, columns, **kwargs)

    monkeypatch.setattr(
        plot_taylor_coefficients,
        "summarize_coefficients",
        recording_summarize,
    )

    plot_taylor_coefficients.main.__wrapped__(cfg)

    with_m_x = [any("m_x" in c for c in cols) for cols in summarized]
    assert summarized
    assert all(with_m_x) if include_parameters else not any(with_m_x)
    # 3 inputs: 3 + 6 TCs per node, 2 inputs without m_x: 2 + 3 TCs
    assert {len(cols) for cols in summarized} == {
        9 if include_parameters else 5
    }
