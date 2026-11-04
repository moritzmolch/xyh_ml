"""
Shared helpers for the plotting scripts of the classification performance.
"""

import logging
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from pandas import DataFrame

from xyh_ml.config import Config

# Set up logging
logger = logging.getLogger(__name__)


DECAY_MODE_LABELS = {
    "y2b_h2tau": r"$\text{X} \to \text{Y}(\text{b}\text{b})\text{H}(\tau\tau)$",
    "y2tau_h2b": r"$\text{X} \to \text{Y}(\tau\tau)\text{H}(\text{b}\text{b})$",
}


CAMPAIGN_LABELS = {
    "2022_pre_ee_nano_v12": "2022preEE",
    "2022_post_ee_nano_v12": "2022postEE",
    "2023_pre_bpix_nano_v12": "2023preBPix",
    "2023_post_bpix_nano_v12": "2023postBPix",
    "2024_nano_v15": "2024",
    "2025_nano_v15": "2025",
}


CHANNEL_LABELS = {
    "et": r"$\text{e}\tau_{\text{h}}$",
    "mt": r"$\mu\tau_{\text{h}}$",
    "tt": r"$\tau_{\text{h}}\tau_{\text{h}}$",
}


def set_plot_style() -> None:
    """
    Use the CMS plot style of `mplhep` for all following plots.
    """
    hep.style.use("CMS")


def get_style_color(index: int) -> str:
    """
    Get a color of the color cycle of the current plot style.

    Parameters
    ----------
    index : int
        Index in the color cycle, wrapped around if it exceeds the number of
        colors.

    Returns
    -------
    str
        Color of the current color cycle.
    """
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return colors[index % len(colors)]


def add_cms_label(
    ax: Axes,
    rlabel: str,
    llabel: str = "Private Work",
) -> None:
    """
    Add the CMS label to the upper left and a description to the upper right
    of the given axes.

    Parameters
    ----------
    ax : Axes
        Axes to add the labels to.
    rlabel : str
        Text on the upper right of the axes.
    llabel : str, optional
        Text following "CMS" on the upper left of the axes. Default is
        `"Private Work"`.
    """
    hep.cms.label(llabel, ax=ax, data=True, rlabel=rlabel, loc=0)


def get_predictions_file(cfg: Config) -> Path:
    """
    Get the predictions file written by `evaluate_test_sample` for the decay
    mode, fold and channel selected in `cfg`.

    Parameters
    ----------
    cfg : Config
        Configuration with the `artifacts` and `mode` sections.

    Returns
    -------
    Path
        Path to the HDF5 file containing the predictions.
    """
    return (
        Path(cfg.artifacts.predictions_dir)
        / f"predictions_{cfg.mode.decay_mode}.h5"
    )


def load_predictions(cfg: Config, merge_folds: bool = False) -> DataFrame:
    """
    Load the predictions written by `evaluate_test_sample`.

    Without `merge_folds`, the predictions of the held-out fold
    `cfg.mode.fold` are loaded. With `merge_folds`, the predictions of all
    folds `0, ..., cfg.mode.n_folds - 1` are loaded and concatenated. Each
    event then carries the prediction of the model that has not been trained
    on it.

    Each fold file holds a reduction rate that already includes the factor
    `1 / n_folds` for evaluating a single fold, so that the weighted sum of
    one fold estimates the full yield. When merging the folds, the reduction
    rate is multiplied by `n_folds` so that the weighted sum of the merged
    events still estimates the full yield.

    Parameters
    ----------
    cfg : Config
        Configuration with the `artifacts` and `mode` sections. The path
        `artifacts.predictions_dir` is resolved separately for each fold.
    merge_folds : bool, optional
        If `True`, merge the predictions of all folds. Default is `False`.

    Returns
    -------
    DataFrame
        Predictions with an additional column `eval_fold` containing the
        index of the fold the events have been evaluated in.

    Raises
    ------
    FileNotFoundError
        If the predictions file of any requested fold does not exist.
    """
    n_folds = cfg.mode.n_folds
    folds = list(range(n_folds)) if merge_folds else [cfg.mode.fold]

    # Resolve the predictions file for each fold by setting the fold in a copy
    # of the configuration, so that the path is interpolated from the config
    input_files = {}
    for fold in folds:
        cfg_fold = deepcopy(cfg)
        cfg_fold.mode.fold = fold
        input_files[fold] = get_predictions_file(cfg_fold)

    # Report all missing files at once
    missing = [str(f) for f in input_files.values() if not f.exists()]
    if missing:
        msg = "Predictions file(s) not found:\n  " + "\n  ".join(missing)
        logger.critical(msg)
        raise FileNotFoundError(msg)

    # Load the predictions of each fold
    data_frames = []
    for fold, input_file in input_files.items():
        df = pd.read_hdf(input_file)
        df["eval_fold"] = fold
        data_frames.append(df)
        logger.info(f"Loaded {len(df)} predictions from {input_file}")

    data_frame = pd.concat(data_frames, ignore_index=True)

    # Correct the reduction rate for merging all folds
    if merge_folds:
        data_frame["reduction_rate"] = data_frame["reduction_rate"] * n_folds

    return data_frame


def add_class_indices(data_frame: DataFrame, cfg: Config) -> DataFrame:
    """
    Add the index of the true and the predicted output class to each event.

    The true class index is taken from the output variable, the predicted
    class index is the class with the highest score. The indices follow the
    order of `cfg.outputs.classes`.

    Parameters
    ----------
    data_frame : DataFrame
        Predictions with the output variable and the `score_<class>` columns.
    cfg : Config
        Configuration with the `outputs` section.

    Returns
    -------
    DataFrame
        The input data frame with the additional columns `true_class_index`
        and `predicted_class_index`.
    """
    output_classes = cfg.outputs.classes
    output_variable_name = cfg.outputs.variables[0].name
    score_columns = [f"score_{c.name}" for c in output_classes]

    data_frame["true_class_index"] = data_frame[output_variable_name].map(
        {c.name: i for i, c in enumerate(output_classes)}
    )
    data_frame["predicted_class_index"] = np.argmax(
        data_frame[score_columns].to_numpy(), axis=1
    )

    return data_frame


def get_fold_label(cfg: Config, merge_folds: bool = False) -> tuple[str, str]:
    """
    Get the label of the evaluated fold(s) for plot titles and file names.

    Parameters
    ----------
    cfg : Config
        Configuration with the `mode` section.
    merge_folds : bool, optional
        If `True`, the label refers to all folds. Default is `False`.

    Returns
    -------
    tuple[str, str]
        Label for the plot title and suffix for the output file name.
    """
    n_folds = cfg.mode.n_folds
    if merge_folds:
        return f"all folds of {n_folds}", f"allfoldsof{n_folds}"
    fold = cfg.mode.fold
    return f"held-out fold {fold} of {n_folds}", f"fold{fold}of{n_folds}"
