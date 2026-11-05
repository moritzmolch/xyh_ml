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
from matplotlib.figure import Figure
from matplotlib.ticker import NullLocator
from matplotlib.transforms import blended_transform_factory
from pandas import DataFrame

from xyh_ml import PROJECT_DIR
from xyh_ml.config import Config, OutputClass

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


def yields_per_predicted_class(
    data_frame: DataFrame,
    cfg: Config,
) -> tuple[DataFrame, DataFrame]:
    """
    Expected number of true background and true signal events in each
    predicted class for every campaign, channel and mass point.

    The yield is the sum of `__total_weight / reduction_rate`, which scales
    the sampled events back to the full expected yield. Events of signal
    output classes (`is_signal`) are counted as signal, all others as
    background. Combinations without events have a yield of 0.

    Parameters
    ----------
    data_frame : DataFrame
        Predictions with the columns `campaign`, `channel`, `m_x`, `m_y`,
        `__total_weight`, `reduction_rate`, the output variable and
        `predicted_class_index` (see `add_class_indices`).
    cfg : Config
        Configuration with the `outputs` section.

    Returns
    -------
    tuple[DataFrame, DataFrame]
        Yields of true background and of true signal events, each with the
        columns `campaign`, `channel`, `m_x`, `m_y`, `predicted_class` and
        `yield`, with one row per mass point and predicted class.
    """
    output_classes = cfg.outputs.classes
    output_variable_name = cfg.outputs.variables[0].name
    signal_names = [c.name for c in output_classes if c.is_signal]
    keys = ["campaign", "channel", "m_x", "m_y"]

    # All mass points of the data frame, so that mass points without events
    # of one truth category get a yield of 0
    mass_points = pd.MultiIndex.from_frame(
        data_frame[keys].drop_duplicates().sort_values(keys)
    )

    weight = data_frame["__total_weight"] / data_frame["reduction_rate"]
    is_signal = data_frame[output_variable_name].isin(signal_names)

    yields = []
    for selection in [~is_signal, is_signal]:
        df = data_frame.loc[selection, keys + ["predicted_class_index"]]
        table = (
            weight[selection]
            .groupby([df[k] for k in keys + ["predicted_class_index"]])
            .sum()
            .unstack("predicted_class_index", fill_value=0.0)
            .reindex(
                index=mass_points,
                columns=range(len(output_classes)),
                fill_value=0.0,
            )
        )
        table.columns = [c.name for c in output_classes]
        table.columns.name = "predicted_class"
        yields.append(
            table.stack()
            .rename("yield")
            .reset_index()
            .astype({"yield": float})
        )

    return yields[0], yields[1]


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


def get_rlabel(
    decay_mode: str,
    campaign: str,
    channel: str,
    *extra: str,
) -> str:
    """
    Text for the upper right of a plot: decay mode, campaign and channel,
    followed by `extra` items (e.g. the selected class and the fold label).
    """
    return ",  ".join(
        [
            DECAY_MODE_LABELS[decay_mode],
            CAMPAIGN_LABELS[campaign],
            CHANNEL_LABELS[channel],
            *extra,
        ]
    )


def plot_mass_grid(
    data: DataFrame,
    line_column: str,
    value_column: str,
    output_classes: list[OutputClass],
    ylabel: str,
    rlabel: str,
    legend_title: str = "Category",
    log_y: bool = False,
    ylim: tuple[float, float] | None = (0.0, 1.05),
    m_x_split: float = 1500,
    m_x_label_y: float = 0.19,
) -> tuple[Figure, np.ndarray]:
    """
    Plot a quantity for all mass points (m_X, m_Y) as one line per output
    class.

    The mass points are ordered by m_X and m_Y and placed next to each other
    on the x axis, labelled with m_Y. Slices of the same m_X are separated by
    vertical dashed lines and labelled with m_X. Mass points with
    `m_x <= m_x_split` are drawn in the upper panel, the others in the lower
    panel.

    Parameters
    ----------
    data : DataFrame
        Data with the columns `m_x`, `m_y`, `line_column` and `value_column`,
        with one row per mass point and output class.
    line_column : str
        Column with the output class name that selects the line of a row.
    value_column : str
        Column with the plotted values.
    output_classes : list[OutputClass]
        Output classes in the order of the lines; their labels are used in
        the legend.
    ylabel : str
        Label of the y axes.
    rlabel : str
        Text on the upper right of the upper panel (see `get_rlabel`).
    legend_title : str, optional
        Title of the legend. Default is `"Category"`.
    log_y : bool, optional
        If `True`, use a logarithmic y axis; values <= 0 are not drawn.
        Default is `False`.
    ylim : tuple[float, float] | None, optional
        Limits of the y axes; `None` keeps the automatic limits. Default is
        `(0.0, 1.05)`.
    m_x_split : float, optional
        Largest m_X of the upper panel. Default is 1500.
    m_x_label_y : float, optional
        Height of the m_X labels in axes coordinates. Default is 0.19.

    Returns
    -------
    tuple[Figure, np.ndarray]
        The figure and its two axes.
    """
    data = data.sort_values(["m_x", "m_y"], kind="stable")

    fig, ax = plt.subplots(nrows=2, figsize=(20, 12))
    m_x_color = get_style_color(5)

    parts = [data[data["m_x"] <= m_x_split], data[data["m_x"] > m_x_split]]
    for j, df_part in enumerate(parts):
        # Positions of the mass points on the x axis, from the rows of the
        # first output class
        df_part_x = df_part.loc[df_part[line_column] == output_classes[0].name]
        x = np.arange(len(df_part_x)) + 0.5
        x_labels = [f"{int(m_y)}" for m_y in df_part_x["m_y"]]

        # One line per output class. On a logarithmic axis, values <= 0 are
        # left out (gap in the line) instead of being drawn at the bottom.
        for output_class in output_classes:
            y = df_part.loc[
                df_part[line_column] == output_class.name, value_column
            ].to_numpy(dtype=float)
            if log_y:
                y = np.where(y > 0, y, np.nan)
            ax[j].plot(x, y, label=output_class.label, linestyle="dashed")

        # Vertical lines at the edges of the m_X slices and m_X labels in the
        # middle of each slice; the label height is given in axes
        # coordinates so that it works for linear and logarithmic axes
        m_x_values = df_part_x["m_x"].to_numpy()
        edge_index = (
            [0]
            + [
                i + 1
                for i in range(len(m_x_values) - 1)
                if m_x_values[i] != m_x_values[i + 1]
            ]
            + [len(m_x_values) - 1]
        )
        transform = blended_transform_factory(ax[j].transData, ax[j].transAxes)
        for i, ei in enumerate(edge_index[:-1]):
            ax[j].axvline(ei, color="black", linestyle="dashed", linewidth=1)
            ax[j].text(
                (ei + edge_index[i + 1]) / 2,
                m_x_label_y,
                f"{int(m_x_values[ei])}",
                va="bottom",
                ha="center",
                color=m_x_color,
                transform=transform,
            )

        # m_X axis label on the right-hand side of the panel
        ax[j].text(
            edge_index[-1] + 4,
            m_x_label_y,
            r"$m_{\text{X}}$ (GeV)",
            va="bottom",
            ha="left",
            color=m_x_color,
            transform=transform,
        )

        # m_Y labels on the x axis
        ax[j].set_xticks(
            x[::2], labels=x_labels[::2], rotation=90, fontsize=14
        )
        ax[j].xaxis.set_minor_locator(NullLocator())

        # Axis scales, limits and labels
        if log_y:
            ax[j].set_yscale("log")
        ax[j].set_xlim(np.min(x) - 0.5, np.max(x) + 0.5)
        if ylim is not None:
            ax[j].set_ylim(*ylim)
        ax[j].set_xlabel(r"$m_{\text{Y}}$ (GeV)")
        ax[j].set_ylabel(ylabel)

        # Legend next to the upper panel only
        if j == 0:
            ax[j].legend(
                title=legend_title,
                bbox_to_anchor=(1.01, 1.0),
                loc="upper left",
                frameon=True,
            )

    add_cms_label(ax[0], rlabel)

    # Tighten layout and leave space for the legend
    fig.tight_layout()
    fig.subplots_adjust(right=0.85)

    return fig, ax


def save_figure(fig: Figure, cfg: Config, stem: str) -> list[Path]:
    """
    Save `fig` as PDF and PNG to the plot directory of the training tag,
    `<PROJECT_DIR>/data/output/<training_tag>/plots/<stem>.{pdf,png}`, and
    close it.

    Returns
    -------
    list[Path]
        The written files.
    """
    output_dir = PROJECT_DIR / "data" / "output" / cfg.tags.training_tag
    output_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = []
    for ext in ["pdf", "png"]:
        output_file = output_dir / f"{stem}.{ext}"
        fig.savefig(output_file)
        logger.info(f"Saved plot to {output_file}")
        output_files.append(output_file)
    plt.close(fig)

    return output_files
