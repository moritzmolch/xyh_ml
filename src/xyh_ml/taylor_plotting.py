"""
Helpers for plotting the distributions of Taylor coefficients (TCs).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from pandas import DataFrame

from xyh_ml.plotting import add_cms_label, get_style_color
from xyh_ml.taylor import get_tc_label, parse_tc_column

SUMMARY_COLUMNS = [
    "median",
    "q_low_68",
    "q_high_68",
    "q_68",
]
_QUANTILES = {
    "q_low_68": 0.16,
    "median": 0.5,
    "q_high_68": 0.84,
    "q_68": 0.68,
}


def summarize_coefficients(
    data_frame: DataFrame, columns: list[str], absolute: bool = False
) -> DataFrame:
    """
    Unweighted median, central 68% interval and 68% quantile of each column,
    or of the absolute values of each column if `absolute` is set.

    Returns
    -------
    DataFrame
        Indexed by column name with the columns `median`, `q_low_68`,
        `q_high_68` (16% / 84% quantiles) and `q_68` (68% quantile, upper
        edge of the one-sided 68% interval), sorted by the absolute median in
        descending order (stable).
    """
    if len(data_frame) == 0 or len(columns) == 0:
        return DataFrame(columns=SUMMARY_COLUMNS, dtype=float)
    values = data_frame[list(columns)].to_numpy(dtype=float)
    if absolute:
        values = np.abs(values)
    rows = {
        name: np.nanquantile(values, q, axis=0)
        for name, q in _QUANTILES.items()
    }
    summary = DataFrame(rows, index=list(columns))[SUMMARY_COLUMNS]
    order = np.argsort(-np.abs(summary["median"].to_numpy()), kind="stable")
    return summary.iloc[order]


def paginate(summary: DataFrame, per_page: int = 15) -> list[DataFrame]:
    """
    Split `summary` into consecutive chunks of at most `per_page` rows.
    """
    if per_page < 1:
        raise ValueError("per_page must be positive")
    return [
        summary.iloc[i : i + per_page]
        for i in range(0, len(summary), per_page)
    ]


def plot_coefficients(
    summary: DataFrame,
    rlabel: str,
    title: str | None = None,
    show_intervals: bool = True,
    absolute: bool = False,
) -> tuple[Figure, Axes]:
    """
    Plot the median and, if `show_intervals` is set, the 68% interval of the
    coefficients in `summary` (one entry per x position, in the order of the
    rows). Set `absolute` if `summary` describes absolute values; the
    interval is then one-sided, from 0 to the 68% quantile, with the part
    from 0 to the 16% quantile in a second color, otherwise it is the
    central interval between the 16% and 84% quantiles.
    """
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.subplots_adjust(bottom=0.3, left=0.12, right=0.97)
    color = get_style_color(0)
    x = np.arange(len(summary))

    ax.axhline(0.0, color="gray", linewidth=1.0, zorder=0)
    if show_intervals:
        _plot_intervals(ax, x, summary, color, one_sided=absolute)
    ax.plot(
        x,
        summary["median"],
        linestyle="none",
        marker="_" if show_intervals else "o",
        markersize=18 if show_intervals else 9,
        markeredgewidth=2.5 if show_intervals else 1.0,
        color="black" if show_intervals else color,
        label="Median of |TC|" if absolute else "Median",
        zorder=3,
    )

    labels = [get_tc_label(parse_tc_column(c)[1]) for c in summary.index]
    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=14,
    )
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.set_xlim(-0.7, max(len(summary), 1) - 0.3)
    ax.set_ylabel("|Taylor coefficient|" if absolute else "Taylor coefficient")
    ymin, ymax = ax.get_ylim()
    if absolute:
        ymin = 0.0
    ax.set_ylim(ymin, ymax + 0.4 * (ymax - ymin))
    ax.legend(
        loc="upper right",
        ncol=2,
        fontsize=14,
        title=title,
        title_fontsize=14,
    )
    add_cms_label(ax, rlabel)
    return fig, ax


def _plot_intervals(
    ax: Axes,
    x: np.ndarray,
    summary: DataFrame,
    color: str,
    one_sided: bool = False,
) -> None:
    # 68% intervals as boxes: one-sided [0, q_68], with the lowest part
    # [0, q_16] highlighted, or central [q_16, q_84]
    if one_sided:
        q_16 = summary["q_low_68"].to_numpy()
        q_50 = summary["median"].to_numpy()
        q_84 = summary["q_high_68"].to_numpy()
        _draw_boxes(
            ax,
            x,
            np.zeros(len(summary)),
            q_16,
            color,
            "16% percentile",
            alpha=0.8,
        )
        _draw_boxes(
            ax,
            x,
            q_16,
            q_50,
            color,
            "50% percentile",
            alpha=0.5,
        )
        _draw_boxes(
            ax,
            x,
            q_50,
            q_84,
            color,
            "84% percentile",
            alpha=0.3,
        )
    else:
        _draw_boxes(
            ax,
            x,
            summary["q_low_68"].to_numpy(),
            summary["q_high_68"].to_numpy(),
            color,
            "68% interval",
        )


def _draw_boxes(
    ax: Axes,
    x: np.ndarray,
    lows: np.ndarray,
    highs: np.ndarray,
    color: str,
    label: str,
    alpha: float = 0.8,
    width: float = 0.5,
) -> None:
    for xi, lo, hi in zip(x, lows, highs):
        ax.add_patch(
            Rectangle(
                (xi - width / 2, lo),
                width,
                hi - lo,
                facecolor=color,
                edgecolor=color,
                alpha=alpha,
                zorder=2,
            )
        )
    # Proxy artist for the legend
    ax.add_patch(
        Rectangle((0, 0), 0, 0, facecolor=color, alpha=alpha, label=label)
    )
