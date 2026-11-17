import matplotlib
import numpy as np
import pandas as pd
import pytest

from xyh_ml.taylor_plotting import (
    SUMMARY_COLUMNS,
    paginate,
    plot_coefficients,
    summarize_coefficients,
)

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


def test_summarize_coefficients_quantiles():
    df = pd.DataFrame({"tc__n__a": np.arange(1001, dtype=float)})

    summary = summarize_coefficients(df, ["tc__n__a"])

    row = summary.loc["tc__n__a"]
    assert row["median"] == pytest.approx(500)
    assert row["q_low_68"] == pytest.approx(160)
    assert row["q_high_68"] == pytest.approx(840)
    assert row["q_68"] == pytest.approx(680)
    assert list(summary.columns) == SUMMARY_COLUMNS


def test_summarize_coefficients_sorted_by_absolute_median():
    df = pd.DataFrame(
        {
            "small": np.full(5, 0.1),
            "neg_large": np.full(5, -3.0),
            "mid": np.full(5, 2.0),
        }
    )

    summary = summarize_coefficients(df, ["small", "neg_large", "mid"])

    assert list(summary.index) == ["neg_large", "mid", "small"]
    assert summary.loc["neg_large", "median"] == -3.0


def test_summarize_coefficients_ties_keep_input_order():
    df = pd.DataFrame({"a": [1.0, 1.0], "b": [-1.0, -1.0], "c": [1.0, 1.0]})

    summary = summarize_coefficients(df, ["a", "b", "c"])

    assert list(summary.index) == ["a", "b", "c"]


def test_summarize_coefficients_ignores_nan():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    assert summarize_coefficients(df, ["a"]).loc["a", "median"] == 2.0


def test_summarize_coefficients_empty_inputs():
    df = pd.DataFrame({"a": [1.0]})
    for summary in (
        summarize_coefficients(df.iloc[:0], ["a"]),
        summarize_coefficients(df, []),
    ):
        assert len(summary) == 0
        assert list(summary.columns) == SUMMARY_COLUMNS


def test_paginate_sizes():
    summary = pd.DataFrame({"median": np.arange(35.0)})
    pages = paginate(summary, 15)
    assert [len(p) for p in pages] == [15, 15, 5]
    assert pd.concat(pages).equals(summary)


def test_paginate_empty_and_exact():
    assert paginate(pd.DataFrame({"m": []}), 15) == []
    assert [len(p) for p in paginate(pd.DataFrame({"m": range(30)}), 15)] == [
        15,
        15,
    ]


@pytest.mark.parametrize("per_page", [0, -3])
def test_paginate_invalid_per_page_raises(per_page):
    with pytest.raises(ValueError, match="per_page"):
        paginate(pd.DataFrame({"m": [1]}), per_page)


@pytest.mark.parametrize("n", [1, 5, 15])
def test_plot_coefficients_ticks(n):
    names = [f"tc__node__v_{i}__w" for i in range(n)]
    df = pd.DataFrame(
        {c: np.linspace(-1, 1, 21) * (i + 1) for i, c in enumerate(names)}
    )
    summary = summarize_coefficients(df, names)

    fig, ax = plot_coefficients(
        summary, rlabel=r"$\mathrm{test}$", title="title"
    )
    try:
        assert len(ax.get_xticks()) == n
        assert len(ax.get_xticklabels()) == n
    finally:
        plt.close(fig)

    assert not plt.fignum_exists(fig.number)


def test_summarize_coefficients_absolute_ranks_by_median_of_abs():
    # Median of "sym" is 0, but its absolute values are large; "pos" has a
    # larger median than "sym" but smaller absolute values.
    df = pd.DataFrame(
        {
            "pos": [0.5, 0.5, 0.5, 0.5, 0.5],
            "sym": [-3.0, -2.0, 0.0, 2.0, 3.0],
            "neg": [-1.0, -1.0, -1.0, -1.0, -1.0],
        }
    )

    signed = summarize_coefficients(df, ["pos", "sym", "neg"])
    assert list(signed.index) == ["neg", "pos", "sym"]

    summary = summarize_coefficients(df, ["pos", "sym", "neg"], absolute=True)
    assert list(summary.index) == ["sym", "neg", "pos"]
    assert summary["median"].tolist() == [2.0, 1.0, 0.5]
    assert (summary[SUMMARY_COLUMNS].to_numpy() >= 0).all()


@pytest.mark.parametrize("show_intervals", [True, False])
def test_plot_coefficients_show_intervals(show_intervals):
    names = [f"tc__node__v_{i}" for i in range(4)]
    df = pd.DataFrame(
        {c: np.linspace(-1, 1, 21) * (i + 1) for i, c in enumerate(names)}
    )
    summary = summarize_coefficients(df, names, absolute=True)

    fig, ax = plot_coefficients(
        summary,
        rlabel="test",
        show_intervals=show_intervals,
        absolute=True,
    )
    try:
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        has_intervals = any("interval" in label for label in labels)
        assert has_intervals == show_intervals
        assert not any("95%" in label for label in labels)
        if show_intervals:
            assert "68% interval (one-sided)" in labels
            assert "0–16% quantile" in labels
            boxes = [p for p in ax.patches if p.get_width() > 0]
            assert len(boxes) == 2 * len(names)
            lower, upper = boxes[: len(names)], boxes[len(names) :]
            assert lower[0].get_facecolor() != upper[0].get_facecolor()
            for low, up, (_, row) in zip(lower, upper, summary.iterrows()):
                assert low.get_y() == 0.0
                assert low.get_height() == pytest.approx(row["q_low_68"])
                assert up.get_y() == pytest.approx(row["q_low_68"])
                assert up.get_y() + up.get_height() == pytest.approx(
                    row["q_68"]
                )
        assert ax.get_ylim()[0] == 0.0
        assert ax.get_ylabel() == "|Taylor coefficient|"
    finally:
        plt.close(fig)
