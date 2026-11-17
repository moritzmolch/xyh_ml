"""
Tests of the ranking of the input variables by their Taylor coefficients:
`xyh_ml.taylor.iter_diagonal_sets`, `xyh_ml.taylor_ranking` and the CLI
`xyh_ml.cli.rank_taylor_variables`.
"""

import logging
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir

from xyh_ml import PROJECT_DIR
from xyh_ml.cli import rank_taylor_variables
from xyh_ml.config import Taylor
from xyh_ml.taylor import (
    get_tc_column,
    get_tc_index_list,
    iter_diagonal_sets,
)
from xyh_ml.taylor_ranking import (
    RANK_COLUMNS,
    collect_variable_ranks,
    format_markdown_table,
    get_best_variable_ranks,
    rank_variables_in_summary,
)

CONFIG_DIR = str(PROJECT_DIR / "config")
DECAY_MODE = "y2b_h2tau"
TAG = "test_tag"
CLASSES = ["sig", "bkg"]


def _compose(config_name: str, overrides: list[str] | None = None):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        return compose(config_name=config_name, overrides=overrides or [])


def _tc_frame(
    events: list[tuple],
    tc_values: dict[str, list | float],
) -> pd.DataFrame:
    """
    Data frame of events `(channel, m_x, m_y, true_index, predicted_index)`
    with the TC columns `tc_values` (a scalar is broadcast over the events).
    """
    n = len(events)
    data = pd.DataFrame(
        events,
        columns=[
            "channel",
            "m_x",
            "m_y",
            "true_class_index",
            "predicted_class_index",
        ],
    )
    for column, value in tc_values.items():
        data[column] = np.broadcast_to(np.asarray(value, dtype=float), n)
    return data


# ---------------------------------------------------------------------------
# iter_diagonal_sets
# ---------------------------------------------------------------------------


def _sets(frame, class_names=CLASSES, excluded=None):
    return {
        (channel, m_x, m_y, name): (events, columns)
        for channel, m_x, m_y, name, events, columns in iter_diagonal_sets(
            frame, class_names, excluded
        )
    }


def test_iter_diagonal_sets_selects_diagonal_events_and_node_columns():
    frame = _tc_frame(
        [
            ("et", 300, 60, 0, 0),
            ("et", 300, 60, 0, 1),  # misclassified
            ("et", 300, 60, 1, 0),  # misclassified
            ("et", 300, 60, 1, 1),
            ("et", 300, 60, 1, 1),
        ],
        {
            "tc__sig__a": 1.0,
            "tc__sig__a__b": 2.0,
            "tc__bkg__a": 3.0,
        },
    )

    sets = _sets(frame)

    assert set(sets) == {("et", 300, 60, "sig"), ("et", 300, 60, "bkg")}
    sig_events, sig_columns = sets[("et", 300, 60, "sig")]
    bkg_events, bkg_columns = sets[("et", 300, 60, "bkg")]
    assert sig_columns == ["tc__sig__a", "tc__sig__a__b"]
    assert bkg_columns == ["tc__bkg__a"]
    assert sig_events.index.tolist() == [0]
    assert bkg_events.index.tolist() == [3, 4]


def test_iter_diagonal_sets_excluded_inputs_drop_all_involving_columns():
    frame = _tc_frame(
        [("et", 300, 60, 0, 0)],
        {
            "tc__sig__a": 1.0,
            "tc__sig__m_x": 1.0,
            "tc__sig__a__m_x": 1.0,
            "tc__sig__m_x__m_x": 1.0,
            "tc__sig__a__b": 1.0,
        },
    )

    ((*_, columns),) = list(iter_diagonal_sets(frame, ["sig"], ["m_x"]))

    assert columns == ["tc__sig__a", "tc__sig__a__b"]


def test_iter_diagonal_sets_skips_set_if_all_columns_excluded():
    frame = _tc_frame([("et", 300, 60, 0, 0)], {"tc__sig__m_x": 1.0})

    assert list(iter_diagonal_sets(frame, ["sig"], ["m_x"])) == []


def test_iter_diagonal_sets_skips_class_without_columns_or_events():
    frame = _tc_frame(
        [("et", 300, 60, 0, 0), ("et", 300, 60, 1, 0)],
        # bkg has columns but no diagonal events; a third class has events
        # but no columns
        {"tc__sig__a": 1.0, "tc__bkg__a": 1.0},
    )

    sets = _sets(frame, ["sig", "bkg", "other"])

    assert set(sets) == {("et", 300, 60, "sig")}


def test_iter_diagonal_sets_groups_by_channel_and_mass_point():
    events = [
        ("et", 300, 60, 0, 0),
        ("et", 300, 60, 0, 0),
        ("et", 500, 100, 0, 0),
        ("tt", 300, 60, 0, 0),
        ("tt", 300, 150, 0, 0),
    ]
    frame = _tc_frame(events, {"tc__sig__a": np.arange(5.0)})

    sets = _sets(frame, ["sig"])

    assert {k: len(v[0]) for k, v in sets.items()} == {
        ("et", 300, 60, "sig"): 2,
        ("et", 500, 100, "sig"): 1,
        ("tt", 300, 60, "sig"): 1,
        ("tt", 300, 150, "sig"): 1,
    }
    # events of a set come from its own group only
    assert sets[("tt", 300, 150, "sig")][0]["tc__sig__a"].tolist() == [4.0]


# ---------------------------------------------------------------------------
# rank_variables_in_summary
# ---------------------------------------------------------------------------


def _summary(medians: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"median": list(medians.values())}, index=list(medians)
    )


def test_rank_variables_in_summary_rank_is_row_position():
    summary = _summary({"tc__sig__b": 5.0, "tc__sig__a": 2.0})

    ranks = rank_variables_in_summary(summary)

    assert ranks[
        ["variable", "rank", "median", "coefficient", "order"]
    ].to_dict("records") == [
        {
            "variable": "b",
            "rank": 1,
            "median": 5.0,
            "coefficient": "b",
            "order": 1,
        },
        {
            "variable": "a",
            "rank": 2,
            "median": 2.0,
            "coefficient": "a",
            "order": 1,
        },
    ]


def test_rank_variables_in_summary_second_order_gives_row_per_variable():
    ranks = rank_variables_in_summary(_summary({"tc__sig__a__b": 3.0}))

    assert ranks["variable"].tolist() == ["a", "b"]
    assert (ranks["rank"] == 1).all()
    assert (ranks["order"] == 2).all()
    assert (ranks["coefficient"] == "a × b").all()
    assert (ranks["median"] == 3.0).all()


def test_rank_variables_in_summary_diagonal_gives_single_row():
    ranks = rank_variables_in_summary(
        _summary({"tc__sig__a__a": 3.0, "tc__sig__a": 1.0})
    )

    assert ranks["variable"].tolist() == ["a", "a"]
    assert ranks["rank"].tolist() == [1, 2]
    assert ranks["order"].tolist() == [2, 1]
    assert ranks["coefficient"].tolist() == ["a × a", "a"]


def test_rank_variables_in_summary_empty():
    ranks = rank_variables_in_summary(_summary({}))

    assert ranks.empty
    assert list(ranks.columns) == [
        "variable",
        "rank",
        "median",
        "coefficient",
        "order",
    ]


# ---------------------------------------------------------------------------
# collect_variable_ranks
# ---------------------------------------------------------------------------


def _single_set_frame(**tc_values):
    return _tc_frame([("et", 300, 60, 0, 0)] * 4, tc_values)


def test_collect_variable_ranks_known_answer():
    frame = _single_set_frame(
        tc__sig__a=-10.0,
        tc__sig__b=4.0,
        tc__sig__a__b=7.0,
        tc__sig__a__a=1.0,
    )

    ranks = collect_variable_ranks(frame, ["sig"])

    assert list(ranks.columns) == RANK_COLUMNS
    rows = {
        (r.variable, r.coefficient): (r["rank"], r.order, r["median"])
        for _, r in ranks.iterrows()
    }
    assert rows == {
        ("a", "a"): (1, 1, 10.0),
        ("a", "a × b"): (2, 2, 7.0),
        ("b", "a × b"): (2, 2, 7.0),
        ("b", "b"): (3, 1, 4.0),
        ("a", "a × a"): (4, 2, 1.0),
    }
    assert set(ranks["channel"]) == {"et"}
    assert set(ranks["output_class"]) == {"sig"}
    assert set(ranks["m_x"]) == {300}
    assert set(ranks["m_y"]) == {60}
    assert set(ranks["n_events"]) == {4}
    assert set(ranks["n_coefficients"]) == {4}


def test_collect_variable_ranks_covers_all_sets():
    frame = _tc_frame(
        [("et", 300, 60, 0, 0), ("tt", 500, 100, 1, 1)],
        {"tc__sig__a": 1.0, "tc__bkg__b": 2.0},
    )

    ranks = collect_variable_ranks(frame, CLASSES)

    assert sorted(
        zip(ranks["channel"], ranks["output_class"], ranks["variable"])
    ) == [("et", "sig", "a"), ("tt", "bkg", "b")]


def test_collect_variable_ranks_excluded_inputs():
    frame = _single_set_frame(tc__sig__m_x=100.0, tc__sig__a=1.0)

    ranks = collect_variable_ranks(frame, ["sig"], ["m_x"])

    assert ranks["variable"].tolist() == ["a"]
    assert ranks["rank"].tolist() == [1]


@pytest.mark.parametrize(
    ("absolute", "first"),
    [(True, "x"), (False, "y")],
    ids=["median_of_abs", "abs_of_median"],
)
def test_collect_variable_ranks_absolute_flag_changes_ranking(absolute, first):
    # x = +-10 (median of |x| = 10, median of x = 0), y = 3
    frame = _single_set_frame(
        tc__sig__x=[-10.0, 10.0, -10.0, 10.0], tc__sig__y=3.0
    )

    ranks = collect_variable_ranks(frame, ["sig"], absolute=absolute)

    assert ranks.set_index("variable")["rank"].idxmin() == first
    assert ranks.loc[ranks["rank"] == 1, "variable"].tolist() == [first]


@pytest.mark.parametrize(
    "frame",
    [
        _tc_frame([], {"tc__sig__a": 1.0}),
        _single_set_frame(tc__bkg__a=1.0),  # no columns of the class
        _tc_frame([("et", 300, 60, 1, 0)], {"tc__sig__a": 1.0}),  # no events
    ],
    ids=["empty_frame", "no_columns", "no_diagonal_events"],
)
def test_collect_variable_ranks_empty_input_gives_empty_frame(frame):
    ranks = collect_variable_ranks(frame, ["sig"])

    assert ranks.empty
    assert list(ranks.columns) == RANK_COLUMNS


# ---------------------------------------------------------------------------
# get_best_variable_ranks
# ---------------------------------------------------------------------------


def _rank_row(
    variable,
    rank,
    median=1.0,
    channel="et",
    output_class="sig",
    m_x=300.0,
    m_y=60.0,
    order=1,
    coefficient=None,
    n_events=5,
    n_coefficients=5,
):
    return {
        "variable": variable,
        "rank": rank,
        "median": median,
        "channel": channel,
        "output_class": output_class,
        "m_x": m_x,
        "m_y": m_y,
        "coefficient": coefficient or variable,
        "order": order,
        "n_events": n_events,
        "n_coefficients": n_coefficients,
    }


def _ranks(*rows):
    return pd.DataFrame(list(rows), columns=RANK_COLUMNS)


def _best_row(best, variable):
    return best.set_index("variable").loc[variable]


def test_get_best_variable_ranks_takes_minimum_over_sets():
    ranks = _ranks(
        _rank_row("a", 3, output_class="sig"),
        _rank_row("a", 1, output_class="bkg", m_x=500.0, m_y=100.0),
        _rank_row("a", 2, output_class="z"),
    )

    best = get_best_variable_ranks(ranks)

    assert len(best) == 1
    row = best.iloc[0]
    assert row["rank"] == 1
    assert row["output_class"] == "bkg"
    assert (row["m_x"], row["m_y"]) == (500.0, 100.0)
    assert row["n_sets_best_rank"] == 1


def test_get_best_variable_ranks_reached_via_second_order_tc():
    ranks = _ranks(
        _rank_row("b", 5, order=1),
        _rank_row("b", 1, order=2, coefficient="a × b", output_class="bkg"),
    )

    row = get_best_variable_ranks(ranks).iloc[0]

    assert row["rank"] == 1
    assert row["order"] == 2
    assert row["coefficient"] == "a × b"
    assert row["best_rank_first_order"] == 5


def test_get_best_variable_ranks_first_order_missing_is_na():
    ranks = _ranks(_rank_row("b", 1, order=2, coefficient="a × b"))

    row = get_best_variable_ranks(ranks).iloc[0]

    assert pd.isna(row["best_rank_first_order"])
    assert row["rank"] == 1


def test_get_best_variable_ranks_tie_broken_by_larger_abs_median():
    ranks = _ranks(
        _rank_row("a", 2, median=0.5, output_class="sig"),
        _rank_row("a", 2, median=-3.0, output_class="bkg"),
        _rank_row("a", 2, median=1.0, output_class="z"),
    )

    row = get_best_variable_ranks(ranks).iloc[0]

    assert row["output_class"] == "bkg"
    assert row["median"] == -3.0
    assert row["n_sets_best_rank"] == 3


def test_get_best_variable_ranks_full_tie_keeps_input_order():
    ranks = _ranks(
        _rank_row("a", 2, median=1.0, output_class="first"),
        _rank_row("a", 2, median=1.0, output_class="second"),
    )

    assert get_best_variable_ranks(ranks).iloc[0]["output_class"] == "first"


def test_get_best_variable_ranks_counts_sets_at_best_rank_only():
    ranks = _ranks(
        _rank_row("a", 1, output_class="sig"),
        _rank_row("a", 1, output_class="bkg"),
        _rank_row("a", 4, output_class="z"),
        _rank_row("b", 2, output_class="sig"),
        _rank_row("b", 3, output_class="bkg"),
    )

    best = get_best_variable_ranks(ranks)

    assert _best_row(best, "a")["n_sets_best_rank"] == 2
    assert _best_row(best, "b")["n_sets_best_rank"] == 1


def test_get_best_variable_ranks_per_channel_columns():
    ranks = _ranks(
        _rank_row("a", 4, channel="et"),
        _rank_row("a", 2, channel="et", output_class="bkg"),
        _rank_row("a", 1, channel="tt"),
        _rank_row("b", 3, channel="et"),
    )

    best = get_best_variable_ranks(ranks)

    assert {"best_rank_et", "best_rank_tt"} <= set(best.columns)
    a, b = _best_row(best, "a"), _best_row(best, "b")
    assert (a["best_rank_et"], a["best_rank_tt"]) == (2, 1)
    assert b["best_rank_et"] == 3
    assert pd.isna(b["best_rank_tt"])
    assert a["channel"] == "tt"


def test_get_best_variable_ranks_variables_without_rank_sort_last():
    ranks = _ranks(_rank_row("a", 7), _rank_row("b", 2))

    best = get_best_variable_ranks(ranks, ["none1", "a", "none2", "b"])

    assert best["variable"].tolist() == ["b", "a", "none1", "none2"]
    missing = _best_row(best, "none1")
    assert pd.isna(missing["rank"])
    assert pd.isna(missing["channel"])
    assert pd.isna(missing["best_rank_et"])
    assert pd.isna(missing["n_sets_best_rank"])


def test_get_best_variable_ranks_ignores_variables_not_requested():
    ranks = _ranks(_rank_row("a", 1), _rank_row("other", 2))

    best = get_best_variable_ranks(ranks, ["a"])

    assert best["variable"].tolist() == ["a"]


def test_get_best_variable_ranks_sorted_by_rank_then_abs_median():
    ranks = _ranks(
        _rank_row("low_median", 1, median=1.0),
        _rank_row("high_median", 1, median=-5.0),
        _rank_row("worse", 2, median=100.0),
    )

    best = get_best_variable_ranks(ranks)

    assert best["variable"].tolist() == ["high_median", "low_median", "worse"]
    assert str(best["rank"].dtype) == "Int64"


# ---------------------------------------------------------------------------
# format_markdown_table
# ---------------------------------------------------------------------------


def _table(absolute=True, variables=None):
    ranks = _ranks(
        _rank_row(
            "a",
            1,
            median=0.123456,
            channel="et",
            m_x=300.0,
            m_y=60.0,
            order=2,
            coefficient="a × b",
        ),
        _rank_row("a", 3, channel="tt", order=1),
        _rank_row("b", 2, channel="tt", m_x=1000.0, m_y=800.0),
    )
    best = get_best_variable_ranks(ranks, variables or ["a", "b", "c"])
    return format_markdown_table(best, absolute=absolute).splitlines()


def test_format_markdown_table_header_and_separator():
    lines = _table()

    assert lines[0] == (
        "| Variable | Best rank | Channel | Class | m_X, m_Y [GeV] | "
        "Coefficient | Median \\|TC\\| | # sets | Best rank (1st order) | "
        "Best rank (et) | Best rank (tt) |"
    )
    assert lines[1] == "|" + "---|" * 11


def test_format_markdown_table_median_label_without_absolute():
    assert "| Median TC |" in _table(absolute=False)[0]
    assert "Median \\|TC\\|" not in _table(absolute=False)[0]


def test_format_markdown_table_one_row_per_variable():
    lines = _table()

    assert len(lines) == 2 + 3
    assert [line.split("|")[1].strip() for line in lines[2:]] == [
        "`a`",
        "`b`",
        "`c`",
    ]


def test_format_markdown_table_cell_contents():
    lines = _table()

    cells = [c.strip() for c in lines[2].strip("|").split("|")]
    assert cells == [
        "`a`",
        "1",
        "et",
        "sig",
        "300, 60",
        "a × b",
        "0.123",
        "1",
        "3",
        "1",
        "3",
    ]
    assert "1000, 800" in lines[3]


def test_format_markdown_table_missing_values_shown_as_dash():
    lines = _table()

    cells = [c.strip() for c in lines[4].strip("|").split("|")]
    assert cells == ["`c`"] + ["–"] * 10
    # first-order and per-channel NA of a variable that has a rank
    b_cells = [c.strip() for c in lines[3].strip("|").split("|")]
    assert b_cells[-2:] == ["–", "2"]


def test_format_markdown_table_ends_with_newline():
    best = get_best_variable_ranks(_ranks(_rank_row("a", 1)))

    assert format_markdown_table(best).endswith("|\n")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_rank_config_composes_with_default_ranking_channels():
    cfg = _compose("rank_taylor_variables")

    assert list(cfg.taylor.ranking_channels) == ["et", "mt", "tt"]
    assert "${" not in str(cfg.artifacts.taylor_dir)
    assert cfg.artifacts.taylor_dir.endswith(f"/{cfg.mode.channel}")


def test_ranking_channels_can_be_overridden():
    cfg = _compose("rank_taylor_variables", ["taylor.ranking_channels=[tt]"])

    assert list(cfg.taylor.ranking_channels) == ["tt"]


def test_ranking_channels_dataclass_default_matches_yaml():
    default = Taylor(mass_points=[])
    cfg = _compose("rank_taylor_variables")

    assert default.ranking_channels == list(cfg.taylor.ranking_channels)
    assert (
        Taylor(mass_points=[]).ranking_channels is not default.ranking_channels
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

PARAMETER = "m_x"
MASS_POINTS = [(300.0, 60.0), (500.0, 100.0)]
N_PER_FOLD = 3


def _compose_cli(tmp_path, monkeypatch, overrides=()):
    cfg = _compose(
        "rank_taylor_variables",
        [
            # One directory per channel and fold, as in artifacts.yaml
            "artifacts.taylor_dir="
            f"'{tmp_path}/taylor/${{mode.channel}}/fold${{mode.fold}}'",
            f"tags.training_tag={TAG}",
            *overrides,
        ],
    )
    monkeypatch.setattr(rank_taylor_variables, "PROJECT_DIR", tmp_path)
    return cfg


def _write_channel(cfg, channel, values, names, folds=None):
    """
    Write synthetic TC files of `channel` for all folds. Every event is of
    its class and predicted as such; the TC with the inputs `k` has the value
    `values[k]` (0.001 if missing).
    """
    classes = [c.name for c in cfg.outputs.classes]
    columns = {
        get_tc_column(node, idx, names): values.get(
            tuple(names[i] for i in idx), 0.001
        )
        for node in classes
        for idx in get_tc_index_list(len(names), 2)
    }
    for fold in range(cfg.mode.n_folds) if folds is None else folds:
        cfg_fold = deepcopy(cfg)
        cfg_fold.mode.channel = channel
        cfg_fold.mode.fold = fold
        frames = []
        for (m_x, m_y), (i, cls) in (
            (m, c) for m in MASS_POINTS for c in enumerate(classes)
        ):
            n = N_PER_FOLD
            data = {
                "class": [cls] * n,
                "channel": [channel] * n,
                "m_x": np.full(n, m_x),
                "m_y": np.full(n, m_y),
                "reduction_rate": np.full(n, 1 / cfg.mode.n_folds),
            }
            for j, c in enumerate(classes):
                data[f"score_{c}"] = np.full(n, 0.7 if i == j else 0.1)
            data.update({k: np.full(n, v) for k, v in columns.items()})
            frames.append(pd.DataFrame(data))
        path = (
            Path(cfg_fold.artifacts.taylor_dir)
            / f"taylor_coefficients_{DECAY_MODE}.h5"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).to_hdf(
            path, key="taylor", mode="w"
        )


@pytest.fixture
def cli_setup(tmp_path, monkeypatch):
    """
    Config with TC files of the channels et and tt (not mt), with the input
    variables v0, v1 and the parameter m_x.
    """

    def setup(overrides=()):
        cfg = _compose_cli(tmp_path, monkeypatch, overrides)
        v0, v1 = (v.name for v in cfg.inputs.variables[:2])
        names = [v0, v1, PARAMETER]
        et = {
            (v0,): 10.0,
            (v0, v1): 5.0,
            (v1,): 1.0,
            (v0, v0): 0.5,
            (v1, v1): 0.1,
            (PARAMETER,): 100.0,
        }
        tt = {
            (v1,): 10.0,
            (v0, v1): 5.0,
            (v0,): 1.0,
            (PARAMETER,): 100.0,
        }
        _write_channel(cfg, "et", et, names)
        _write_channel(cfg, "tt", tt, names)
        return cfg, v0, v1

    return setup


def _output_files(tmp_path):
    output_dir = tmp_path / "data" / "output" / TAG / "tables" / "taylor"
    stem = f"variable_ranks__{DECAY_MODE}__allfoldsof"
    return output_dir, stem


def test_cli_writes_tables_and_skips_missing_channel(
    tmp_path, cli_setup, caplog
):
    cfg, v0, v1 = cli_setup()
    stem = f"variable_ranks__{DECAY_MODE}__allfoldsof{cfg.mode.n_folds}"

    with caplog.at_level(logging.WARNING):
        rank_taylor_variables.main.__wrapped__(cfg)

    output_dir = tmp_path / "data/output" / TAG / "tables/taylor"
    assert sorted(p.name for p in output_dir.iterdir()) == [
        f"{stem}.csv",
        f"{stem}.md",
        f"{stem}__all.csv",
    ]
    assert "Skipping channel mt" in caplog.text

    markdown = (output_dir / stem).with_suffix(".md").read_text()
    assert "no TC files: mt" in markdown
    assert "channels: et, tt" in markdown
    assert "Median \\|TC\\|" in markdown
    assert "Best rank (et)" in markdown
    assert "Best rank (tt)" in markdown
    assert "Best rank (mt)" not in markdown

    best = pd.read_csv(output_dir / f"{stem}.csv").set_index("variable")
    # parameters are excluded, all input variables are listed
    assert list(best.index) == [v.name for v in cfg.inputs.variables][
        :0
    ] + list(best.index)
    assert PARAMETER not in best.index
    assert set(v.name for v in cfg.inputs.variables) == set(best.index)
    # v0 and v1 are first in the table with rank 1 each
    assert best.index[:2].tolist() == [v0, v1]
    assert best.loc[[v0, v1], "rank"].tolist() == [1, 1]
    assert best.loc[v0, "best_rank_et"] == 1
    assert best.loc[v0, "best_rank_tt"] == 2  # v0 x v1 (rank 2) in tt
    assert best.loc[v1, "best_rank_et"] == 2
    assert best.loc[v1, "best_rank_tt"] == 1
    # 4 classes x 2 mass points in the channel reaching rank 1
    n_sets = len(cfg.outputs.classes) * len(MASS_POINTS)
    assert best.loc[v0, "n_sets_best_rank"] == n_sets
    assert best.loc[v0, "best_rank_first_order"] == 1
    # other variables have no coefficients
    others = best.drop([v0, v1])
    assert others["rank"].isna().all()

    all_ranks = pd.read_csv(output_dir / f"{stem}__all.csv")
    assert list(all_ranks.columns) == RANK_COLUMNS
    assert set(all_ranks["channel"]) == {"et", "tt"}
    assert len(
        all_ranks.groupby(["channel", "m_x", "m_y", "output_class"])
    ) == (2 * n_sets)
    # no coefficient involves the parameter
    assert not all_ranks["coefficient"].str.contains(PARAMETER).any()
    # events of all folds and the single era are pooled
    assert (all_ranks["n_events"] == N_PER_FOLD * cfg.mode.n_folds).all()


def test_cli_include_parameters_ranks_parameter_too(tmp_path, cli_setup):
    cfg, v0, v1 = cli_setup(["taylor.include_parameters=true"])

    rank_taylor_variables.main.__wrapped__(cfg)

    output_dir, stem = _output_files(tmp_path)
    best = pd.read_csv(output_dir / f"{stem}{cfg.mode.n_folds}.csv").set_index(
        "variable"
    )
    assert best.index[0] == PARAMETER
    assert best.loc[PARAMETER, "rank"] == 1
    # m_x takes rank 1, so v0 is second in et and v1 is second in tt
    assert best.loc[v0, "best_rank_et"] == 2
    assert best.loc[v1, "best_rank_tt"] == 2
    all_ranks = pd.read_csv(output_dir / f"{stem}{cfg.mode.n_folds}__all.csv")
    assert all_ranks["coefficient"].str.contains(PARAMETER).any()


def test_cli_only_requested_channels_are_used(tmp_path, cli_setup):
    cfg, v0, _ = cli_setup(["taylor.ranking_channels=[tt]"])

    rank_taylor_variables.main.__wrapped__(cfg)

    output_dir, stem = _output_files(tmp_path)
    best = pd.read_csv(output_dir / f"{stem}{cfg.mode.n_folds}.csv")
    assert "best_rank_tt" in best.columns
    assert "best_rank_et" not in best.columns


def test_cli_raises_if_no_channel_has_files(tmp_path, monkeypatch):
    cfg = _compose_cli(tmp_path, monkeypatch)

    with pytest.raises(FileNotFoundError, match="No Taylor coefficient"):
        rank_taylor_variables.main.__wrapped__(cfg)

    assert not (tmp_path / "data").exists()


def test_cli_incomplete_folds_count_as_missing_channel(tmp_path, monkeypatch):
    cfg = _compose_cli(tmp_path, monkeypatch)
    names = ["a", "b"]
    _write_channel(cfg, "et", {("a",): 1.0}, names, folds=[0])

    with pytest.raises(FileNotFoundError, match="No Taylor coefficient"):
        rank_taylor_variables.main.__wrapped__(cfg)


def test_load_channel_ranks_returns_none_for_missing_channel(
    tmp_path, monkeypatch
):
    cfg = _compose_cli(tmp_path, monkeypatch)

    assert rank_taylor_variables.load_channel_ranks(cfg, "mt") is None


def test_load_channel_ranks_does_not_change_config(tmp_path, cli_setup):
    cfg, v0, _ = cli_setup()
    channel, fold = cfg.mode.channel, cfg.mode.fold

    ranks = rank_taylor_variables.load_channel_ranks(cfg, "tt")

    assert (cfg.mode.channel, cfg.mode.fold) == (channel, fold)
    assert set(ranks["channel"]) == {"tt"}
    assert list(ranks.columns) == RANK_COLUMNS
    assert ranks.loc[ranks["rank"] == 1, "variable"].unique().tolist() != []
