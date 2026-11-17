"""
Ranking of the input variables by their Taylor coefficients (TCs).

Within each diagonal TC set (channel x mass point x class `c`: the TCs of the
output node `c` for events of true and predicted class `c`, see
`xyh_ml.taylor.iter_diagonal_sets`), all TCs are ranked by their median
(of |TC| by default), rank 1 being the largest. A variable takes the rank of
its own first-order TC and of every second-order TC that it is part of. Its
best rank is the minimum over all sets.
"""

import numpy as np
import pandas as pd
from pandas import DataFrame

from xyh_ml.taylor import get_tc_label, iter_diagonal_sets, parse_tc_column
from xyh_ml.taylor_plotting import summarize_coefficients

RANK_COLUMNS = [
    "variable",
    "rank",
    "median",
    "channel",
    "output_class",
    "m_x",
    "m_y",
    "coefficient",
    "order",
    "n_events",
    "n_coefficients",
]


def rank_variables_in_summary(summary: DataFrame) -> DataFrame:
    """
    Rank the variables of one TC set.

    Parameters
    ----------
    summary
        Output of `summarize_coefficients`, indexed by TC column name and
        sorted by the ranking (first row = rank 1).

    Returns
    -------
    DataFrame
        One row per TC and variable involved in it (one row for the diagonal
        second-order TC of a variable) with the columns `variable`, `rank`,
        `median`, `coefficient` and `order`.
    """
    rows = []
    for rank, (column, median) in enumerate(
        summary["median"].items(), start=1
    ):
        inputs = parse_tc_column(column)[1]
        for variable in dict.fromkeys(inputs):
            rows.append(
                {
                    "variable": variable,
                    "rank": rank,
                    "median": median,
                    "coefficient": get_tc_label(inputs),
                    "order": len(inputs),
                }
            )
    return DataFrame(
        rows, columns=["variable", "rank", "median", "coefficient", "order"]
    )


def collect_variable_ranks(
    data_frame: DataFrame,
    class_names: list[str],
    excluded_inputs: list[str] | None = None,
    absolute: bool = True,
) -> DataFrame:
    """
    Ranks of all variables in all diagonal TC sets of `data_frame`.

    Parameters
    ----------
    data_frame
        Events with TC columns and class indices, see `iter_diagonal_sets`.
    class_names
        Output class names in the order of the class indices.
    excluded_inputs
        Inputs whose TCs are ignored (e.g. the parameters m_x, m_y).
    absolute
        Rank by the median of |TC| instead of the absolute median of TC.

    Returns
    -------
    DataFrame
        Long table with the columns `RANK_COLUMNS`.
    """
    tables = []
    for channel, m_x, m_y, class_name, events, columns in iter_diagonal_sets(
        data_frame, class_names, excluded_inputs
    ):
        summary = summarize_coefficients(events, columns, absolute=absolute)
        table = rank_variables_in_summary(summary).assign(
            channel=channel,
            output_class=class_name,
            m_x=m_x,
            m_y=m_y,
            n_events=len(events),
            n_coefficients=len(columns),
        )
        tables.append(table)
    if not tables:
        return DataFrame(columns=RANK_COLUMNS)
    return pd.concat(tables, ignore_index=True)[RANK_COLUMNS]


def get_best_variable_ranks(
    ranks: DataFrame, variables: list[str] | None = None
) -> DataFrame:
    """
    Best (lowest) rank of each variable over all TC sets in `ranks`.

    Ties are broken by the larger |median|, then by the order in `ranks`.

    Parameters
    ----------
    ranks
        Output of `collect_variable_ranks` (possibly of several channels).
    variables
        Variables to report. Variables without any rank get empty entries.
        Defaults to the variables in `ranks`.

    Returns
    -------
    DataFrame
        One row per variable, sorted by the best rank and then by the
        |median|, with the columns of `ranks` for the set of the best rank,
        `n_sets_best_rank` (number of sets reaching the best rank),
        `best_rank_first_order` (best rank of the first-order TC alone) and
        `best_rank_<channel>` for every channel in `ranks`.
    """
    if variables is None:
        variables = list(dict.fromkeys(ranks["variable"]))
    ranks = ranks.assign(abs_median=ranks["median"].abs())
    best = (
        ranks.sort_values(
            ["rank", "abs_median"], ascending=[True, False], kind="stable"
        )
        .groupby("variable", sort=False)
        .head(1)
        .set_index("variable")
    )
    is_best = ranks["rank"].to_numpy() == best["rank"].reindex(
        ranks["variable"]
    ).to_numpy(dtype=float)
    best["n_sets_best_rank"] = ranks.loc[is_best].groupby("variable").size()
    best["best_rank_first_order"] = (
        ranks.loc[ranks["order"] == 1].groupby("variable")["rank"].min()
    )
    for channel in sorted(ranks["channel"].unique()):
        best[f"best_rank_{channel}"] = (
            ranks.loc[ranks["channel"] == channel]
            .groupby("variable")["rank"]
            .min()
        )

    best = best.reindex(variables)
    best.index.name = "variable"
    best = best.reset_index()
    order = np.lexsort(
        (
            -best["abs_median"].fillna(-np.inf).to_numpy(),
            best["rank"].fillna(np.inf).to_numpy(dtype=float),
        )
    )
    best = best.iloc[order].drop(columns="abs_median")

    # Integer columns with missing values
    int_columns = ["rank", "order", "n_events", "n_coefficients"] + [
        c for c in best.columns if c.startswith(("best_rank", "n_sets"))
    ]
    return best.astype({c: "Int64" for c in int_columns}).reset_index(
        drop=True
    )


def format_markdown_table(best: DataFrame, absolute: bool = True) -> str:
    """
    Format the output of `get_best_variable_ranks` as a Markdown table. Set
    `absolute` if the medians are medians of |TC|.
    """
    channels = [
        c.removeprefix("best_rank_")
        for c in best.columns
        if c.startswith("best_rank_") and c != "best_rank_first_order"
    ]
    header = [
        "Variable",
        "Best rank",
        "Channel",
        "Class",
        "m_X, m_Y [GeV]",
        "Coefficient",
        r"Median \|TC\|" if absolute else "Median TC",
        "# sets",
        "Best rank (1st order)",
    ] + [f"Best rank ({c})" for c in channels]

    def fmt(value, spec=""):
        if pd.isna(value):
            return "–"
        return format(value, spec)

    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    for _, row in best.iterrows():
        mass_point = (
            "–" if pd.isna(row["m_x"]) else f"{row['m_x']:g}, {row['m_y']:g}"
        )
        cells = [
            f"`{row['variable']}`",
            fmt(row["rank"]),
            fmt(row["channel"]),
            fmt(row["output_class"]),
            mass_point,
            fmt(row["coefficient"]),
            fmt(row["median"], ".3g"),
            fmt(row["n_sets_best_rank"]),
            fmt(row["best_rank_first_order"]),
        ] + [fmt(row[f"best_rank_{c}"]) for c in channels]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
