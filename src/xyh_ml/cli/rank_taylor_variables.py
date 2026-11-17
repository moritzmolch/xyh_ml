"""
Rank the input variables by their Taylor coefficients (TCs).

For every input variable, find its best rank over all diagonal TC sets
(channel x mass point x class, all eras and folds pooled) of the channels in
`taylor.ranking_channels`, either through its first-order TC or as part of a
second-order TC (see `xyh_ml.taylor_ranking`). Needs the outputs of
`evaluate_taylor_coefficients` for every fold; channels without outputs are
skipped. Writes to `data/output/<training_tag>/tables/taylor/`:

- `variable_ranks__<decay_mode>__<fold_suffix>.md`: Markdown table,
- `variable_ranks__<decay_mode>__<fold_suffix>.csv`: the same as CSV,
- `variable_ranks__<decay_mode>__<fold_suffix>__all.csv`: the rank of every
  variable in every TC set.
"""

import logging
from copy import deepcopy
from datetime import date

import hydra
import pandas as pd

from xyh_ml import PROJECT_DIR
from xyh_ml.config import Config
from xyh_ml.plotting import (
    add_class_indices,
    get_fold_label,
    load_predictions,
)
from xyh_ml.taylor import get_taylor_coefficients_file
from xyh_ml.taylor_ranking import (
    collect_variable_ranks,
    format_markdown_table,
    get_best_variable_ranks,
)

# Set up logging
logger = logging.getLogger(__name__)


def load_channel_ranks(cfg: Config, channel: str) -> pd.DataFrame | None:
    """
    Ranks of all variables in all diagonal TC sets of `channel`, or None if
    the TC files of the channel are missing.
    """
    cfg_channel = deepcopy(cfg)
    cfg_channel.mode.channel = channel
    try:
        data_frame = load_predictions(
            cfg_channel,
            merge_folds=True,
            get_file=get_taylor_coefficients_file,
            key="taylor",
        )
    except FileNotFoundError as error:
        logger.warning(f"Skipping channel {channel}: {error}")
        return None
    data_frame = add_class_indices(data_frame, cfg_channel)
    parameter_names = [p.name for p in cfg.inputs.parameters]
    ranks = collect_variable_ranks(
        data_frame,
        [c.name for c in cfg.outputs.classes],
        None if cfg.taylor.include_parameters else parameter_names,
        absolute=cfg.taylor.absolute,
    )
    logger.info(
        f"Ranked {ranks['variable'].nunique()} variables in "
        f"{len(ranks.groupby(['m_x', 'm_y', 'output_class']))} TC sets of "
        f"channel {channel}"
    )
    return ranks


def get_description(cfg: Config, channels: list[str], missing: list[str]):
    fold_title, _ = get_fold_label(cfg, merge_folds=True)
    measure = "median of |TC|" if cfg.taylor.absolute else "|median of TC|"
    lines = [
        f"# Best Taylor-coefficient rank per input variable "
        f"({cfg.mode.decay_mode})",
        "",
        f"Generated on {date.today().isoformat()} by "
        "`python -m xyh_ml.cli.rank_taylor_variables`.",
        "",
        f"- Training tag: `{cfg.tags.training_tag}`; channels: "
        f"{', '.join(channels)}"
        + (f" (no TC files: {', '.join(missing)})" if missing else "")
        + f"; all eras, {fold_title}.",
        "- TC sets: channel × benchmark mass point × class c (TCs of the "
        "output node c for events of true and predicted class c).",
        f"- Within each set, all TCs are ranked by the {measure} "
        "(rank 1 = largest), not only the plotted ones.",
        "- A variable takes the rank of its first-order TC and of every "
        "second-order TC it is part of; the table shows its best rank over "
        "all sets and the set and TC where it is reached (ties: larger "
        "median).",
        "- `# sets`: number of sets reaching the best rank. `Best rank (1st "
        "order)`: best rank of the first-order TC alone. `Best rank "
        "(<channel>)`: best rank within that channel.",
        "",
    ]
    return "\n".join(lines)


@hydra.main(
    config_path=str(PROJECT_DIR / "config"),
    config_name="rank_taylor_variables",
    version_base="1.3",
)
def main(cfg: Config):
    channels, missing, tables = [], [], []
    for channel in cfg.taylor.ranking_channels:
        ranks = load_channel_ranks(cfg, channel)
        if ranks is None:
            missing.append(channel)
            continue
        channels.append(channel)
        tables.append(ranks)
    if not tables:
        raise FileNotFoundError(
            "No Taylor coefficient files found for the channels "
            f"{list(cfg.taylor.ranking_channels)}"
        )
    ranks = pd.concat(tables, ignore_index=True)
    if len(ranks) == 0:
        raise ValueError(
            f"No diagonal TC sets with events found in the channels {channels}"
        )

    variables = [v.name for v in cfg.inputs.variables]
    if cfg.taylor.include_parameters:
        variables += [p.name for p in cfg.inputs.parameters]
    best = get_best_variable_ranks(ranks, variables)

    # Write the tables
    _, fold_suffix = get_fold_label(cfg, merge_folds=True)
    output_dir = (
        PROJECT_DIR
        / "data"
        / "output"
        / cfg.tags.training_tag
        / "tables"
        / "taylor"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"variable_ranks__{cfg.mode.decay_mode}__{fold_suffix}"
    ranks.to_csv(output_dir / f"{stem}__all.csv", index=False)
    best.to_csv(output_dir / f"{stem}.csv", index=False)
    markdown_file = output_dir / f"{stem}.md"
    markdown_file.write_text(
        get_description(cfg, channels, missing)
        + "\n"
        + format_markdown_table(best, absolute=cfg.taylor.absolute)
    )
    logger.info(f"Wrote the variable ranking to {markdown_file}")


if __name__ == "__main__":
    main()
