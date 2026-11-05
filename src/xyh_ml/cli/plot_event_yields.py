"""
Plot the expected number of events predicted as signal as a function of the
mass point (m_X, m_Y), with one line for each true process class.
"""

import logging

import hydra

from xyh_ml import PROJECT_DIR
from xyh_ml.config import Config
from xyh_ml.plotting import (
    add_class_indices,
    get_fold_label,
    get_rlabel,
    load_predictions,
    plot_mass_grid,
    save_figure,
    set_plot_style,
    yields_per_true_class,
)

set_plot_style()

# Set up logging
logger = logging.getLogger(__name__)


@hydra.main(
    config_path=str(PROJECT_DIR / "config"),
    config_name="plot_event_yields",
    version_base="1.3",
)
def main(cfg: Config):
    # Load the predictions written by the categorization step for the
    # selected decay mode and channel, either of the held-out fold or merged
    # for all folds, and assign the true and predicted class to each event
    merge_folds = cfg.plotting.merge_folds
    data_frame = load_predictions(cfg, merge_folds=merge_folds)
    data_frame = add_class_indices(data_frame, cfg)
    fold_title, fold_suffix = get_fold_label(cfg, merge_folds=merge_folds)

    # Expected number of events of each true class in the signal categories
    for signal_class in cfg.outputs.classes:
        if not signal_class.is_signal:
            continue

        yields = yields_per_true_class(data_frame, cfg, signal_class.name)
        for (campaign, channel), df in yields.groupby(["campaign", "channel"]):
            fig, _ = plot_mass_grid(
                df,
                line_column="true_class",
                value_column="yield",
                output_classes=cfg.outputs.classes,
                ylabel="Events",
                rlabel=get_rlabel(
                    cfg.mode.decay_mode,
                    campaign,
                    channel,
                    f"predicted {signal_class.label}",
                    fold_title,
                ),
                legend_title="True process",
                log_y=True,
                ylim=None,
            )
            save_figure(
                fig,
                cfg,
                f"yields_predicted_{signal_class.name}"
                f"__{cfg.mode.decay_mode}__{campaign}"
                f"__{channel}__{fold_suffix}",
            )


if __name__ == "__main__":
    main()
