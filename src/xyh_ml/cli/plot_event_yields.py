"""
Plot the expected number of events in each predicted output class as a
function of the mass point (m_X, m_Y), separately for true background events
(all classes except the signal) and for true signal events.
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
    yields_per_predicted_class,
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

    # Expected number of true background and true signal events in each
    # predicted class
    yields_background, yields_signal = yields_per_predicted_class(
        data_frame, cfg
    )
    signal_label = " + ".join(
        c.label for c in cfg.outputs.classes if c.is_signal
    )

    for name, yields, ylabel, truth_label in [
        ("background", yields_background, "Background events", "true bkg."),
        ("signal", yields_signal, "Signal events", f"true {signal_label}"),
    ]:
        for (campaign, channel), df in yields.groupby(["campaign", "channel"]):
            fig, _ = plot_mass_grid(
                df,
                line_column="predicted_class",
                value_column="yield",
                output_classes=cfg.outputs.classes,
                ylabel=ylabel,
                rlabel=get_rlabel(
                    cfg.mode.decay_mode,
                    campaign,
                    channel,
                    truth_label,
                    fold_title,
                ),
                legend_title="Prediction",
                log_y=True,
                ylim=None,
            )
            save_figure(
                fig,
                cfg,
                f"yields_{name}__{cfg.mode.decay_mode}__{campaign}"
                f"__{channel}__{fold_suffix}",
            )


if __name__ == "__main__":
    main()
