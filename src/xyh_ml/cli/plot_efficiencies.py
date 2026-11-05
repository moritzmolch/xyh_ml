import logging

import hydra
import pandas as pd

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
)

set_plot_style()

# Set up logging
logger = logging.getLogger(__name__)


@hydra.main(
    config_path=str(PROJECT_DIR / "config"),
    config_name="plot_efficiencies",
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

    # Store efficiency of categorizing events into the correct class for each
    # unique parameter combination
    efficiencies = []
    for (campaign, channel, m_x, m_y), df in data_frame.groupby(
        ["campaign", "channel", "m_x", "m_y"]
    ):
        # Get efficiency of categorizing events into the correct class for each
        # truth class
        for i, output_class in enumerate(cfg.outputs.classes):
            df_class = df[df["true_class_index"] == i]
            correct = df_class["predicted_class_index"] == i
            efficiency = correct.sum() / df_class.shape[0]
            efficiencies.append(
                {
                    "campaign": campaign,
                    "channel": channel,
                    "m_x": m_x,
                    "m_y": m_y,
                    "class": output_class.name,
                    "efficiency": efficiency,
                }
            )
    efficiencies = pd.DataFrame.from_records(efficiencies)

    # Plot the efficiencies for each channel and campaign
    for (campaign, channel), df in efficiencies.groupby(
        ["campaign", "channel"]
    ):
        fig, _ = plot_mass_grid(
            df,
            line_column="class",
            value_column="efficiency",
            output_classes=cfg.outputs.classes,
            ylabel=r"$\epsilon$(correct class)",
            rlabel=get_rlabel(
                cfg.mode.decay_mode, campaign, channel, fold_title
            ),
        )
        save_figure(
            fig,
            cfg,
            f"efficiency__{cfg.mode.decay_mode}__{campaign}__{channel}"
            f"__{fold_suffix}",
        )


if __name__ == "__main__":
    main()
