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
    config_name="plot_output_class_fractions",
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

    # Store the fraction of each true class among the events categorized
    # into each predicted class for each unique parameter combination
    fractions = []
    for (campaign, channel, m_x, m_y), df in data_frame.groupby(
        ["campaign", "channel", "m_x", "m_y"]
    ):
        for i, predicted_output_class in enumerate(cfg.outputs.classes):
            df_class = df[df["predicted_class_index"] == i]
            n_total = (
                df_class["__total_weight"] * 1 / df_class["reduction_rate"]
            ).sum()
            for j, true_output_class in enumerate(cfg.outputs.classes):
                df_class_process = df_class[df_class["true_class_index"] == j]
                n_process = (
                    df_class_process["__total_weight"]
                    * 1
                    / df_class_process["reduction_rate"]
                ).sum()
                fraction = n_process / n_total if n_total > 0 else 0.0
                fractions.append(
                    {
                        "campaign": campaign,
                        "channel": channel,
                        "m_x": m_x,
                        "m_y": m_y,
                        "predicted_class": predicted_output_class.name,
                        "true_class": true_output_class.name,
                        "fraction": fraction,
                    }
                )
    fractions = pd.DataFrame.from_records(fractions)

    # Plot the fractions for each channel, campaign and predicted class
    for predicted_class in cfg.outputs.classes:
        df_predicted = fractions[
            fractions["predicted_class"] == predicted_class.name
        ]
        for (campaign, channel), df in df_predicted.groupby(
            ["campaign", "channel"]
        ):
            fig, _ = plot_mass_grid(
                df,
                line_column="true_class",
                value_column="fraction",
                output_classes=cfg.outputs.classes,
                ylabel="Event fraction",
                rlabel=get_rlabel(
                    cfg.mode.decay_mode,
                    campaign,
                    channel,
                    f"pred. {predicted_class.label} class",
                    fold_title,
                ),
            )
            save_figure(
                fig,
                cfg,
                f"fractions_{predicted_class.name}__{cfg.mode.decay_mode}"
                f"__{campaign}__{channel}__{fold_suffix}",
            )


if __name__ == "__main__":
    main()
