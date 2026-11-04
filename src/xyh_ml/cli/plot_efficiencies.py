import logging

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullLocator

from xyh_ml import PROJECT_DIR
from xyh_ml.config import Config, OutputClass
from xyh_ml.plotting import (
    CAMPAIGN_LABELS,
    CHANNEL_LABELS,
    DECAY_MODE_LABELS,
    add_class_indices,
    add_cms_label,
    get_fold_label,
    get_style_color,
    load_predictions,
    set_plot_style,
)

set_plot_style()

# Set up logging
logger = logging.getLogger(__name__)


def _plot_efficiencies(
    efficiencies: pd.DataFrame,
    fold_title: str,
    decay_mode: str,
    campaign: str,
    channel: str,
    output_classes: list[OutputClass],
):
    # Split data frames for upper and lower plot panels
    eff_upper, eff_lower = (
        efficiencies.loc[(efficiencies["m_x"] <= 1500)],
        efficiencies.loc[(efficiencies["m_x"] > 1500)],
    )

    # Create the figure
    fig, ax = plt.subplots(nrows=2, figsize=(20, 12))

    # Color for m_x labels
    m_x_color = get_style_color(5)

    for j, df_part in enumerate([eff_upper, eff_lower]):
        # Define the x axis positions and set x axis labels to m_y masses
        df_part_x = df_part.loc[df_part["class"] == output_classes[0].name]
        x = np.arange(len(df_part_x)) + 0.5
        x_labels = [
            f"{int(df_part_x.iloc[i]['m_y'])}" for i in range(len(df_part_x))
        ]

        # Define the y axis positions for the different output classes list of
        # efficiencies for each output class
        y = [
            df_part.loc[df_part["class"] == output_class.name][
                "efficiency"
            ].to_numpy()
            for output_class in output_classes
        ]

        # Set output class labels
        labels = [output_class.label for output_class in output_classes]

        # Plot the efficiencies for each output class
        for i in range(len(y)):
            ax[j].plot(x, y[i], label=labels[i], linestyle="dashed")

        # Get edges of areas with different m_x and add vertical dashed lines.
        # Also add m_x labels in the middle of each area.
        edge_index = (
            [0]
            + [
                i + 1
                for i in range(len(df_part_x) - 1)
                if df_part_x.iloc[i]["m_x"] != df_part_x.iloc[i + 1]["m_x"]
            ]
            + [len(df_part_x) - 1]
        )
        for i, ei in enumerate(edge_index[:-1]):
            ax[j].axvline(ei, color="black", linestyle="dashed", linewidth=1)
            m_x = int(df_part_x.iloc[ei]["m_x"])
            ax[j].text(
                (ei + edge_index[i + 1]) / 2,
                0.2,
                rf"{m_x}",
                va="bottom",
                ha="center",
                color=m_x_color,
                transform=ax[j].transData,
            )

        # Add m_x label on the right-hand side of the plot
        ax[j].text(
            edge_index[-1] + 4,
            0.2,
            r"$m_{\text{X}}$ (GeV)",
            va="bottom",
            ha="left",
            color=m_x_color,
            transform=ax[j].transData,
        )

        # Set x labels to the m_y masses
        ax[j].set_xticks(
            x[::2], labels=x_labels[::2], rotation=90, fontsize=14
        )
        ax[j].xaxis.set_minor_locator(NullLocator())

        # Set axis limits and labels
        ax[j].set_xlim(np.min(x) - 0.5, np.max(x) + 0.5)
        ax[j].set_ylim(0.0, 1.05)
        ax[j].set_xlabel(r"$m_{\text{Y}}$ (GeV)")
        ax[j].set_ylabel(r"$\epsilon$(correct class)")

        # Add legend to the upper plot only
        if j == 0:
            ax[j].legend(
                title="Category",
                bbox_to_anchor=(1.01, 1.0),
                loc="upper left",
                frameon=True,
            )

    # Add the CMS label and information about decay mode, campaign, channel,
    # and evaluated fold(s) to the upper plot
    add_cms_label(
        ax[0],
        ",  ".join(
            [
                DECAY_MODE_LABELS[decay_mode],
                CAMPAIGN_LABELS[campaign],
                CHANNEL_LABELS[channel],
                fold_title,
            ]
        ),
    )

    # Tighten layout and adjust spacing to avoid overlapping elements
    fig.tight_layout()
    plt.subplots_adjust(right=0.85)

    return fig, ax


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

    # Convert records into a proper DataFrame for easier analysis and plotting
    efficiencies = pd.DataFrame.from_records(efficiencies)
    efficiencies = efficiencies.set_index(
        ["m_x", "m_y"],
        drop=False,
    ).sort_index()

    # Plot the efficiencies for each channel and campaign
    for (campaign, channel), df in efficiencies.groupby(
        ["campaign", "channel"]
    ):
        fig, ax = _plot_efficiencies(
            df,
            fold_title,
            cfg.mode.decay_mode,
            campaign,
            channel,
            cfg.outputs.classes,
        )
        for ext in ["pdf", "png"]:
            output_file = (
                PROJECT_DIR
                / "data"
                / "output"
                / cfg.tags.training_tag
                / "plots"
                / (
                    f"efficiency__{cfg.mode.decay_mode}__{campaign}"
                    f"__{channel}__{fold_suffix}.{ext}"
                )
            )
            if not output_file.parent.exists():
                output_file.parent.mkdir(parents=True)
            fig.savefig(output_file)
            logger.info(f"Saved efficiency plot to {output_file}")
        plt.close(fig)


if __name__ == "__main__":
    main()
