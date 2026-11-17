import logging

import hydra

from xyh_ml import PROJECT_DIR
from xyh_ml.config import Config
from xyh_ml.plotting import (
    CHANNEL_LABELS,
    DECAY_MODE_LABELS,
    add_class_indices,
    get_fold_label,
    load_predictions,
    save_figure,
    set_plot_style,
)
from xyh_ml.taylor import (
    get_taylor_coefficients_file,
    iter_diagonal_sets,
)
from xyh_ml.taylor_plotting import (
    paginate,
    plot_coefficients,
    summarize_coefficients,
)

set_plot_style()

# Set up logging
logger = logging.getLogger(__name__)


@hydra.main(
    config_path=str(PROJECT_DIR / "config"),
    config_name="plot_taylor_coefficients",
    version_base="1.3",
)
def main(cfg: Config):
    # Load the coefficients of all folds (each event evaluated by the model
    # not trained on it) and pool all eras: one plot set per channel and mass
    # point, not per fold or era
    data_frame = load_predictions(
        cfg,
        merge_folds=True,
        get_file=get_taylor_coefficients_file,
        key="taylor",
    )
    fold_title, fold_suffix = get_fold_label(cfg, merge_folds=True)
    decay_mode = cfg.mode.decay_mode
    campaigns = sorted(data_frame["campaign"].unique())
    logger.info(f"Merging eras {campaigns} and {fold_title}")

    # Assign the true class and the predicted class (highest score) to each
    # event. Only the "diagonal" is plotted: the TCs of the output node of
    # class c for events of true class c that are also predicted as class c.
    data_frame = add_class_indices(data_frame, cfg)
    parameter_names = [p.name for p in cfg.inputs.parameters]
    output_classes = {c.name: c for c in cfg.outputs.classes}

    # Drop the TCs of the input parameters (also from files written with
    # taylor.include_parameters=true)
    for channel, m_x, m_y, class_name, df, columns in iter_diagonal_sets(
        data_frame,
        list(output_classes),
        None if cfg.taylor.include_parameters else parameter_names,
    ):
        output_class = output_classes[class_name]
        point = f"{channel}, m_x={m_x:g}, m_y={m_y:g}"
        summary = summarize_coefficients(
            df, columns, absolute=cfg.taylor.absolute
        )
        if cfg.taylor.max_coefficients is not None:
            summary = summary.head(cfg.taylor.max_coefficients)
        pages = paginate(summary, cfg.taylor.per_plot)
        logger.info(
            f"Plotting {len(summary)} coefficients of class "
            f"{output_class.name} ({len(df)} events, {point}) on "
            f"{len(pages)} page(s)"
        )
        for page, page_summary in enumerate(pages, start=1):
            fig, _ = plot_coefficients(
                page_summary,
                show_intervals=cfg.taylor.show_intervals,
                absolute=cfg.taylor.absolute,
                rlabel=",  ".join(
                    [
                        DECAY_MODE_LABELS[decay_mode],
                        CHANNEL_LABELS[channel],
                    ]
                ),
                title=(
                    rf"true & predicted: {output_class.label}"
                    "\n"
                    rf"$m_\mathrm{{X}}$={m_x:g}, "
                    rf"$m_\mathrm{{Y}}$={m_y:g} GeV, all eras, "
                    f"{fold_title}"
                ),
            )
            save_figure(
                fig,
                cfg,
                f"taylor__{output_class.name}"
                f"__{decay_mode}__{channel}"
                f"__mx{int(m_x)}_my{int(m_y)}"
                f"__alleras_{fold_suffix}"
                f"__page{page}",
                subdir="taylor",
            )


if __name__ == "__main__":
    main()
