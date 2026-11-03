import logging
from pathlib import Path

import hydra
import numpy as np
import torch

from xyh_ml import PROJECT_DIR
from xyh_ml.config import (
    Config,
)
from xyh_ml.data.data_frame import (
    get_unique_input_parameter_combinations,
    load_data_frame,
    train_val_split,
)
from xyh_ml.data.reweighting import reweight_by_group
from xyh_ml.data.torch import create_torch_dataset
from xyh_ml.data.transformations import (
    apply_transformations_to_unique_input_parameters,
    derive_transformations,
)
from xyh_ml.io import (
    dump_config,
    dump_transformations,
)
from xyh_ml.nn.lightning import train_and_validate
from xyh_ml.nn.mlp import create_multi_layer_perceptron_model

# Set up logging
logger = logging.getLogger(__name__)


def prepare_output_dir(
    output_dir: Path,
):
    # Create the output directory if it does not exist
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        logger.debug(f"Created output directory {output_dir}")


@hydra.main(
    config_path=str(PROJECT_DIR / "config"),
    config_name="train",
    version_base="1.3",
)
def main(cfg: Config):
    # Create output directory if it does not exist
    training_dir = Path(cfg.artifacts.training_dir)
    if not training_dir.exists():
        training_dir.mkdir(parents=True)
        logger.debug(f"Created output directory {training_dir}")

    # Load the input files into a data frame
    data_frame = load_data_frame(
        Path(cfg.artifacts.data_frame_dir),
        cfg.mode.decay_mode,
        cfg.outputs.classes,
        cfg.mode.n_folds,
        cfg.mode.fold,
        "train",
        cfg.mode.channel,
    )

    # Get unique parameter combinations
    unique_parameters = get_unique_input_parameter_combinations(
        data_frame,
        cfg.inputs.parameters,
    )

    # Define training weight column and turn negative weights into positive
    # weights
    train_weight = "train_weight"
    data_frame.loc[:, train_weight] = np.abs(
        data_frame.loc[:, "__total_weight"]
    )

    # Split into a training and validation data frame
    data_frames = train_val_split(
        data_frame,
        train_frac=cfg.dataset.train_frac,
    )

    # Reweight events so that all campaigns have the same sum of weights
    for split in data_frames:
        data_frames[split] = reweight_by_group(
            data_frames[split],
            "campaign",
            train_weight,
        )

    # Reweight events so that all signal hypotheses have the same sum of
    # weights
    for split in data_frames:
        data_frames[split] = reweight_by_group(
            data_frames[split],
            ["m_x", "m_y"],
            train_weight,
            mask=data_frames[split]["is_signal"],
        )

    # Reweight events so that all categories have the same sum of weights
    for split in data_frames:
        data_frames[split] = reweight_by_group(
            data_frames[split],
            "class",
            train_weight,
        )

        # # Reweight events so that signal : background ratio is 1 : 1
        # for split in data_frames:
        #     data_frames[split] = reweight_by_group(
        #         data_frames[split],
        #         "is_signal",
        #         train_weight,
        #     )

    # Fit transformations to preprocess input and output variables on the
    # training dataset
    transformations = derive_transformations(
        data_frames["train"],
        cfg.inputs.variables + cfg.inputs.parameters + cfg.outputs.variables,
    )

    # Prepare training dataset by applying transformations and creating torch
    # tensors from the data frame. Create separate datasets for training and
    # validation.
    datasets = {}
    for split, df_split in data_frames.items():
        datasets[split] = create_torch_dataset(
            df_split,
            cfg.inputs.variables,
            cfg.inputs.parameters,
            cfg.outputs.variables[0],
            train_weight,
            transformations,
        )

    # Apply transformations to the unique parameter combinations and create a
    # torch tensor from the data frame
    unique_parameters_transformed = torch.from_numpy(
        (
            apply_transformations_to_unique_input_parameters(
                unique_parameters, transformations
            )
            .astype(np.float32)
            .to_numpy()
        )
    )

    # Create the neural network model
    model = create_multi_layer_perceptron_model(
        len(cfg.inputs.variables) + len(cfg.inputs.parameters),
        len(cfg.outputs.classes),
        cfg.network,
    )

    # Store the configuration and the transformations
    dump_config(training_dir / "config.json", cfg)
    dump_transformations(training_dir / "transformations.pkl", transformations)

    # Run the training and store the best model and training logs
    train_and_validate(
        model,
        datasets["train"],
        datasets["val"],
        unique_parameters_transformed,
        cfg.training,
        "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_output_file=training_dir / "best_model.ckpt",
        metrics_output_file=training_dir / "metrics.csv",
    )


if __name__ == "__main__":
    main()
