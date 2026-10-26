import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch.nn import Module

from xyh_ml import PROJECT_DIR
from xyh_ml.cli.evaluate_classifier import (
    load_best_checkpoint,
    load_checkpoints,
    load_metrics,
    load_transformations,
    predict,
)
from xyh_ml.cli.train_classifier import (
    create_multi_layer_perceptron_model,
    create_torch_dataset,
    get_unique_input_parameter_combinations,
    load_data_frame,
)
from xyh_ml.config import (
    Config,
    Variable,
)

# Set up logging
logger = logging.getLogger(__name__)

# Seeds
TRAIN_TEST_SPLIT_SEED = 4712
PARAMETER_SAMPLING_SEED = 583


def filter_and_set_evaluation_parameters(
    data_frame: DataFrame,
    unique_parameter_values: DataFrame,
    n_samples_per_category: int | None = None,
):
    # Store reduction rate of the sampled events to be able to rescale it to
    # the original cross section. Without sampling, the reduction rate is 0.5
    # because of the even-odd parity split of the events.
    reduction_rates = {
        category: 0.5 for category in data_frame["class"].unique()
    }

    # Sample a few events from all categories to reduce evaluation time if
    # n_samples_per_category is set.
    if n_samples_per_category is not None:
        # Create a seeded random number generator so that the sampling is
        # reproducible for every parameter combination and independent of the
        # order in which the combinations are evaluated
        rng = np.random.default_rng(PARAMETER_SAMPLING_SEED)

        # Create the sample index for each category, select all events if less
        # than n_samples_per_category have been found
        sample_index = []
        for category in data_frame["class"].unique():
            # Get sampled index or all events if less than
            # n_samples_per_category have been found
            index = data_frame.loc[data_frame["class"] == category].index
            if len(index) <= n_samples_per_category:
                sample_index.append(index)
            else:
                i = rng.choice(len(index), size=(n_samples_per_category,))
                sample_index.append(index[i])

            # Calculate reduction rate of this sample due to sampling
            reduction_rates[category] *= len(sample_index[-1]) / len(index)

        # Concatenate sample index and select events from the data frame
        sample_index = np.concatenate(sample_index, axis=0)
        data_frame = data_frame.loc[sample_index]
        logger.debug(
            f"Sampled at maximum {n_samples_per_category} for each class"
        )

    # Add reduction rate to events
    logger.debug(f"Reduction rates: {reduction_rates}")
    data_frame["reduction_rate"] = data_frame["class"].map(reduction_rates)

    # Split into signal and background events as they require different
    # treatment for filling the parameter values.
    df_signal = data_frame.loc[data_frame["is_signal"]]
    df_bkg = data_frame.loc[~data_frame["is_signal"]]

    # For signal events, just keep events with parameter value combinations in
    # unique_parameter_values
    df_signal = df_signal[
        df_signal[list(unique_parameter_values.columns)]
        .isin(unique_parameter_values.to_dict("list"))
        .all(axis=1)
    ]

    # For background events, repeat the events and set each unique parameter
    # value combination once.
    df_bkg = pd.concat(
        [
            df_bkg.assign(
                **{p: v for p, v in zip(unique_parameter_values.columns, row)}
            )
            for row in unique_parameter_values.to_numpy()
        ],
        ignore_index=True,
    )

    # Concatenate signal and background events, shuffle, and return the data
    # frame
    data_frame = (
        pd.concat([df_signal, df_bkg], ignore_index=True)
        .sample(frac=1)
        .reset_index(drop=True)
    )

    return data_frame


def predict_for_parameters(
    data_frame: DataFrame,
    model: Module,
    input_variables: list[Variable],
    input_parameters: list[Variable],
    output_variable: Variable,
    weight: str,
    transformations: dict,
    unique_parameter_values: DataFrame,
    n_samples_per_category: int | None = None,
):
    # Filter signal events for values that the network is going to be evaluated
    # on and set these parameter values for all background events
    data_frame = filter_and_set_evaluation_parameters(
        data_frame,
        unique_parameter_values,
        n_samples_per_category=n_samples_per_category,
    )

    # Prepare evaluation dataset by applying transformations and creating torch
    # tensors from the data frame.
    dataset = create_torch_dataset(
        data_frame,
        input_variables,
        input_parameters,
        output_variable,
        weight,
        transformations,
    )

    # Evaluate output of the network for given dataset
    predictions, targets, weights = predict(
        model,
        dataset,
        "cuda" if torch.cuda.is_available() else "cpu",
    )

    return data_frame, predictions, targets, weights


@hydra.main(
    config_path=str(PROJECT_DIR / "config"),
    config_name="evaluate_categorization",
    version_base="1.3",
)
def main(cfg: Config):
    # Create output directory if it does not exist
    predictions_dir = Path(cfg.artifacts.predictions_dir)
    if not predictions_dir.exists():
        predictions_dir.mkdir(parents=True)
        logger.debug(f"Created output directory {predictions_dir}")

    # Load transformations from the training artifacts
    transformations = load_transformations(
        Path(cfg.artifacts.training_dir) / "transformations.pkl"
    )

    # Load metrics from the training artifacts
    metrics = load_metrics(Path(cfg.artifacts.training_dir) / "metrics.json")

    # Load checkpoints from the training artifacts
    checkpoints = load_checkpoints(
        Path(cfg.artifacts.training_dir) / "checkpoints.pt"
    )

    # Load the input files into a data frame from the fold that the network has
    # not been trained on
    data_frame = load_data_frame(
        Path(cfg.artifacts.data_frame_dir),
        cfg.mode.decay_mode,
        cfg.outputs.classes,
        (cfg.mode.event_parity + 1) % 2,
        cfg.mode.channel,
    )

    # Use the physics weight as evaluation weight
    eval_weight = "__total_weight"

    # Get unique parameter combinations
    unique_parameters = get_unique_input_parameter_combinations(
        data_frame,
        cfg.inputs.parameters,
    )

    # Create the neural network model
    model = create_multi_layer_perceptron_model(
        len(cfg.inputs.variables) + len(cfg.inputs.parameters),
        len(cfg.outputs.classes),
        cfg.network,
    )
    model = load_best_checkpoint(model, checkpoints, metrics)

    # Retrieve predictions for each unique parameter combination
    data_frame, predictions, targets, _ = predict_for_parameters(
        data_frame,
        model,
        cfg.inputs.variables,
        cfg.inputs.parameters,
        cfg.outputs.variables[0],
        eval_weight,
        transformations,
        unique_parameters,
        n_samples_per_category=10_000,
    )

    # Add predictions to the data frame
    for i in range(predictions.shape[1]):
        t = transformations["ordinal_encoding"]["transformation"]
        j = t.feature_names_in_.tolist().index(cfg.outputs.variables[0].name)
        c = t.categories_[j].tolist()
        class_name = c[i]
        data_frame[f"score_{class_name}"] = predictions[:, i]
    data_frame["target"] = targets

    # Write the data frame with predictions to a HDF5 file
    output_file = predictions_dir / f"predictions_{cfg.mode.decay_mode}.h5"
    data_frame.to_hdf(output_file, key="predictions", mode="w")


if __name__ == "__main__":
    main()
