import logging
from pathlib import Path
from time import time

import hydra
import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch.nn import Module

from xyh_ml import PROJECT_DIR
from xyh_ml.config import (
    Config,
    Variable,
)
from xyh_ml.constants import PARAMETER_SAMPLING_SEED
from xyh_ml.data.data_frame import (
    get_unique_input_parameter_combinations,
    load_data_frame,
)
from xyh_ml.data.torch import create_torch_dataset
from xyh_ml.io import load_transformations
from xyh_ml.nn.mlp import create_multi_layer_perceptron_model

# Set up logging
logger = logging.getLogger(__name__)


def predict(
    model: torch.nn.Module,
    dataset: torch.utils.data.TensorDataset,
    device: str,
    data_loader_num_workers: int = 1,
    data_loader_pin_memory: bool = False,
    batch_size: int = 65536,
):
    # Create the data loader for predictions
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_loader_num_workers,
        pin_memory=data_loader_pin_memory,
    )

    # Move the model to the training device and turn on evaluation mode
    model = model.eval().to(device)

    # Start time tracking
    start = time()

    pred = []
    targets = []
    weights = []
    for x, p, y, w, _ in data_loader:
        # Move tensors to training device
        x = x.to(device)
        p = p.to(device)

        # Get the network prediction and append it to the list of prediction
        # tensors. Directly calculate normalized softmax scores
        with torch.no_grad():
            pred.append(
                torch.nn.functional.softmax(
                    model(torch.cat((x, p), dim=-1)),
                    dim=-1,
                )
            )

        # Collect output targets and event weights
        targets.append(y)
        weights.append(w)

    # Concatenate predictions, targets, and weights and convert to numpy arrays
    pred = torch.cat(pred, dim=0).cpu().numpy()
    targets = torch.cat(targets, dim=0).cpu().numpy()
    weights = torch.cat(weights, dim=0).cpu().numpy()

    # Stop time tracking
    stop = time()

    # Log number of events and time spent for evaluation
    n_events = len(pred)
    delta = round(stop - start, 2)
    logger.debug(
        f"Evaluated model predictions for {n_events} events in {delta} s"
    )

    return pred, targets, weights


def filter_and_set_evaluation_parameters(
    data_frame: DataFrame,
    unique_parameter_values: DataFrame,
    n_folds: int,
    n_samples_per_category: int | None = None,
):
    # Store reduction rate of the sampled events to be able to rescale it to
    # the original cross section. Without sampling, the reduction rate is
    # 1 / n_folds because only the held-out fold is evaluated.
    reduction_rates = {
        (output_class, m_x, m_y): 1 / n_folds
        for output_class, m_x, m_y in data_frame[["class", "m_x", "m_y"]]
        .drop_duplicates()
        .values
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

        for (output_class, m_x, m_y), df in data_frame.groupby(
            ["class", "m_x", "m_y"]
        ):
            assert (
                output_class != "xyh" and m_x == -10 and m_y == -10
            ) or output_class == "xyh", (
                f"Found unexpected parameter values for class {output_class}: "
                f"m_x={m_x}, m_y={m_y}"
            )

            # Get sampled index or all events if less than
            # n_samples_per_category have been found
            index = df.index
            if len(index) <= n_samples_per_category:
                sample_index.append(index)
            else:
                i = rng.choice(len(index), size=(n_samples_per_category,))
                sample_index.append(index[i])

            # Calculate the reduction rate
            reduction_rates[(output_class, m_x, m_y)] *= len(
                sample_index[-1]
            ) / len(index)

        # Concatenate sample index and select events from the data frame
        sample_index = np.concatenate(sample_index, axis=0)
        data_frame = data_frame.loc[sample_index]
        logger.debug(
            f"Sampled at maximum {n_samples_per_category} for each class"
        )

    # Add reduction rate to events
    logger.debug(f"Reduction rates: {reduction_rates}")
    data_frame["reduction_rate"] = data_frame[["class", "m_x", "m_y"]].apply(
        lambda x: reduction_rates[(x["class"], x["m_x"], x["m_y"])], axis=1
    )

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
    n_folds: int,
    batch_size=65536,
    n_samples_per_category: int | None = None,
):
    # Filter signal events for values that the network is going to be evaluated
    # on and set these parameter values for all background events
    data_frame = filter_and_set_evaluation_parameters(
        data_frame,
        unique_parameter_values,
        n_folds,
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
        batch_size=batch_size,
    )

    return data_frame, predictions, targets, weights


@hydra.main(
    config_path=str(PROJECT_DIR / "config"),
    config_name="evaluate_test_sample",
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

    # Create the neural network model
    model = create_multi_layer_perceptron_model(
        len(cfg.inputs.variables) + len(cfg.inputs.parameters),
        len(cfg.outputs.classes),
        cfg.network,
    )

    # Load checkpoints from the training artifacts
    best_model_ckpt_file = Path(cfg.artifacts.training_dir) / "best_model.ckpt"
    best_model_state_dict = torch.load(
        best_model_ckpt_file, map_location="cpu", weights_only=True
    )["state_dict"]

    # Rename keys in the state dict to match the bare model's keys if they are
    # prefixed with "model."
    for key in list(best_model_state_dict.keys()):
        new_key = key
        if key.startswith("model."):
            new_key = key.replace("model.", "", 1)
        best_model_state_dict[new_key] = best_model_state_dict.pop(key)
    model.load_state_dict(best_model_state_dict)

    # Load the input files into a data frame from the fold that the network has
    # not been trained on
    data_frame = load_data_frame(
        Path(cfg.artifacts.data_frame_dir),
        cfg.mode.decay_mode,
        cfg.outputs.classes,
        cfg.mode.n_folds,
        cfg.mode.fold,
        "eval",
        cfg.mode.channel,
    )

    # Get unique parameter combinations
    unique_parameters = get_unique_input_parameter_combinations(
        data_frame,
        cfg.inputs.parameters,
    )

    # Use the physics weight as evaluation weight
    eval_weight = "__total_weight"

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
        cfg.mode.n_folds,
        batch_size=cfg.training.batch_size,
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
