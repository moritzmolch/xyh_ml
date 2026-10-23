import json
import logging
import pickle
from pathlib import Path
from time import time
from typing import Any

import hydra
import numpy as np
import torch

from xyh_ml import PROJECT_DIR
from xyh_ml.cli.train_classifier import (
    apply_transformations_to_unique_input_parameters,
    create_multi_layer_perceptron_model,
    create_torch_dataset,
    get_unique_input_parameter_combinations,
    load_data_frame,
)
from xyh_ml.config import (
    Config,
)

# Set up logging
logger = logging.getLogger(__name__)

# Seeds
TRAIN_TEST_SPLIT_SEED = 4712
PARAMETER_SAMPLING_SEED = 583


def load_transformations(input_file: Path):
    """
    Deserialize transformation objects from a pickle file.

    Parameters
    ----------
    input_file : Path
        Path to the pickle file containing the serialized transformations.

    Returns
    -------
    transformations : object
        The deserialized transformation objects. For more information, see the
        `xyh_ml.cli.train_classifier.derive_transformations` function.
    """
    # Deserialize transformation objects
    with input_file.open(mode="rb") as f:
        transformations = pickle.load(f)
    logger.debug(f"Loaded transformations from {input_file}")
    return transformations


def load_metrics(input_file: Path):
    """
    Load tracked metrics from a JSON file.

    Parameters
    ----------
    input_file : Path
        Path to the JSON file containing the tracked metrics.

    Returns
    -------
    metrics : dict
        A dictionary containing the tracked metrics. For more information,
        see the `xyh_ml.cli.train_classifier.train_and_validate` function.
    """

    # Deserialize tracked metrics from a JSON file
    with input_file.open(mode="r") as f:
        metrics = json.load(f)
    logger.debug(f"Loaded metrics from {input_file}")

    return metrics


def load_checkpoints(input_file: Path):
    """
    Load tracked checkpoints from a PyTorch file.

    Parameters
    ----------
    input_file : Path
        Path to the PyTorch file containing the tracked checkpoints.

    Returns
    -------
    checkpoints : dict
        A dictionary containing the tracked checkpoints. For more information,
        see the `xyh_ml.cli.train_classifier.train_and_validate` function.
    """

    # Deserialize tracked checkpoints
    checkpoints = torch.load(input_file, map_location="cpu")
    logger.debug(f"Loaded {len(checkpoints)} checkpoints from {input_file}")
    return checkpoints


def load_best_checkpoint(
    model: torch.nn.Module,
    checkpoints: list[dict[str, Any]],
    metrics: dict[str, Any],
):
    """
    Load best checkpoint into the model based on the lowest validation loss.

    Parameters
    ----------
    model : torch.nn.Module
        The model into which the best checkpoint will be loaded.

    checkpoints : list[dict[str, Any]]
        A list of dictionaries containing the tracked checkpoints. For more
        information, see the `xyh_ml.cli.train_classifier.train_and_validate`
        function.

    metrics : dict[str, Any]
        A dictionary containing the tracked metrics. For more information, see
        the `xyh_ml.cli.train_classifier.train_and_validate` function.

    Returns
    -------
    model : torch.nn.Module
        The model with the best checkpoint loaded based on the lowest
        validation loss.
    """

    # Get the epoch with the best validation loss
    best_epoch = metrics["val_epoch"][np.argmin(metrics["val_loss"])]

    # Load the model weights from the best epoch
    model.load_state_dict(checkpoints[best_epoch]["model_state_dict"])
    logger.debug(f"Loaded model weights from epoch {best_epoch}")

    return model


def predict(
    model: torch.nn.Module,
    dataset: torch.utils.data.TensorDataset,
    device: str,
    data_loader_num_workers: int = 1,
    data_loader_pin_memory: bool = False,
):
    # Create the data loader for predictions
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=len(dataset),
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


@hydra.main(
    config_path=str(PROJECT_DIR / "config"),
    config_name="train_classifier",
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
    )

    # Use the physics weight as evaluation weight
    eval_weight = "__total_weight"

    # Get unique parameter combinations
    unique_parameters = get_unique_input_parameter_combinations(
        data_frame,
        cfg.inputs.parameters,
    )

    # Prepare training dataset by applying transformations and creating torch
    # tensors from the data frame. Create separate datasets for training and
    # validation.
    dataset = create_torch_dataset(
        data_frame,
        cfg.inputs.variables,
        cfg.inputs.parameters,
        cfg.outputs.variables[0],
        eval_weight,
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
    model = load_best_checkpoint(model, checkpoints, metrics)

    # TODO continue


if __name__ == "__main__":
    main()
