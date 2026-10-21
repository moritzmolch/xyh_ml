import logging
import pickle
from pathlib import Path
from time import time

import hydra
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from xyh_ml import PROJECT_DIR
from xyh_ml.cli.train_classifier import (
    _create_multi_layer_perceptron_model,
    _filter_event_parity,
    _load_data_frame,
    _prepare_dataset,
    _prepare_output_dir,
)
from xyh_ml.config import (
    Config,
    OutputClass,
)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set seaborn plot style
sns.set_style("white")
sns.set_theme("talk")
sns.set_palette("colorblind")


EXTENSIONS = ["pdf", "png"]


def _load_transformations(input_file: Path):
    # Deserialize transformation objects
    with input_file.open(mode="rb") as f:
        transformations = pickle.load(f)
    logger.debug(f"Loaded transformations from {input_file}")
    return transformations


def _load_checkpoints(input_file: Path):
    # Deserialize tracked checkpoints
    checkpoints = torch.load(input_file, map_location="cpu")
    logger.debug(f"Loaded {len(checkpoints)} checkpoints from {input_file}")
    return checkpoints


def _predict(
    model: torch.nn.Module,
    dataset: torch.utils.data.TensorDataset,
    device: str,
    data_loader_num_workers: int = 1,
    data_loader_pin_memory: bool = False,
):
    # Create the data loader for predictions
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2**16,
        shuffle=False,
        num_workers=data_loader_num_workers,
        pin_memory=data_loader_pin_memory,
    )

    # Move the model to the training device and turn on evaluation mode
    model = model.eval().to(device)

    # Start time tracking
    start = time()

    pred = []
    for x, y, w in data_loader:
        # Move tensors to training device
        x, y, w = x.to(device), y.to(device), w.to(device)

        # Get the network prediction and append it to the list of prediction
        # tensors. Directly calculate normalized softmax scores
        with torch.no_grad():
            pred.append(torch.nn.functional.softmax(model(x), dim=-1))

    # Concatenate predictions and convert to numpy array
    pred = torch.cat(pred, dim=0).cpu().numpy()

    # Stop time tracking
    stop = time()

    # Log number of events and time spent for evaluation
    n_events = len(pred)
    delta = round(stop - start, 2)
    logger.debug(
        f"Evaluated model predictions for {n_events} events in {delta} s"
    )

    return pred


def _add_predictions_to_data_frame(
    data_frame: pd.DataFrame,
    predictions: np.ndarray,
    output_classes: list[OutputClass],
):
    # Get current columns in the data frame
    current_columns = data_frame.columns.tolist()

    # Store score at each output node
    output_class_names = [output_class.name for output_class in output_classes]
    score_columns = [
        f"score_{output_class_names[i]}" for i in range(predictions.shape[1])
    ]
    data_frame.loc[:, score_columns] = predictions

    # Store output node index with highest score
    category_pred = np.argmax(predictions, axis=1)
    data_frame.loc[:, "class_pred"] = category_pred

    # Store output node index with highest score
    data_frame.loc[:, "class_pred_score"] = np.reshape(
        np.take_along_axis(
            predictions,
            np.reshape(category_pred, (-1, 1)),
            axis=1,
        ),
        (-1,),
    )

    # Get new columns in the data frame
    new_columns = set(data_frame.columns.tolist()).difference(current_columns)

    return data_frame, new_columns


def _save_predictions(
    data_frame: pd.DataFrame,
    columns: list[str],
):
    # Store score at each output node
    score_columns = [f"score_{i}" for i in range(predictions.shape[1])]
    data_frame.loc[:, score_columns] = predictions

    # Store output node index with highest score
    category_pred = np.argmax(predictions, axis=1)
    data_frame.loc[:, "class_pred"] = category_pred

    # Store output node index with highest score
    data_frame.loc[:, "class_pred_score"] = np.reshape(
        np.take_along_axis(
            predictions,
            np.reshape(category_pred, (-1, 1)),
            axis=1,
        ),
        (-1,),
    )

    return data_frame


@hydra.main(
    config_path=str(PROJECT_DIR / "config"),
    config_name="evaluate_predictions",
    version_base="1.3",
)
def main(cfg: Config):
    # Get input and output directories from the configuration
    data_frame_dir = Path(cfg.artifacts.data_frame_dir)
    training_dir = Path(cfg.artifacts.training_dir)
    predictions_dir = Path(cfg.artifacts.predictions_dir)

    # Prepare the output directory
    _prepare_output_dir(predictions_dir)

    # Load transformations for variable preprocessing
    transformation_file = training_dir / "transformations.pkl"
    transformations = _load_transformations(transformation_file)

    # Load model checkpoints
    checkpoint_file = training_dir / "checkpoints.pt"
    checkpoints = _load_checkpoints(checkpoint_file)
    checkpoint = checkpoints[-1]

    # Load the input files into a data frame
    data_frame = _load_data_frame(
        data_frame_dir,
        cfg.mode.decay_mode,
        cfg.outputs.classes,
    )

    # Filter events according to the event parity
    data_frame = _filter_event_parity(
        data_frame,
        (cfg.mode.event_parity + 1) % 2,
    )

    # Prepare evaluation dataset by applying transformations and creating torch
    # tensors from the data frame.
    dataset = _prepare_dataset(
        data_frame,
        cfg.inputs.variables + cfg.inputs.parameters,
        cfg.outputs.variables,
        "__total_weight",
        transformations,
    )

    # Create the neural network model
    model = _create_multi_layer_perceptron_model(
        len(cfg.inputs.variables) + len(cfg.inputs.parameters),
        len(cfg.outputs.classes),
        cfg.network,
    )

    # Load the weights from last training epoch
    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate output of the network for given dataset
    predictions = _predict(
        model,
        dataset,
        "cuda" if torch.cuda.is_available() else "cpu",
    )

    # Concatenate predictions
    data_frame = _add_predictions_to_data_frame(
        data_frame,
        predictions,
    )


if __name__ == "__main__":
    main()
