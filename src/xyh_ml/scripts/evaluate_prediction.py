from pathlib import Path
import logging
import torch
import numpy as np
import pickle
from time import time
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from xyh_ml.scripts.train_dnn import (
    EVENT_PARITY,
    OUTPUT_DIR,
    INPUT_DIR,
    CATEGORIES,
    INPUT_VARIABLES,
    OUTPUT_VARIABLES,
    NETWORK_HYPERPARAMETERS,
    _prepare_output_dir,
    _prepare_input_files,
    _load_data_frame,
    _filter_event_parity,
    _prepare_dataset,
    _create_multi_layer_perceptron_model,
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
    logger.debug(
        f"Loaded transformations from {input_file}"
    )
    return transformations


def _load_checkpoints(input_file: Path):
    # Deserialize tracked checkpoints
    checkpoints = torch.load(input_file, map_location="cpu")
    logger.debug(
        f"Loaded {len(checkpoints)} checkpoints from {input_file}"
    )
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
        f"Evaluated model predictions for {len(pred)} events in {delta} s"
    )

    return pred


def _add_predictions_to_data_frame(
    data_frame: pd.DataFrame,
    predictions: np.ndarray,
):
    # Store score at each output node
    score_columns = [f"score_{i}" for i in range(predictions.shape[1])]
    data_frame.loc[:, score_columns] = predictions

    # Store output node index with highest score
    category_pred = np.argmax(predictions, axis=1)
    data_frame.loc[:, "category_pred"] = category_pred

    # Store output node index with highest score
    data_frame.loc[:, "category_pred_score"] =  np.reshape(
        np.take_along_axis(
            predictions,
            np.reshape(category_pred, (-1, 1)),
            axis=1,
        ),
        (-1, ),
    )

    return data_frame


def _plot_confusion_matrix(
    data_frame: pd.DataFrame,
    category_labels: list[str] | None = None,
):
    # Calculate confusion matrix
    cm = confusion_matrix(
        data_frame["category"].astype(np.int32).to_numpy(),
        data_frame["category_pred"].astype(np.int32).to_numpy(),
        normalize="true",
    )

    # Create the figure
    fig, ax = plt.subplots()

    # Plot the confusion matrix as heatmap
    sns.heatmap(cm, annot=True, ax=ax)

    # Set x and y labels
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("Truth")

    # Add category labels to x and y axis
    ax.set_xticks(
        np.arange(len(category_labels)) + 0.5,
        labels=category_labels,
    )
    ax.set_yticks(
        np.arange(len(category_labels)) + 0.5,
        labels=category_labels,
    )

    return fig, ax


def main():
    # Prepare the output directory
    _prepare_output_dir(OUTPUT_DIR)

    # Load transformations for variable preprocessing
    transformation_file = OUTPUT_DIR / "transformations.pkl"
    transformations = _load_transformations(transformation_file)

    # Load model checkpoints
    checkpoint_file = OUTPUT_DIR / "checkpoints.pt"
    checkpoints = _load_checkpoints(checkpoint_file)

    # Load the input files
    input_files = _prepare_input_files(INPUT_DIR, CATEGORIES)

    # Load the input files into a data frame
    data_frame = _load_data_frame(input_files, CATEGORIES)

    # Filter events according to the event parity
    data_frame = _filter_event_parity(data_frame, (EVENT_PARITY + 1) % 2)

    # Prepare evaluation dataset by applying transformations and creating torch
    # tensors from the data frame.
    dataset = _prepare_dataset(
        data_frame,
        INPUT_VARIABLES,
        OUTPUT_VARIABLES,
        {
            "name": "weight",
        },
        transformations,
    )

    # Create the neural network model
    model = _create_multi_layer_perceptron_model(
        len(INPUT_VARIABLES),
        len(CATEGORIES),
        NETWORK_HYPERPARAMETERS,
    )

    for epoch, checkpoint in enumerate(checkpoints):

        # Load the weights from last training epoch
        model.load_state_dict(checkpoints[epoch]["model_state_dict"])

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

        # Get the category labels
        ord_enc = transformations["ordinal_encoding"]["transformation"]
        category_order = ord_enc.categories_[
            ord_enc.feature_names_in_.tolist().index("category")
        ].tolist()
        category_lookup = {
            category["name"]: category
            for category in CATEGORIES
        }
        category_labels = [category_lookup[c]["label"] for c in category_order]

        # Plot the confusion matrix
        fig, ax = _plot_confusion_matrix(
            data_frame,
            category_labels=category_labels,
        )
        for ext in EXTENSIONS:
            fig.savefig(OUTPUT_DIR / f"confusion_matrix_eff_epoch_{epoch}.{ext}")

        plt.clf()


if __name__ == "__main__":
    main()

