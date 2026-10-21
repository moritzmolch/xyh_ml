import copy
import json
import logging
import pickle
from pathlib import Path
from time import time

import hydra
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from xyh_ml import PROJECT_DIR
from xyh_ml.cli.model import MultiLayerPerceptron
from xyh_ml.config import (
    Config,
    Network,
    OutputClass,
    Training,
    Variable,
)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Seeds
TRAIN_TEST_SPLIT_SEED = 4712


def _prepare_output_dir(
    output_dir: Path,
):
    # Create the output directory if it does not exist
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        logger.debug(f"Created output directory {output_dir}")


def load_data_frame(
    data_frame_dir: Path,
    decay_mode: str,
    output_classes: list[OutputClass],
    event_parity: int | None = None,
):
    """
    Load the data frame from the preprocessed HDF5 files.

    For signal events, the data frame is filtered for the given `decay_mode`.
    If `event_parity` is set to 0 or 1, the data frame is filtered for even
    or odd events. The full data frame is loaded otherwise.

    The data frame is shuffled before being returned.

    Parameters
    ----------
    data_frame_dir : Path
        Path to the directory containing the preprocessed HDF5 files.

    decay_mode : str
        Decay mode to filter the signal events for. Must be either `y2b_h2tau`
        or `y2tau_h2b`.

    output_classes : list[OutputClass]
        List of output classes to load the data frames for.

    event_parity : int | None, optional
        Event parity to filter the events for. Must be either 0, 1, or `None`.
        If `None`, no filtering is applied. Default is `None`.

    Returns
    -------
    pd.DataFrame
        Concatenated data frame containing the events for all output classes.

    Raises
    ------
    ValueError
        - If `decay_mode` is not `y2b_h2tau` or `y2tau_h2b`.
        - If `event_parity` is not 0, 1, or `None`.
    """

    # Raise exception if decay mode is not `y2b_h2tau` or `y2tau_h2b`
    if decay_mode not in ["y2b_h2tau", "y2tau_h2b"]:
        msg = f"Decay mode must be either 'y2b_h2tau' or 'y2tau_h2b', got {decay_mode}"
        logger.critical(msg)
        raise ValueError(msg)

    # Raise exception if parity is not 0, 1, or `None`
    if event_parity not in [0, 1, None]:
        msg = f"Event parity must be either 0, 1 or `None`, got {event_parity}"
        logger.critical(msg)
        raise ValueError(msg)

    # List of data frames which are going to be concatenated
    dfs: list[pd.DataFrame] = []

    # Track time for loading the data frames
    start = time()

    for output_class in output_classes:
        # Load the data frame from the preprocessed HDF5 file
        df = pd.read_hdf(
            data_frame_dir / f"{output_class.name}.h5",
            key="ntuple",
        )

        # If this is a signal output class, filter the data frame for the given
        # decay mode
        if output_class.is_signal:
            logger.debug(
                f"Select signal events for decay mode {decay_mode} in data "
                + f" frame for class {output_class.name}"
            )
            df = df.loc[df["decay_mode"] == decay_mode]

        # Filter for event parity if it has been set in the function call
        if event_parity is not None:
            logger.debug(
                f"Filter events for event parity {event_parity} in data frame "
                + f"for class {output_class.name}"
            )
            df = df.loc[df["event"] % 2 == event_parity]

        dfs.append(df)

        # Log information about the loaded data frame
        logger.info(f"Loaded data frame for output class {output_class.name}")

    # Concatenate the data frames and shuffle the events
    df = pd.concat(dfs, axis=0).sample(frac=1.0).reset_index(drop=True)

    # Stop time tracking
    stop = time()

    # Log information about data frame loader
    n_events = len(df)
    mem = round(df.memory_usage().sum() / (1024**2), 2)
    delta = round(stop - start, 2)
    logger.debug(
        f"Loaded data frame with {n_events} events of size {mem} MB in "
        + f"{delta} s"
    )

    return df


def _train_val_split(
    df: pd.DataFrame,
    train_frac: float,
):
    # Raise an exception if training fraction is not between 0 and 1
    if train_frac < 0 or train_frac > 1:
        msg = f"Training fraction must be between 0 and 1, got {train_frac}"
        logger.critical(msg)
        raise ValueError(msg)

    df_train, df_val = train_test_split(
        df,
        train_size=train_frac,
        shuffle=True,
        random_state=TRAIN_TEST_SPLIT_SEED,
    )

    return {
        "train": df_train,
        "val": df_val,
    }


def _reweight_by_group(
    df: pd.DataFrame,
    group: str | list[str],
    input_weight_name: str,
    output_weight_name: str,
    mask: np.ndarray | None = None,
):
    # Log the results
    logger.debug(
        f"Reweight column groups {group} from input weight "
        + f"{input_weight_name} to output weight {output_weight_name}"
    )

    # If the output columns doesn't exist yet, create them and set them to the
    # input weight column
    if output_weight_name not in df.columns:
        df[output_weight_name] = df[input_weight_name]
        logger.debug(
            f"Output weight column {output_weight_name} did not exist yet, "
            + "created it and set it to the input weight column"
        )

    # Create dummy mask that accepts all events if the mask has not been set
    # in the function call
    mask = mask if mask is not None else np.ones(df.shape[0], dtype=bool)

    # Create the grouper
    df_groups = df.loc[mask].groupby(by=group)

    # Calculate sum of weights for each group and get the average value over
    # all individual group values
    sum_of_weights = df_groups.agg({input_weight_name: "sum"})
    mean_sum_of_weights = sum_of_weights.mean()[0]
    logger.debug(
        f"Sum of weights per group before reweighting:\n{sum_of_weights.to_string()}"
    )
    logger.debug(f"Average sum of weights: \n{mean_sum_of_weights}")

    # Calculate the scale factor for each group, which scales the sum of weights
    # of all groups to the mean sum of weights
    scale_factor = mean_sum_of_weights / sum_of_weights
    logger.debug(f"Scale factors for reweighting:\n{scale_factor.to_string()}")

    # Apply the scale factor
    for group_values, df_group in df_groups:
        df.loc[df_group.index, output_weight_name] = (
            scale_factor.loc[group_values][0]
            * df.loc[df_group.index, input_weight_name]
        )

    # Calculate sum of weights after reweighting (for logging purposes)
    sum_of_weights = (
        df.loc[mask].groupby(by=group).agg({output_weight_name: "sum"})
    )
    logger.debug(
        f"Sum of weights per group after reweighting:\n{sum_of_weights.to_string()}"
    )

    return df


def _setup_transformations(
    df: pd.DataFrame,
    variables: list[Variable],
):

    # Set up the transformations with sklearn preprocessing objects
    transformations = {
        "standard_scaling": {
            "transformation": StandardScaler(),
        },
        "ordinal_encoding": {
            "transformation": OrdinalEncoder(),
        },
    }

    # Add information about transformed variables to the transformations
    # dictionary
    for transformation, transformation_dict in transformations.items():
        transformation_dict.update(
            {
                "variables": [
                    v.name
                    for v in variables
                    if v.transformation == transformation
                ]
            }
        )

    # Fit the transformations
    for transformation_dict in transformations.values():
        # Get the transformation object and the variables to transform
        t = transformation_dict["transformation"]
        v = transformation_dict["variables"]

        # Fit the transformation
        t.fit(df[v])
        logger.debug(
            f"Fitted transformation '{transformation}' for variables {v}"
        )

    return transformations


def _apply_transformations(
    df: pd.DataFrame,
    transformations: dict,
):
    for transformation, transformation_dict in transformations.items():
        # Get the transformation object and the variables to transform
        t = transformation_dict["transformation"]
        v = transformation_dict["variables"]

        # Apply transform to the columns
        df.loc[:, v] = t.transform(df[v])
        logger.debug(
            f"Applied transformation {transformation} for variables {v}"
        )

    return df


def _prepare_dataset(
    df: pd.DataFrame,
    input_variables: list[Variable],
    output_variables: list[Variable],
    weight: str,
    transformations: dict,
):
    # Get column names
    input_variable_names = [v.name for v in input_variables]
    output_variable_names = [v.name for v in output_variables]
    weight_variable_name = weight

    # Transform input and output variables for training dataset creation
    df_trans = _apply_transformations(df, transformations)

    # Create torch tensors from transformed data frame
    x = torch.from_numpy(
        df_trans.loc[:, input_variable_names].astype(np.float32).to_numpy()
    ).float()
    y = torch.from_numpy(
        df_trans.loc[:, output_variable_names].astype(np.int64).to_numpy()
    ).long()
    w = torch.from_numpy(
        df_trans.loc[:, weight_variable_name].astype(np.float32).to_numpy()
    ).float()

    dataset = torch.utils.data.TensorDataset(x, y, w)
    logger.debug("Created dataset with torch tensor objects from data frame")

    return dataset


def _create_multi_layer_perceptron_model(
    input_size: int,
    output_size: int,
    network: Network,
):
    # Create the model
    model = MultiLayerPerceptron(
        input_size,
        output_size,
        network.hidden_layers,
        network.activation,
        network.dropout_frac,
        activation_kwargs=network.activation_kwargs,
    )

    return model


def cross_entropy_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    # Get the single losses for each event
    elements = torch.nn.functional.cross_entropy(
        prediction,
        target,
        reduction="none",
    )

    # Calculate weighted sum
    loss = torch.sum(weight * elements) / torch.sum(weight)

    return loss


def _train_and_validate(
    model: torch.nn.Module,
    dataset_train: torch.utils.data.TensorDataset,
    dataset_val: torch.utils.data.TensorDataset,
    training: Training,
    device: str,
    n_steps_logging: int = 25,
):
    # Get training hyperparameters
    max_epochs = training.max_epochs
    batch_size = training.batch_size
    optimizer_learning_rate = training.optimizer_learning_rate
    data_loader_num_workers = training.data_loader_num_workers
    data_loader_pin_memory = training.data_loader_pin_memory

    # Create the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=optimizer_learning_rate,
    )

    # Create the data loaders for training and validation
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_loader_num_workers,
        pin_memory=data_loader_pin_memory,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=len(dataset_val),
        shuffle=False,
        num_workers=data_loader_num_workers,
        pin_memory=data_loader_pin_memory,
    )

    # Move the model to the training device
    model = model.to(device)

    # Metrics container
    metrics = {}

    # Checkpoints container
    checkpoints = []

    # Global step counter
    step = 0

    for epoch in range(max_epochs):
        # Log training epoch
        logger.debug(f"Start training epoch {epoch}")

        # Use model in training mode
        model = model.train()

        for batch, (x, y, w) in enumerate(data_loader_train):
            # Clear gradients from previous iteration
            optimizer.zero_grad()

            # Move tensors to training device
            x, y, w = x.to(device), y.to(device), w.to(device)

            # Get the network prediction and calculate the loss
            pred = model(x)
            loss = cross_entropy_loss(pred, y.view(-1), w)

            # Perform backpropagation
            loss.backward()
            optimizer.step()

            # Track training metrics
            metrics.setdefault("train_step", []).append(step)
            metrics.setdefault("train_epoch", []).append(epoch)
            metrics.setdefault("train_batch", []).append(batch)
            metrics.setdefault("train_count", []).append(w.size(0))
            metrics.setdefault("train_weight_sum", []).append(w.sum().item())
            metrics.setdefault("train_loss", []).append(loss.item())

            # Log results
            if batch % n_steps_logging == 0:
                logger.debug(
                    "Training metrics "
                    + "  /  ".join(
                        [
                            f"{name}: {metric_list[-1]}"
                            for name, metric_list in metrics.items()
                            if name.startswith("train")
                        ]
                    )
                )

            # Increment step counter
            step += 1

        # Use model in evaluation mode
        model = model.eval()

        for batch, (x, y, w) in enumerate(data_loader_val):
            # Move tensors to training device
            x, y, w = x.to(device), y.to(device), w.to(device)

            # Get the network prediction and calculate the loss
            with torch.no_grad():
                pred = model(x)
                loss = cross_entropy_loss(pred, y.view(-1), w)

            # Track training metrics
            metrics.setdefault("val_step", []).append(step)
            metrics.setdefault("val_epoch", []).append(epoch)
            metrics.setdefault("val_count", []).append(w.size(0))
            metrics.setdefault("val_weight_sum", []).append(w.sum().item())
            metrics.setdefault("val_loss", []).append(loss.item())

            # Log results
            logger.debug(
                "Validation metrics "
                + "  /  ".join(
                    [
                        f"{name}: {metric_list[-1]}"
                        for name, metric_list in metrics.items()
                        if name.startswith("val")
                    ]
                )
            )

        # Add model checkpoint after epoch
        checkpoints.append(
            {
                "epoch": epoch,
                "step": step,
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
            }
        )

    return {
        "checkpoints": checkpoints,
        "metrics": metrics,
    }


def _dump_transformations(output_file: Path, transformations: dict):
    # Serialize transformation objects in a pickle file
    with output_file.open(mode="wb") as f:
        pickle.dump(transformations, f)
    logger.debug(
        f"Serialized transformations to {output_file} in pickle format"
    )


def _dump_metrics(output_file: Path, metrics: dict):
    # Serialize tracked metrics in a JSON file
    with output_file.open(mode="w") as f:
        json.dump(metrics, f)
    logger.debug(f"Serialized metrics to {output_file} in JSON format")


def _dump_checkpoints(output_file: Path, checkpoints: list):
    # Serialize tracked checkpoints with the torch.save function
    torch.save(checkpoints, output_file)
    logger.debug(
        f"Serialized model checkpoints to {output_file} in torch format"
    )


@hydra.main(
    config_path=str(PROJECT_DIR / "config"),
    config_name="train_classifier",
    version_base="1.3",
)
def main(cfg: Config):

    # Prepare the output directory
    _prepare_output_dir(Path(cfg.artifacts.training_dir))

    # Load the input files into a data frame
    data_frame = _load_data_frame(
        Path(cfg.artifacts.data_frame_dir),
        cfg.mode.decay_mode,
        cfg.outputs.classes,
    )

    # Filter events according to the event parity
    data_frame = _filter_event_parity(data_frame, cfg.mode.event_parity)

    # Turn negative weights into positive weights
    data_frame.loc[:, "__total_weight"] = np.abs(
        data_frame.loc[:, "__total_weight"]
    )

    # # Split into a training and validation data frame
    data_frames = _train_val_split(
        data_frame,
        train_frac=cfg.dataset.train_frac,
    )

    # Reweight events so that all signal hypotheses have the same sum of weights
    for split in data_frames:
        data_frames[split] = _reweight_by_group(
            data_frames[split],
            ["m_x", "m_y"],
            "__total_weight",
            "train_weight",
            mask=data_frames[split]["is_signal"],
        )

    # Reweight events so that all categories have the same sum of weights
    for split in data_frames:
        data_frames[split] = _reweight_by_group(
            data_frames[split],
            "class",
            "train_weight",
            "train_weight",
        )

    # Fit transformations to preprocess input and output variables on the
    # training dataset
    transformations = _setup_transformations(
        data_frames["train"],
        cfg.inputs.variables + cfg.inputs.parameters + cfg.outputs.variables,
    )

    # Prepare training dataset by applying transformations and creating torch
    # tensors from the data frame. Create separate datasets for training and
    # validation.
    datasets = {}
    for split, data_frame in data_frames.items():
        datasets[split] = _prepare_dataset(
            data_frame,
            cfg.inputs.variables + cfg.inputs.parameters,
            cfg.outputs.variables,
            "train_weight",
            transformations,
        )

    # Create the neural network model
    model = _create_multi_layer_perceptron_model(
        len(cfg.inputs.variables) + len(cfg.inputs.parameters),
        len(cfg.outputs.classes),
        cfg.network,
    )

    # Run the training
    training_results = _train_and_validate(
        model,
        datasets["train"],
        datasets["val"],
        cfg.training,
        "cuda" if torch.cuda.is_available() else "cpu",
    )

    # Serialize transformations, checkpoints, and metrics
    output_dir = Path(cfg.artifacts.training_dir)
    _dump_transformations(output_dir / "transformations.pkl", transformations)
    _dump_metrics(output_dir / "metrics.json", training_results["metrics"])
    _dump_checkpoints(
        output_dir / "checkpoints.pt", training_results["checkpoints"]
    )


if __name__ == "__main__":
    main()
