import copy
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import TensorDataset

from xyh_ml import PROJECT_DIR
from xyh_ml.config import (
    Config,
    Training,
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
from xyh_ml.nn.losses import get_loss_fn
from xyh_ml.nn.mlp import create_multi_layer_perceptron_model
from xyh_ml.nn.parameters import resample_input_parameters

# Set up logging
logger = logging.getLogger(__name__)

# Seeds
PARAMETER_SAMPLING_SEED = 583


def prepare_output_dir(
    output_dir: Path,
):
    # Create the output directory if it does not exist
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        logger.debug(f"Created output directory {output_dir}")


def train_and_validate(
    model: Module,
    dataset_train: TensorDataset,
    dataset_val: TensorDataset,
    unique_parameters: Tensor,
    training: Training,
    device: str,
    n_steps_logging: int = 25,
) -> dict[str, Any]:
    """
    Training and validation loop for the model.

    The function returns a dictionary with the training results. The dictionary
    contains the following keys:

    - `"checkpoints"`: List of model checkpoints after each epoch. Each
        checkpoint is a dictionary containing the following keys:
        - `"epoch"`: Epoch number.
        - `"step"`: Global step number.
        - `"model_state_dict"`: State dictionary of the model.
        - `"optimizer_state_dict"`: State dictionary of the optimizer.

    - `"metrics"`: Dictionary containing the tracked metrics during training
      and validation. The keys are the metric names and the values are lists of
      metric values for each step.

    Parameters
    ----------

    model : Module
        PyTorch model to train and validate.

    dataset_train : TensorDataset
        Training dataset. For more information, see the `create_torch_dataset`
        function.

    dataset_val : TensorDataset
        Validation dataset. For more information, see the
        `create_torch_dataset` function.

    unique_parameters : Tensor
        Tensor containing unique combinations of input parameter values. Shape
        is (num_unique_combinations, num_parameters).

    training : Training
        Training configuration containing hyperparameter settings for the
        training loop.

    device : str
        Device to use for training and validation. Must be either `"cpu"` or
        `"cuda"`.

    n_steps_logging : int, optional
        Number of training steps between logging of training metrics. Default
        is 25.

    Returns
    -------

    dict[str, Any]
        Dictionary containing the training results. For more information, see
        the function description.
    """

    # Get training hyperparameters
    max_epochs = training.max_epochs
    batch_size = training.batch_size
    optimizer_learning_rate = training.optimizer_learning_rate
    optimizer_weight_decay = training.optimizer_weight_decay
    data_loader_num_workers = training.data_loader_num_workers
    data_loader_pin_memory = training.data_loader_pin_memory

    # Get the loss function
    loss_fn = get_loss_fn(training.loss_fn, training.loss_fn_kwargs)

    # Create the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=optimizer_learning_rate,
        weight_decay=optimizer_weight_decay,
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
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_loader_num_workers,
        pin_memory=data_loader_pin_memory,
    )

    # Move the model to the training device
    model = model.to(device)

    # Create the generator for parameter sampling
    sampling_rng = torch.Generator(device=device)
    sampling_rng.manual_seed(PARAMETER_SAMPLING_SEED)

    # Move parameters to the training device
    unique_parameters = unique_parameters.to(device)

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

        for batch, (x, p, y, w, r) in enumerate(data_loader_train):
            # Clear gradients from previous iteration
            optimizer.zero_grad()

            # Move tensors to training device
            x = x.to(device)
            p = p.to(device)
            y = y.to(device)
            w = w.to(device)
            r = r.to(device)

            # Resample parameters for events that are marked for resampling
            p = resample_input_parameters(
                p, r, unique_parameters, rng=sampling_rng
            )

            # Get the network prediction and calculate the loss
            pred = model(torch.cat((x, p), dim=-1))
            loss = loss_fn(pred, y.view(-1), w)

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

        for batch, (x, p, y, w, r) in enumerate(data_loader_val):
            # Move tensors to training device
            x = x.to(device)
            p = p.to(device)
            y = y.to(device)
            w = w.to(device)
            r = r.to(device)

            # Resample parameters for events that are marked for resampling
            p = resample_input_parameters(
                p, r, unique_parameters, rng=sampling_rng
            )

            # Get the network prediction and calculate the loss
            with torch.no_grad():
                pred = model(torch.cat((x, p), dim=-1))
                loss = loss_fn(pred, y.view(-1), w)

            # Track validation metrics
            metrics.setdefault("val_step", []).append(step)
            metrics.setdefault("val_epoch", []).append(epoch)
            metrics.setdefault("val_count", []).append(w.size(0))
            metrics.setdefault("val_weight_sum", []).append(w.sum().item())
            metrics.setdefault("val_loss", []).append(loss.item())

            # Log results
            if batch % n_steps_logging == 0:
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


def dump_config(
    output_file: Path,
    config: Config,
):
    """
    Serialize training configuration to a JSON file.

    Parameters
    ----------
    output_file : Path
        Path to the output JSON file.

    config : Config
        Training configuration to serialize. Must be an instance of the
        `Config` dataclass.
    """

    # Serialize training configuration in a JSON file
    with output_file.open(mode="w") as f:
        json.dump(config, f, default=str, indent=4)
    logger.debug(f"Serialized configuration to {output_file} in JSON format")


def dump_transformations(
    output_file: Path,
    transformations: dict[str, dict[str, Any]],
):
    """
    Serialize fitted transformations to a pickle file.

    Parameters
    ----------
    output_file : Path
        Path to the output pickle file.

    transformations : dict
        Dictionary containing the fitted transformation objects and the
        corresponding variable names. For more information, see the
        `derive_transformations` function.
    """

    # Serialize transformation objects in a pickle file
    with output_file.open(mode="wb") as f:
        pickle.dump(transformations, f)
    logger.debug(
        f"Serialized transformations to {output_file} in pickle format"
    )


def dump_metrics(
    output_file: Path,
    metrics: list[dict[str, Any]],
):
    """
    Serialize tracked metrics to a JSON file.

    Parameters
    ----------
    output_file : Path
        Path to the output JSON file.

    metrics : dict
        Dictionary containing the tracked metrics.
    """

    # Serialize tracked metrics in a JSON file
    with output_file.open(mode="w") as f:
        json.dump(metrics, f)
    logger.debug(f"Serialized metrics to {output_file} in JSON format")


def dump_checkpoints(
    output_file: Path,
    checkpoints: list[dict[str, Any]],
):
    """
    Serialize tracked model checkpoints to a torch file.

    Parameters
    ----------
    output_file : Path
        Path to the output torch file.

    checkpoints : list
        List of model checkpoints after each epoch.
    """

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

    # Run the training
    training_results = train_and_validate(
        model,
        datasets["train"],
        datasets["val"],
        unique_parameters_transformed,
        cfg.training,
        "cuda" if torch.cuda.is_available() else "cpu",
    )

    # Serialize transformations, checkpoints, and metrics
    output_dir = Path(cfg.artifacts.training_dir)
    dump_config(output_dir / "config.json", cfg)
    dump_transformations(output_dir / "transformations.pkl", transformations)
    dump_metrics(output_dir / "metrics.json", training_results["metrics"])
    dump_checkpoints(
        output_dir / "checkpoints.pt", training_results["checkpoints"]
    )


if __name__ == "__main__":
    main()
