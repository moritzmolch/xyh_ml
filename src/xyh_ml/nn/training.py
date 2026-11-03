import copy
import logging
from typing import Any

import torch
from torch import Generator, Tensor
from torch.nn import Module
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from xyh_ml.config.dataclasses import Training
from xyh_ml.constants import PARAMETER_SAMPLING_SEED
from xyh_ml.nn.losses import get_loss_fn
from xyh_ml.nn.parameters import resample_input_parameters

# Set up logging
logger = logging.getLogger(__name__)


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
    logger.debug(f"Using loss function: {loss_fn}")

    # Create the optimizer
    optimizer = Adam(
        model.parameters(),
        lr=optimizer_learning_rate,
        weight_decay=optimizer_weight_decay,
    )

    # Create the data loaders for training and validation
    data_loader_train = DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_loader_num_workers,
        pin_memory=data_loader_pin_memory,
    )
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_loader_num_workers,
        pin_memory=data_loader_pin_memory,
    )

    # Move the model to the training device
    model = model.to(device)

    # Create the generator for parameter sampling
    sampling_rng = Generator(device=device)
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
