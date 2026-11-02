import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Literal

import lightning
import pandas as pd
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


class PNNLightning(lightning.LightningModule):
    def __init__(
        self,
        model: Module,
        loss_fn: Callable[[Tensor, Tensor, Tensor], Tensor],
        optimizer_kwargs: dict[str, Any],
        unique_parameters: Tensor,
    ):
        super().__init__()

        # Store input parameters as attributes
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_kwargs = optimizer_kwargs
        self.unique_parameters = unique_parameters

        # Create the generator for parameter sampling
        self.sampling_rng = Generator(device=self.device)
        self.sampling_rng.manual_seed(PARAMETER_SAMPLING_SEED)

        # Internal store to track individual losses during epoch for proper
        # averaging
        # Format: {<loop_type>: [(batch_loss, batch_weight_sum), ...], ...}
        self._epoch_losses = {}

    def forward(self, x: Tensor) -> Tensor:
        # Forward input to model
        return self.model(x)

    def _eval_model_and_loss_step(
        self,
        batch: tuple[Tensor, ...],
        batch_idx: int,
    ):
        """
        Evaluate the model and calculate the loss for a given batch.

        These steps are used for both training and validation, so they are
        factored out into a separate method.
        """

        # Expand the batch input into its components
        # - x: input features
        # - p: input parameters
        # - y: target labels
        # - w: event weights
        # - r: flags whether parameters need to be resampled
        x, p, y, w, r = batch

        # Resample parameters for events that are marked for resampling
        p = resample_input_parameters(
            p,
            r,
            self.unique_parameters,
            rng=self.sampling_rng,
        )

        # Propagate the input through the model
        x_and_p = torch.cat((x, p), dim=-1)
        y_hat = self.model(x_and_p)

        # Calculate the loss
        loss = self.loss_fn(y_hat, y.view(-1), w)

        # Calculate the unweighted loss; does not need gradient tracking, as it
        # is only used for logging
        with torch.no_grad():
            loss_unweighted = self.loss_fn(
                y_hat, y.view(-1), torch.ones_like(w)
            )

        return {
            "x": x,
            "p": p,
            "y": y,
            "w": w,
            "r": r,
            "y_hat": y_hat,
            "loss": loss,
            "loss_unweighted": loss_unweighted,
        }

    def _log_step(
        self,
        eval_results: dict[str, Any],
        batch_idx: int,
        loop_type: Literal["train", "val"],
    ):
        """
        Log the results of a training or validation step.

        This method is used for both training and validation, so it is factored
        out into a separate method.
        """

        # Loss value
        loss = eval_results["loss"].item()
        self.log(
            f"{loop_type}_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # Loss value, unweighted
        loss_unweighted = eval_results["loss_unweighted"].item()
        self.log(
            f"{loop_type}_loss_unweighted",
            loss_unweighted,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        # Sum of weights
        sum_w = eval_results["w"].sum().item()
        self.log(
            f"{loop_type}_weight_sum",
            sum_w,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )

        # Number of training events
        self.log(
            f"{loop_type}_count",
            eval_results["w"].shape[0],
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )

        # Epoch
        self.log(
            f"{loop_type}_epoch",
            self.current_epoch,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

        # Batch index
        self.log(
            f"{loop_type}_batch_index",
            batch_idx,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
        )

        # Store the loss for the epoch average calculation
        self._epoch_losses[loop_type].append((loss, sum_w))

    def training_step(self, batch, batch_idx):
        # Run model and loss evaluation
        eval_results = self._eval_model_and_loss_step(batch, batch_idx)

        # Get the loss, which is used for backpropagation
        loss = eval_results["loss"]

        # Log metrics
        self._log_step(eval_results, batch_idx, "train")

        return loss

    def validation_step(self, batch, batch_idx):
        # Run model and loss evaluation
        eval_results = self._eval_model_and_loss_step(batch, batch_idx)

        # Get the loss, which is used for backpropagation
        loss = eval_results["loss"]

        # Log metrics
        self._log_step(eval_results, batch_idx, "train")

        return loss

    def _log_average_epoch_loss(self, loop_type: Literal["train", "val"]):
        # Calculate average loss for the epoch
        losses_and_weight_sums = self._epoch_losses[loop_type]
        avg_loss = 0
        total_weight_sum = 0
        for loss, weight_sum in zip(*losses_and_weight_sums):
            avg_loss += loss * weight_sum
            total_weight_sum += weight_sum
        avg_loss /= total_weight_sum

        # Add average loss to central logger
        self.logger.log_metrics(
            f"{loop_type}_loss",
            avg_loss,
            step=self.global_step,
        )

        # Clear the stored losses for the epoch
        self._epoch_losses[loop_type].clear()

    def on_training_epoch_end(self):
        # Log the average training loss for the epoch
        self._log_average_epoch_loss("train")

    def on_validation_epoch_end(self):
        # Log the average validation loss for the epoch
        self._log_average_epoch_loss("val")

    def configure_optimizers(self):
        # Create the optimizer
        optimizer = Adam(
            self.model.parameters(),
            **self.optimizer_kwargs,
        )

        # Create the learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "train_loss",
            },
        }


def train_and_validate(
    model: Module,
    dataset_train: TensorDataset,
    dataset_val: TensorDataset,
    unique_parameters: Tensor,
    training: Training,
    device: str,
    n_steps_logging: int = 25,
) -> lightning.LightningModule:
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
    data_loader_num_workers = training.data_loader_num_workers
    data_loader_pin_memory = training.data_loader_pin_memory

    # Get the loss function
    loss_fn = get_loss_fn(training.loss_fn, training.loss_fn_kwargs)
    logger.debug(
        f"Using loss function {loss_fn} with kwargs {training.loss_fn_kwargs}"
    )

    # Collect optimizer's kwargs
    optimizer_kwargs = {
        "lr": training.optimizer_learning_rate,
        "weight_decay": training.optimizer_weight_decay,
    }

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

    # Create the PyTorch Lightning module
    lightning_module = PNNLightning(
        model=model,
        loss_fn=loss_fn,
        optimizer_kwargs=optimizer_kwargs,
        unique_parameters=unique_parameters,
    )

    # Create the EarlyStopping callback
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=training.early_stopping_patience,
        min_delta=training.early_stopping_min_delta,
        verbose=False,
    )

    # Create a temporary directory for storing checkpoints during training
    tmp_parent_dir = Path(os.getcwd(), "tmp")
    if not tmp_parent_dir.exists():
        tmp_parent_dir.mkdir(parents=True)
        logger.debug(f"Created temporary directory {tmp_parent_dir}")
    with tempfile.TemporaryDirectory(
        dir=tmp_parent_dir,
        delete=False,
    ) as tmp_results_dir:
        logger.info(
            f"Created temporary directory for training results: {tmp_results_dir}"
        )
        # Create the checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(tmp_results_dir, "checkpoints"),
            filename="best-model_epoch{epoch}_step{step}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            verbose=False,
        )

        # Create the checkpoint callback
        training_logger = lightning.pytorch.loggers.CSVLogger(
            save_dir=os.path.join(tmp_results_dir, "logs"),
            name=None,
        )

        # Create the PyTorch Lightning trainer and run the training
        device = "gpu" if torch.cuda.is_available() else "cpu"
        trainer = lightning.Trainer(
            max_epochs=max_epochs,
            callbacks=[
                early_stopping_callback,
                checkpoint_callback,
            ],
            logger=training_logger,
            accelerator=device,
            devices=1,
            enable_progress_bar=True,
        )
        trainer.fit(
            lightning_module,
            train_dataloaders=data_loader_train,
            val_dataloaders=data_loader_val,
        )

        # Load the best model checkpoint after training
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path:
            logger.debug(
                f"Loading best model from checkpoint: {best_model_path}"
            )
            lightning_module.load_from_checkpoint(best_model_path)

        # Load the logs after training
        logs_path = logging_callback.log_dir / "metrics.csv"
        if logs_path.exists():
            logger.debug(f"Loading training logs from: {logs_path}")
            logs = pd.read_csv(logs_path)

    return lightning_module, logs
