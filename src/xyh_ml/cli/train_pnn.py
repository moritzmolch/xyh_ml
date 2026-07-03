import json
import logging
import tempfile
from dataclasses import asdict
from pathlib import Path

import click
import torch
from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from xyh_ml.config.data import load_data_config
from xyh_ml.config.network import load_training_config
from xyh_ml.data.data_frame import (
    load_data_frame_from_config,
    split_data,
)
from xyh_ml.data.datasets import create_pnn_data_loaders
from xyh_ml.data.reweighting import reweight
from xyh_ml.data.transformations import transform
from xyh_ml.lightning.pnn_classifier import create_pnn_model
from xyh_ml.mlflow_utils import get_sqlite_tracking_uri, log_sklearn_artifacts

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


# Default parameters
TRAIN_FRACTION = 0.8
NUM_WORKERS = 4
SEED = 4712


@click.command()
@click.option(
    "--data-config",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=True,
    help="Path to the data configuration file.",
)
@click.option(
    "--training-config",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=True,
    help="Path to the training configuration file.",
)
@click.option(
    "--output-dir",
    type=click.Path(exists=False, dir_okay=True, readable=True),
    required=True,
    help="Path to the output directory.",
)
@click.option(
    "--training-tag",
    type=str,
    required=True,
    help="Tag of the training campaign.",
)
@click.option(
    "--split",
    type=click.Choice(["even", "odd"]),
    required=True,
    help="The data split to use.",
)
def train_pnn(
    *,
    data_config: str,
    training_config: str,
    output_dir: str,
    training_tag: str,
    split: str,
):
    # Convert output directory to a Path object
    output_dir = Path(output_dir)

    # Load the data configuration
    log.debug(f"Loading data configuration from {data_config}")
    data_config = load_data_config(data_config)

    # Load the training configuration
    log.debug(f"Loading training configuration from {training_config}")
    training_config = load_training_config(training_config)

    # Load the training dataset from the configuration
    log.debug("Loading the training dataset")
    data_frame = load_data_frame_from_config(data_config)
    log.debug(f"Training dataset loaded with {len(data_frame)} entries")

    # Get the analysis channel
    channel = data_config.data_source.channel

    # Construct the training output directory
    training_output_dir = output_dir / training_tag / channel / split
    if not training_output_dir.exists():
        training_output_dir.mkdir(parents=True)
    log.debug(f"Training artifacts will be saved to {training_output_dir}")

    # Split the full dataset into
    # - samples with even/odd event number parity
    # - training and test subsets
    # - only keep the subsets of the considered split
    log.debug(
        "Splitting training dataset into even/odd and train/test subsets.\n"
        + f"    Training fraction: {TRAIN_FRACTION}\n"
        + f"    Seed: {SEED}\n"
        + f"    Split: {split}"
    )
    data_frames = {
        k: v
        for k, v in split_data(
            data_frame,
            TRAIN_FRACTION,
            seed=SEED,
        ).items()
        if k[0] == split
    }

    # Reweight the samples
    log.debug("Applying reweighting to the data frames")
    data_frames = reweight(
        data_frames,
        weight_name="weight",
        training_weight_name="training_weight",
        signal_processes=["xyh_y2b_h2tau", "xyh_y2tau_h2b"],
    )

    # Transform the data frames (fit scalers and encoders on the training data,
    # apply transformations to all datasets)
    log.debug("Applying transformations to the data frames")
    data_frames, transforms = transform(data_frames, data_config)

    # Create data loaders for each split and each subset
    log.debug("Creating data loaders for each subset")
    data_loaders, datasets = create_pnn_data_loaders(
        data_frames,
        data_config,
        batch_size=training_config.batch_and_epochs.batch_size,
        weight_name="training_weight",
        signal_flag_name="is_signal",
        num_workers=NUM_WORKERS,
    )

    # Extract possible parameter values
    parameter_values = []
    for dataset in datasets.values():
        parameter_values.append(dataset.parameters)
    parameter_values = torch.cat(parameter_values, dim=0)
    parameter_values = torch.tensor([tuple(x) for x in parameter_values])
    parameter_sampling_values = torch.unique(parameter_values, dim=0)

    # Train even and odd models separately
    log.debug("Creating pytorch model")
    lightning = create_pnn_model(
        input_size=len(data_config.input_variables),
        parameter_size=len(data_config.input_parameters),
        output_size=len(data_config.output_classes),
        training_config=training_config,
        parameter_sampling_values=parameter_sampling_values,
        parameter_sampling_seed=SEED,
    )

    # Define output path for the model
    model_checkpoint_file = training_output_dir / "models" / "best_model.ckpt"
    if not model_checkpoint_file.parent.exists():
        model_checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    if model_checkpoint_file.exists():
        log.warning(
            f"Model checkpoint file {model_checkpoint_file} already exists "
            + "and will be deleted."
        )
        model_checkpoint_file.unlink()

    # Define callbacks to save model and to trigger early stopping
    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            dirpath=str(model_checkpoint_file.parent),
            filename=model_checkpoint_file.stem,
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            **training_config.early_stopping,
        ),
    ]

    # Set up the MLFlow logger with proper SQLite tracking URI
    mlruns_dir = output_dir / "mlruns"
    if not mlruns_dir.exists():
        mlruns_dir.mkdir(parents=True, exist_ok=True)
    sqlite_file = mlruns_dir / "sqlite.db"
    tracking_uri = get_sqlite_tracking_uri(sqlite_file)

    log.debug(
        f"Setting up MLFlow logger for experiment {training_tag} with "
        + f"tracking URI {tracking_uri} and artifact location {mlruns_dir}"
    )
    mlflow_logger = MLFlowLogger(
        experiment_name=training_tag,
        tracking_uri=tracking_uri,
        artifact_location=str(mlruns_dir),
    )

    # Track the metadata as parameters and artifacts
    mlflow_logger.log_hyperparams(
        {
            "training_tag": training_tag,
            "train_split": split,
            "channel": data_config.data_source.channel,
        }
    )

    # Log configurations as JSON artifacts for better readability
    with tempfile.TemporaryDirectory() as tmpdir:
        config_dir = Path(tmpdir)
        training_config_path = config_dir / "training_config.json"
        data_config_path = config_dir / "data_config.json"

        with open(training_config_path, "w") as f:
            json.dump(asdict(training_config), f, indent=2, default=str)
        with open(data_config_path, "w") as f:
            json.dump(asdict(data_config), f, indent=2, default=str)

        mlflow_logger.experiment.log_artifact(
            local_path=str(training_config_path),
            artifact_path="configs",
            run_id=mlflow_logger.run_id,
        )
        mlflow_logger.experiment.log_artifact(
            local_path=str(data_config_path),
            artifact_path="configs",
            run_id=mlflow_logger.run_id,
        )

    # Run the training
    trainer = Trainer(
        max_epochs=training_config.batch_and_epochs.max_epochs,
        callbacks=callbacks,
        logger=mlflow_logger,
        gradient_clip_val=training_config.optimizer.gradient_clip_val,
        devices=1,
        accelerator="auto",
    )
    trainer.fit(
        lightning,
        train_dataloaders=data_loaders[(split, "train")],
        val_dataloaders=data_loaders[(split, "val")],
    )

    # Log the best model
    if model_checkpoint_file.exists():
        mlflow_logger.experiment.log_artifact(
            local_path=str(model_checkpoint_file),
            artifact_path="models",
            run_id=mlflow_logger.run_id,
        )
    log.info(f"Logged best model for split {split}")

    # Log the preprocessing transformers (scaler and encoder)
    # transforms[split] contains [standard_scaler, classes_encoder]
    if split in transforms:
        standard_scaler, classes_encoder = transforms[split]
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts_to_log = {
                "standard_scaler": standard_scaler,
                "classes_encoder": classes_encoder,
            }
            log_sklearn_artifacts(
                artifact_dir=Path(tmpdir),
                artifacts=artifacts_to_log,
                run_id=mlflow_logger.run_id,
            )
        log.info(f"Logged preprocessing artifacts for split {split}")


if __name__ == "__main__":
    train_pnn()
