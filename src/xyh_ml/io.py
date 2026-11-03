import json
import logging
import pickle
from pathlib import Path
from typing import Any

import torch

from xyh_ml.config.dataclasses import Config

logger = logging.getLogger(__name__)


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
