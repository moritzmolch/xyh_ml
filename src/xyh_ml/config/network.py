from typing import Any

import yaml
from pydantic.dataclasses import dataclass

# -----------------------------------------------------------------------------
# Network configuration dataclasses
# -----------------------------------------------------------------------------


@dataclass
class HiddenLayer:
    """
    Configuration of a single hidden layer of a multi-layer perceptron.

    Attributes
    ----------
    size : int
        The number of neurons in the layer.
    activation : str
        The activation function the layer is followed by.
    norm : str
        The normalization technique to be used in the layer.
    dropout : float | str
        The dropout fraction.
    """

    size: int
    activation: str
    norm: str
    dropout: float | str


@dataclass
class OutputLayer:
    """
    Configuration of the output layer of a multi-layer perceptron.

    Attributes
    ----------
    activation : str
        The activation function the output layer is followed by.
    """

    activation: str


@dataclass
class LossConfig:
    """
    Configuration of the loss function.

    Attributes
    ----------
    loss_fn : str
        The class name of the loss function.
    kwargs : dict[str, Any]
        Additional keyword arguments for the loss function.
    """

    fn_name: str
    kwargs: dict[str, Any]


@dataclass
class OptimizerConfig:
    """
    Configuration of the optimizer.

    Attributes
    ----------
    class_name : str
        The class name of the optimizer (e.g., "Adam", "SGD").
    kwargs : dict[str, Any]
        The keyword arguments for the optimizer constructor (e.g., learning rate, weight decay).
    """

    class_name: str
    kwargs: dict[str, Any]
    gradient_clip_val: float


@dataclass
class BatchAndEpochsConfig:
    """
    Configuration of the batch size and the number of epochs.

    Attributes
    ----------
    size : int
        The batch size to be used during training.
    max_epochs : int
        The maximum number of epochs to train for.
    """

    max_epochs: int
    batch_size: int


@dataclass
class TrainingConfig:
    """
    Configuration of the neural network.

    Attributes
    ----------
    hidden_layers : list[HiddenLayer]
        A list of hidden layers of the multi-layer perceptron.
    output_layer : OutputLayer
        The output layer of the multi-layer perceptron.
    batch_and_epochs : BatchAndEpochsConfig
        The batch size and epoch configuration.
    loss : LossConfig
        The loss function configuration.
    optimizer : OptimizerConfig
        The optimizer configuration.
    early_stopping : dict[str, Any]
        The early stopping configuration.
    """

    hidden_layers: list[HiddenLayer]
    output_layer: OutputLayer
    batch_and_epochs: BatchAndEpochsConfig
    loss: LossConfig
    optimizer: OptimizerConfig
    early_stopping: dict[str, Any]


# -----------------------------------------------------------------------------
# Network configuration loader
# -----------------------------------------------------------------------------


def load_training_config(config_file: str) -> TrainingConfig:
    """
    Load the training configuration file.

    The configuration is loaded from a YAML file and validated using the
    `TrainingConfig` dataclass.

    Parameters
    ----------
    config_file : str
        The path to the input configuration file.

    Returns
    -------
    TrainingConfig
        The loaded training configuration.
    """
    # Load the YAML configuration file
    with open(config_file, "r") as f:
        config_data = yaml.safe_load(f)

    # Load and validate the training configuration
    training_config = TrainingConfig(**config_data)

    return training_config
