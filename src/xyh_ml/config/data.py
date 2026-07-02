from typing import Any, Optional

import yaml
from pydantic import Field
from pydantic.dataclasses import dataclass

# -----------------------------------------------------------------------------
# Data configuration dataclasses
# -----------------------------------------------------------------------------


@dataclass
class DataSource:
    """
    Configuration of the data source for training and evaluation.

    Attributes
    ----------
    base_dir : str
        The base directory of the training and evaluation data.
    channel : str
        The channel of the data (e.g., "mt", "et", "tt").
    """

    base_dir: str
    channel: str


@dataclass
class Variable:
    """
    Configuration of a single variable for the training.

    Attributes
    ----------
    name : str
        The name of the variable.
    transformation : str
        The transformation to be applied to the variable.
    """

    name: str
    transformation: str


@dataclass
class Class:
    """
    Configuration of output class of a neural network..

    Attributes
    ----------
    name : str
        The name of the class.
    label : str
        The label of the class to be used in plots.
    file_basenames : list[str]
        A list of file basenames corresponding to the class.
    is_signal : bool
        Whether the class is a signal class or a background class.
    """

    name: str
    label: str
    file_basenames: list[str]
    is_signal: bool


@dataclass
class Reweighting:
    """
    Configuration of the reweighting of the training data.

    Attributes
    ----------
    name : str
        The name of the reweighting method.
    classes : list[str]
        A list of class names to which the reweighting method should be applied.
    """

    name: str
    classes: list[str] | None = None
    parameters: Optional[dict[str, Any]] = Field(default_factory=dict)


@dataclass
class DataConfig:
    """
    Configuration of the data for training and evaluation, as well as the
    neural network's input features and output classes.

    Attributes
    ----------
    classes : list[Class]
        A list of output classes for the neural network.
    """

    data_source: DataSource
    input_variables: list[Variable] | None = Field(default_factory=list)
    input_parameters: list[Variable] | None = Field(default_factory=list)
    output_classes: list[Class] | None = Field(default_factory=list)
    reweightings: list[Reweighting] | None = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Data configuration loader
# -----------------------------------------------------------------------------


def load_data_config(config_file: str) -> DataConfig:
    """
    Load the data configuration file.

    The configuration is loaded from a YAML file and validated using the
    `DataConfig` dataclass.

    Parameters
    ----------
    config_file : str
        The path to the input configuration file.

    Returns
    -------
    DataConfig
        The loaded data configuration.
    """
    # Load the YAML configuration file
    with open(config_file, "r") as f:
        config_data = yaml.safe_load(f)

    # Load and validate the data configuration
    data_config = DataConfig(**config_data)

    return data_config
