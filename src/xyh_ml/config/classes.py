import yaml
from pydantic.dataclasses import dataclass

# -----------------------------------------------------------------------------
# Output class configuration dataclasses
# -----------------------------------------------------------------------------


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
class ClassesConfig:
    """
    Configuration of the neural network.

    Attributes
    ----------
    classes : list[Class]
        A list of output classes for the neural network.
    """

    classes: list[Class]


# -----------------------------------------------------------------------------
# Network configuration loader
# -----------------------------------------------------------------------------


def load_classes_config(config_file: str) -> ClassesConfig:
    """
    Load the output class configuration file.

    The configuration is loaded from a YAML file and validated using the
    `ClassesConfig` dataclass.

    Parameters
    ----------
    config_file : str
        The path to the input configuration file.

    Returns
    -------
    ClassesConfig
        The loaded output class configuration.
    """
    # Load the YAML configuration file
    with open(config_file, "r") as f:
        config_data = yaml.safe_load(f)

    # Load and validate the output class configuration
    classes_config = ClassesConfig(**config_data)

    return classes_config
