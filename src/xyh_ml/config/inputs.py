import yaml
from pydantic import Field
from pydantic.dataclasses import dataclass

# -----------------------------------------------------------------------------
# Inputs configuration dataclasses
# -----------------------------------------------------------------------------


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
class InputsConfig:
    """
    Configuration of training inputs.

    Attributes
    ----------
    variables : list[Variable] | None
        A list of variables to be used for training. Each variable is defined
        by a name and a transformation.
    parameters : list[Variable] | None
        A list of parameters to be used for training. Each parameter is defined
        by a name and a transformation.
    """

    variables: list[Variable] | None = Field(default_factory=list)
    parameters: list[Variable] | None = Field(default_factory=list)


# -----------------------------------------------------------------------------
# Inputs configuration loader
# -----------------------------------------------------------------------------


def load_inputs_config(config_file: str) -> InputsConfig:
    """
    Load the input configuration file.

    The configuration is loaded from a YAML file and validated using the
    `InputsConfig` dataclass.

    Parameters
    ----------
    config_file : str
        The path to the input configuration file.

    Returns
    -------
    InputsConfig
        The loaded input configuration.
    """
    # Load the YAML configuration file
    with open(config_file, "r") as f:
        config_data = yaml.safe_load(f)

    # Load and validate the input configuration
    inputs_config = InputsConfig(**config_data)

    return inputs_config
