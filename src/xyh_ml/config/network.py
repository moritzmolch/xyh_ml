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
class NetworkConfig:
    """
    Configuration of the neural network.

    Attributes
    ----------
    hidden_layers : list[HiddenLayer]
        A list of hidden layers of the multi-layer perceptron.
    output_layer : OutputLayer
        The output layer of the multi-layer perceptron.
    """

    hidden_layers: list[HiddenLayer]
    output_layer: OutputLayer


# -----------------------------------------------------------------------------
# Network configuration loader
# -----------------------------------------------------------------------------


def load_network_config(config_file: str) -> NetworkConfig:
    """
    Load the network configuration file.

    The configuration is loaded from a YAML file and validated using the
    `NetworkConfig` dataclass.

    Parameters
    ----------
    config_file : str
        The path to the input configuration file.

    Returns
    -------
    NetworkConfig
        The loaded network configuration.
    """
    # Load the YAML configuration file
    with open(config_file, "r") as f:
        config_data = yaml.safe_load(f)

    # Load and validate the network configuration
    network_config = NetworkConfig(**config_data)

    return network_config
