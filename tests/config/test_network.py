import os

from xyh_ml.config.network import (
    HiddenLayer,
    NetworkConfig,
    OutputLayer,
    load_network_config,
)

# Directory paths
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "config"))


def test_load_network_config():
    """Test the loading of the network configuration file."""
    # Load the test network configuration
    config_file = os.path.join(CONFIG_DIR, "network.yaml")
    network_config = load_network_config(config_file)

    # Perform consistency checks on the loaded configuration
    assert isinstance(network_config, NetworkConfig)
    assert isinstance(network_config.hidden_layers, list)
    assert all(
        (isinstance(h, HiddenLayer) for h in network_config.hidden_layers)
    )
    assert isinstance(network_config.output_layer, OutputLayer)
