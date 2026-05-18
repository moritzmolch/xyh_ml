import os

from xyh_ml.config.inputs import InputsConfig, Variable, load_inputs_config

# Directory paths
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "config"))


def test_load_inputs_config():
    """Test the loading of the input configuration file."""
    # Load the test input configuration
    config_file = os.path.join(CONFIG_DIR, "inputs.yaml")
    inputs_config = load_inputs_config(config_file)

    # Perform consistency checks on the loaded configuration
    assert isinstance(inputs_config, InputsConfig)
    assert isinstance(inputs_config.variables, list)
    assert all((isinstance(v, Variable) for v in inputs_config.variables))
    assert isinstance(inputs_config.parameters, list)
    assert all((isinstance(v, Variable) for v in inputs_config.parameters))
