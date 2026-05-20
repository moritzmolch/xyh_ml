import os

from xyh_ml.config.classes import Class, ClassesConfig, load_classes_config

# Directory paths
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "config"))


def test_load_classes_config():
    """Test the loading of the output class configuration file."""
    # Load the test output class configuration
    config_file = os.path.join(CONFIG_DIR, "classes.yaml")
    classes_config = load_classes_config(config_file)

    # Perform consistency checks on the loaded configuration
    assert isinstance(classes_config, ClassesConfig)
    assert isinstance(classes_config.classes, list)
    assert all((isinstance(c, Class) for c in classes_config.classes))
