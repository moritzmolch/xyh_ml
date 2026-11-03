from typing import Any

import torch
from torch.nn import Dropout, Linear, Module, ModuleList, ReLU

from xyh_ml.config.dataclasses import Network


def _create_activation_layer(activation: str, **kwargs):
    # Map activation function names to torch classes
    activation_layers = {
        "relu": ReLU,
    }

    # Raise an exception if the activation function is not defined
    if activation not in activation_layers:
        raise ValueError(f"Activation function '{activation}' not implemented")

    return activation_layers[activation](**kwargs)


def _create_dropout_layer(dropout_frac: float):
    # Raise an exception if the dropout fraction is smaller than 0 or greater
    # than 1
    if dropout_frac < 0.0 or dropout_frac > 1.0:
        raise ValueError(
            f"Dropout fraction is {dropout_frac}, value must be located in "
            + "interval [0, 1]"
        )

    return Dropout(dropout_frac)


class MultiLayerPerceptron(Module):
    """
    A simple multi-layer perceptron model.

    Attributes
    ----------
    input_size : int
        Size of the input layer.

    output_size : int
        Size of the output layer.

    hidden_layer_sizes : list[int]
        Number of neurons per hidden layer. The length of the list defines the
        number of hidden layers.

    activation : str
        Name of the activation function to use in the hidden layers.

    dropout_frac : float
        Fraction of nodes per hidden layer to deactivate randomly during
        training. Must be a number between 0 an 1.

    activation_kwargs : dict[str, Any]
        Keyword arguments passed to the activation function. (default: `{}`)
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layer_sizes: list[int],
        activation: str,
        dropout_frac: float,
        activation_kwargs: dict[str, Any] | None = None,
    ):
        # Initialize base class
        super().__init__()

        # Store input parameters as attributes
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.dropout_frac = dropout_frac
        self.activation_kwargs = activation_kwargs or {}

        # Arrange neural network layers
        self.layers = ModuleList()
        last_size = self.input_size
        for hidden_layer_size in self.hidden_layer_sizes:
            # Add hidden linear layer
            self.layers.append(Linear(last_size, hidden_layer_size))

            # Add activation function layer
            self.layers.append(
                _create_activation_layer(
                    self.activation,
                    **self.activation_kwargs,
                )
            )

            # Add dropout layer
            if dropout_frac != 0.0:
                self.layers.append(_create_dropout_layer(dropout_frac))

            # Remember output size of this linear layer
            last_size = hidden_layer_size

        # Add output layer
        self.layers.append(Linear(last_size, output_size))

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x


def create_multi_layer_perceptron_model(
    input_size: int,
    output_size: int,
    network: Network,
) -> MultiLayerPerceptron:
    """
    Create a simple multi-layer perceptron model.

    Parameters
    ----------
    input_size : int
        Number of input features.

    output_size : int
        Number of output classes.

    network : Network
        Network configuration containing hyperparameter settings of the network
        architecture.

    Returns
    -------
    MultiLayerPerceptron
        Multi-layer perceptron model with the specified architecture and
        hyperparameters.
    """

    # Create the model
    model = MultiLayerPerceptron(
        input_size,
        output_size,
        network.hidden_layers,
        network.activation,
        network.dropout_frac,
        activation_kwargs=network.activation_kwargs,
    )

    return model
