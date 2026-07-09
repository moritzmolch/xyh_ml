import torch
from typing import Any


def _create_activation_layer(activation: str, **kwargs):
    # Map activation function names to torch classes
    activation_layers = {
        "relu": torch.nn.ReLU,
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

    return torch.nn.Dropout(dropout_frac)


class MultiLayerPerceptron(torch.nn.Module):

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
        self.layers = torch.nn.ModuleList()
        last_size = self.input_size
        for hidden_layer_size in self.hidden_layer_sizes:
            # Add hidden linear layer
            self.layers.append(torch.nn.Linear(last_size, hidden_layer_size))

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
        self.layers.append(torch.nn.Linear(last_size, output_size))

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x
