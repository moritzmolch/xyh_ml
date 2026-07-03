from math import sqrt

import torch
from torch import nn

from xyh_ml.config.network import HiddenLayer, OutputLayer

# Available normalization layers
NORM_LAYERS = {
    "batch_norm_1d": nn.BatchNorm1d,
    "layer_norm": nn.LayerNorm,
    "none": None,
}


# Available activation layers
ACTIVATION_LAYERS = {
    "relu": nn.ReLU,
    "lrelu": nn.LeakyReLU,
    "elu": nn.ELU,
    "prelu": nn.PReLU,
    "softplus": nn.Softplus,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "softmax": nn.Softmax,
    "none": None,
}


def _get_norm_layer(norm: str, dim_out: int):
    """Helper to get the normalization layer object from the name."""
    try:
        norm_layer = NORM_LAYERS.get(norm)
    except KeyError:
        raise ValueError(f"Norm layer {norm} is not implemented")

    return norm_layer(dim_out) if norm_layer is not None else None


def _get_activation_layer(activation: str):
    """Helper to get the activation layer object from the name."""
    try:
        activation_layer = ACTIVATION_LAYERS.get(activation)
    except KeyError:
        raise ValueError(f"Activation layer {activation} is not implemented")

    return activation_layer() if activation_layer is not None else None


def _get_dropout_layer(dropout: str | float):
    """Helper to get the dropout layer object from the fraction or name."""
    if isinstance(dropout, str) and dropout == "none":
        return None
    elif dropout < 0.0 or dropout > 1.0:
        raise ValueError(
            "Dropout fraction must have a value between 0 and 1, fraction has "
            + f"value {dropout} here"
        )
    elif dropout == 0.0:
        return None

    return nn.Dropout(p=dropout)


class MLPBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: str = "none",
        norm: str = "none",
        dropout: float = 0.0,
    ):
        super(MLPBlock, self).__init__()

        # Save block size parameters as attributes
        self.input_size = input_size
        self.output_size = output_size

        # Linear layer that maps inputs to outputs with a linear map
        self.linear_layer = nn.Linear(self.input_size, self.output_size)

        # Activation function
        self.activation_layer = _get_activation_layer(activation)

        # Batch normalization
        self.norm_layer = _get_norm_layer(norm, output_size)

        # Dropout layer
        self.dropout_layer = _get_dropout_layer(dropout)

    def forward(self, inputs):
        _tmp = inputs
        _tmp = self.linear_layer(_tmp)
        if self.activation_layer is not None:
            _tmp = self.activation_layer(_tmp)
        if self.norm_layer is not None:
            _tmp = self.norm_layer(_tmp)
        if self.dropout_layer is not None:
            _tmp = self.dropout_layer(_tmp)
        return _tmp


class MultiLayerPerceptron(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: list[HiddenLayer],
        output_layer: OutputLayer,
    ):
        # Initialize the parent class
        super(MultiLayerPerceptron, self).__init__()

        # Size parameters of the network
        self.input_size = input_size
        self.output_size = output_size

        # Hidden and output layer specifications
        self.hidden_layers_spec = hidden_layers
        self.output_layer_spec = output_layer
        self.hidden_features = self.hidden_layers_spec[0].size

        # First block that also receives the MLP input
        _input_spec = self.hidden_layers_spec[0]
        self.input_block = MLPBlock(
            input_size=input_size,
            output_size=_input_spec.size,
            activation=_input_spec.activation,
            norm=_input_spec.norm,
            dropout=_input_spec.dropout,
        )

        # Blocks with only hidden layers
        self.hidden_blocks = []
        for i in range(1, len(self.hidden_layers_spec)):
            _prev_spec = self.hidden_layers_spec[i - 1]
            _spec = self.hidden_layers_spec[i]
            _hidden_block = MLPBlock(
                input_size=_prev_spec.size,
                output_size=_spec.size,
                activation=_spec.activation,
                norm=_spec.norm,
                dropout=_spec.dropout,
            )
            self.hidden_blocks.append(_hidden_block)
        self.hidden_blocks = nn.ModuleList(self.hidden_blocks)

        # Block with the output layer
        _prev_spec = self.hidden_layers_spec[-1]
        _output_spec = self.output_layer_spec
        self.output_block = MLPBlock(
            input_size=_prev_spec.size,
            output_size=self.output_size,
            activation=_output_spec.activation,
            norm=None,
            dropout=0.0,
        )

        # Initialize all network weights with Xavier normal method
        self._xavier_normal_init()

    def forward(self, input, context=None):
        # Start with the plain input
        tmp = input

        # Concatenate the context to the plain input
        if context is not None:
            tmp = torch.cat((tmp, context), dim=-1)

        # Process the concatenated vector through the MLP blocks
        tmp = self.input_block(tmp)
        for hidden_block in self.hidden_blocks:
            tmp = hidden_block(tmp)
        tmp = self.output_block(tmp)
        return tmp

    def _xavier_normal_init(self):
        # Apply Xavier normal weight initialization to network parameters
        for name, param in self.named_parameters():
            if name.endswith(".linear_layer.weight"):
                std = sqrt(2 / (param.shape[0] + param.shape[1]))
                param.data.normal_(0, std)
            elif name.endswith(".linear_layer.bias"):
                param.data.fill_(0)
