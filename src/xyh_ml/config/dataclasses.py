"""
Dataclasses representing the configuration structure of the project.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Tags:
    snapshots_tag: str
    training_tag: str


@dataclass
class Variable:
    name: str
    transformation: str
    fallback_value: float = field(default=-10.0)
    is_part_of_snapshot: bool = field(default=True)


@dataclass
class OutputClass:
    name: str
    label: str
    file_basenames: list[str]
    is_signal: bool


@dataclass
class Inputs:
    variables: list[Variable]
    parameters: list[Variable]


@dataclass
class Outputs:
    variables: list[Variable]
    classes: list[OutputClass]


@dataclass
class Artifacts:
    data_frame_dir: str
    training_dir: str
    predictions_dir: str


@dataclass
class DataSource:
    base_dir: str
    campaigns: list[str]
    channels: list[str]


@dataclass
class Mode:
    decay_mode: str
    n_folds: int
    fold: int
    channel: str


@dataclass
class Dataset:
    train_frac: float


@dataclass
class Network:
    hidden_layers: list[int]
    activation: str
    dropout: float
    activation_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Training:
    max_epochs: int
    batch_size: int
    optimizer_learning_rate: float
    optimizer_weight_decay: float
    data_loader_num_workers: int
    data_loader_pin_memory: bool


@dataclass
class Config:
    mode: Mode
    tags: Tags
    data_source: DataSource
    inputs: Inputs
    outputs: Outputs
    artifacts: Artifacts
    dataset: Dataset
    network: Network
    training: Training
