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
    taylor_dir: str | None = field(default=None)


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
class Plotting:
    merge_folds: bool = field(default=False)


@dataclass
class Taylor:
    # Benchmark (m_x, m_y) hypotheses the Taylor coefficients are evaluated at
    mass_points: list[list[float]]
    n_samples_per_category: int | None = field(default=5000)
    batch_size: int = field(default=4096)
    max_order: int = field(default=2)
    # Compute/plot TCs with respect to the input parameters (m_x, m_y) too
    include_parameters: bool = field(default=False)
    # Number of coefficients with the largest (absolute) median to plot per
    # output node, true class and mass point (None: all)
    max_coefficients: int | None = field(default=45)
    per_plot: int = field(default=15)
    # Summarize and rank the absolute values of the coefficients
    absolute: bool = field(default=True)
    # Draw the 68% interval (one-sided [0, q68] if absolute, else central)
    show_intervals: bool = field(default=True)
    # Channels combined in the ranking of the input variables
    ranking_channels: list[str] = field(
        default_factory=lambda: ["et", "mt", "tt"]
    )


@dataclass
class Dataset:
    train_frac: float


@dataclass
class Network:
    hidden_layers: list[int]
    activation: str
    dropout_frac: float
    activation_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Optimizer:
    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Loss:
    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Training:
    max_epochs: int
    batch_size: int
    optimizer_learning_rate: float
    optimizer_weight_decay: float
    data_loader_num_workers: int
    data_loader_pin_memory: bool
    loss_fn: str
    loss_fn_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Campaign:
    name: str
    era: str


@dataclass
class Channel:
    name: str


@dataclass
class Friend:
    name: str
    include_channels: list[str] | None = field(default=None)
    exclude_channels: list[str] | None = field(default=None)
    include_campaigns: list[str] | None = field(default=None)
    exclude_campaigns: list[str] | None = field(default=None)
    include_samples: list[str] | None = field(default=None)
    exclude_samples: list[str] | None = field(default=None)

    def __post_init__(self):
        # Ensure that only include_* or exclude_* are set, but not both
        if self.include_channels and self.exclude_channels:
            raise ValueError(
                f"Friend {self.name} cannot have both include and exclude "
                + "channels."
            )
        if self.include_campaigns and self.exclude_campaigns:
            raise ValueError(
                f"Friend {self.name} cannot have both include and exclude "
                + "campaigns."
            )
        if self.include_samples and self.exclude_samples:
            raise ValueError(
                f"Friend {self.name} cannot have both include and exclude "
                + "samples."
            )


@dataclass
class NTupleSource:
    tag: str
    xrootd_server: str
    base_dir: str
    friends: list[Friend]
    campaigns: list[str] | None = field(default=None)
    samples: list[str] | None = field(default=None)


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
    campaigns: list[Campaign]
    channels: list[Channel]
    ntuple: NTupleSource
    plotting: Plotting
    taylor: Taylor | None = field(default=None)
