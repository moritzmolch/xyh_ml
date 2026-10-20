import logging
import re
from collections.abc import Generator
from dataclasses import dataclass, field
from itertools import groupby, product
from pathlib import Path

import hydra
import pandas as pd
import uproot
from hydra.core.config_store import ConfigStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


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


@dataclass
class DataSource:
    base_dir: str
    campaigns: list[str]
    channels: list[str]


@dataclass
class Mode:
    decay_mode: str
    event_parity: int


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


def yield_base_dirs(
    data_source: DataSource,
) -> Generator[tuple[tuple[str, str], Path], None, None]:
    for campaign, channel in product(
        data_source.campaigns, data_source.channels
    ):
        yield (
            (campaign, channel),
            Path(
                data_source.base_dir.format(campaign=campaign, channel=channel)
            ),
        )


cs = ConfigStore.instance()
cs.store(name="preprocess_snapshots", node=Config)


# -----------------------------------------------------------------------------
# Snapshot grouping
# -----------------------------------------------------------------------------


def group_snapshots(
    base_dir: Path,
    campaign: str,
    channel: str,
    output_classes: list[OutputClass],
) -> dict[tuple[str, str, str], list[Path]]:

    # Get unstructured list of all snapshot files for this campaign and channel
    snapshot_files = base_dir.glob("*.root")

    # Container for snapshot file groups divided by output classes
    snapshot_groups: dict[tuple[str, str, str], list[Path]] = {}

    # Create a lookup table to map file basenames to output classes
    file_basename_output_class_map = {
        file_basename: output_class.name
        for output_class in output_classes
        for file_basename in output_class.file_basenames
    }

    # Iterate through output files and map them to their corresponding output
    # class
    for snapshot_file in snapshot_files:
        # Add counter to check if the snapshot file has been matched
        match_count = 0

        # Iterate through basenames and check if the snapshot file name matches
        # any of the patterns
        for (
            file_basename,
            output_class_name,
        ) in file_basename_output_class_map.items():
            if re.match(file_basename, snapshot_file.name):
                snapshot_groups.setdefault(
                    (campaign, channel, output_class_name),
                    [],
                ).append(snapshot_file)
                match_count += 1

        # Add debug info if the snapshot file has been matched to no output
        # class
        if match_count == 0:
            logger.debug(
                f"Snapshot file {snapshot_file} not matched to any output "
                + "class."
            )

        # Add warning if the snapshot file has been matched to more than one
        # output class
        if match_count > 1:
            logger.warning(
                f"Snapshot file {snapshot_file} matched to more than one "
                + "output class. Please check the file name patterns."
            )

    return snapshot_groups


def merge_snapshots(
    snapshot_files: dict[tuple[str, str, str], list[Path]],
    inputs: Inputs,
    outputs: Outputs,
    output_classes: list[OutputClass],
    data_frame_dir: Path,
):
    # Create output directory if it does not exist
    if not data_frame_dir.exists():
        data_frame_dir.mkdir(parents=True)

    # Construct input parameters lookup table
    input_parameters_lookup = {
        parameter.name: parameter for parameter in inputs.parameters
    }

    # Construct output class lookup tables
    output_classes_lookup = {
        output_class.name: output_class for output_class in output_classes
    }

    # Construct list of columns to load from the snapshot files
    columns = [var.name for var in inputs.variables] + [
        # "event",
        "__total_weight",
    ]

    # Group snapshots by output class. Campaign and channel are added as
    # columns to the output data frame
    sort_key = lambda x: x[0][2]
    for output_class_name, grouped_snapshot_file_items in groupby(
        sorted(snapshot_files.items(), key=sort_key),
        key=sort_key,
    ):
        logger.info(f"Merging snapshots for output class {output_class_name}")

        # Get the output class
        output_class = output_classes_lookup[output_class_name]

        # List of data frames loaded from the snapshot files
        dfs = []

        for (
            (campaign, channel, _),
            snapshot_files,
        ) in grouped_snapshot_file_items:
            for snapshot_file in snapshot_files:
                with uproot.open(snapshot_file) as f:
                    df_snapshot = f["ntuple"].arrays(columns, library="pd")

                # Add auxiliary columns to the data frame
                df_snapshot["class"] = output_class_name
                df_snapshot["is_signal"] = output_class.is_signal
                df_snapshot["campaign"] = campaign
                df_snapshot["channel"] = channel

                # For signal events, add decay mode, m_x, and m_y columns. Set
                # default values for background events.
                if output_class.is_signal and (
                    match := re.search(
                        r"(y2b_h2tau|y2tau_h2b)_mx(\d+)_my(\d+)",
                        snapshot_file.name,
                    )
                ):
                    df_snapshot["decay_mode"] = match.group(1)
                    df_snapshot["m_x"] = float(match.group(2))
                    df_snapshot["m_y"] = float(match.group(3))
                else:
                    df_snapshot["decay_mode"] = "none"
                    df_snapshot["m_x"] = input_parameters_lookup[
                        "m_x"
                    ].fallback_value
                    df_snapshot["m_y"] = input_parameters_lookup[
                        "m_y"
                    ].fallback_value

                dfs.append(df_snapshot)

        # Merge all data frames of this output class into a single object
        df = pd.concat(dfs, ignore_index=True)
        df.to_hdf(data_frame_dir / f"{output_class_name}.h5", key="ntuple")


@hydra.main(
    config_path="../../../config",
    config_name="preprocess_snapshots",
    version_base="1.3",
)
def main(cfg: Config):
    # Get snapshot groups for all campaigns and channels
    snapshot_groups: dict[tuple[str, str, str], list[Path]] = {}

    for (campaign, channel), base_dir in yield_base_dirs(cfg.data_source):
        # Construct snapshot groups for this era and channel
        _snapshot_groups = group_snapshots(
            base_dir, campaign, channel, cfg.outputs.classes
        )

        # Check if any keys are overlapping with the existing groups
        if not snapshot_groups.keys().isdisjoint(_snapshot_groups.keys()):
            raise ValueError(
                "Overlapping keys (campaign, channel, output_class) found in "
                + "snapshot groups."
            )

        # Append current group to the global list of snapshot groups
        snapshot_groups.update(_snapshot_groups)

    # Store merged data frames to disk
    merge_snapshots(
        snapshot_groups,
        cfg.inputs,
        cfg.outputs,
        cfg.outputs.classes,
        Path(cfg.artifacts.data_frame_dir),
    )


if __name__ == "__main__":
    main()
