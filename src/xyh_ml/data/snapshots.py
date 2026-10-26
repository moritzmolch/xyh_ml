"""
Utilities for preprocessing snapshot files and saving them as data frames for
the training.
"""

import logging
import re
from itertools import groupby
from pathlib import Path

import pandas as pd
import uproot

from xyh_ml.config.dataclasses import (
    OutputClass,
    Variable,
)

# Set up logging
logger = logging.getLogger(__name__)


def group_snapshots(
    base_dir: Path,
    campaign: str,
    channel: str,
    output_classes: list[OutputClass],
) -> dict[tuple[str, str, str], list[Path]]:
    """
    Structure snapshot files into groups based on the campaign, the channel,
    and the output class.

    Parameters
    ----------
    base_dir : Path
        The base directory where the snapshot files are located.

    campaign : str
        Name of the data-taking campaign (e.g., `"2024_nano_v15"`).

    channel : str
        Name of the analysis channel (e.g., `"mt"`).

    output_classes : list[OutputClass]
        List of output class specifications.

    Returns
    -------
    dict[tuple[str, str, str], list[Path]]
        A dictionary where the keys are tuples of
        `(campaign, channel, output_class.name)` and the values are lists of
        snapshot file paths corresponding to that group.
    """

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


def save_data_frames(
    snapshot_groups: dict[tuple[str, str, str], list[Path]],
    input_variables: list[Variable],
    input_parameters: list[Variable],
    output: Variable,
    output_classes: list[OutputClass],
    data_frame_dir: Path,
):
    """
    Take snapshot files and merge them into a single data frame for each output
    class.

    The merged data frames contain the input and output variables, the `event`
    index for $n$-fold cross-validation, and the total weight
    (`__total_weight`) of the event. In addtion, the following auxiliary
    columns are added to the data frame:

    - `campaign`: The name of the data-taking campaign.

    - `channel`: The name of the analysis channel.

    - `outputs[0].name`: The name of the output class.

    - `is_signal`: A boolean indicating whether the output class is a signal.
      This property is derived from the `OutputClass` specification.

    - `m_x`: The mass of the X particle. This property is derived from the
      snapshot file name and is only applicable to signal events. For
      background events, this property is set to the fallback value of the
      `m_x` input parameter.

    - `m_y`: The mass of the Y particle. This property is derived from the
      snapshot file name and is only applicable to signal events. For
      background events, this property is set to the fallback value of the
      `m_y` input parameter.

    - `decay_mode`: The decay mode of the signal event. This property is
      derived from the snapshot file name and is only applicable to signal
      events. For background events, this property is set to `"none"`.

    The inference of `m_x`, `m_y`, and `decay_mode` from the snapshot file name
    is based on the following regex pattern:

    ```python
    r"(y2b_h2tau|y2tau_h2b)_mx(\\d+)_my(\\d+)"
    ```

    The output files are stored in HDF5 format in the specified
    `data_frame_dir` with the name of the output class as filename.

    Parameters
    ----------

    snapshot_groups : dict[tuple[str, str, str], list[Path]]
        A dictionary where the keys are tuples of
        `(campaign, channel, output_class.name)` and the values are lists of
        snapshot file paths corresponding to that group.

    input_variables : list[Variable]
        List of input variable specifications.

    input_parameters : list[Variable]
        List of input parameter specifications.

    output : Variable
        The output variable.

    output_classes : list[OutputClass]
        List of output class specifications.

    data_frame_dir : Path
        The directory where the resulting data frames will be saved.
    """

    # Create output directory if it does not exist
    if not data_frame_dir.exists():
        logger.debug(f"Creating output directory {data_frame_dir}")
        data_frame_dir.mkdir(parents=True)

    # Construct input parameters and output class lookup tables
    input_parameters_lookup = {
        parameter.name: parameter for parameter in input_parameters
    }
    output_classes_lookup = {
        output_class.name: output_class for output_class in output_classes
    }

    # Construct list of columns to load from the snapshot files
    snapshot_columns = [
        var.name
        for var in input_variables
        if var.get("is_part_of_snapshot", True)
    ] + [
        "event",
        "__total_weight",
    ]

    # Group snapshots by output class
    # Campaign and channel are added as columns to the output data frame
    sort_key = lambda x: x[0][2]
    for output_class_name, grouped_snapshot_file_items in groupby(
        sorted(snapshot_groups.items(), key=sort_key),
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
                    df_snapshot = f["ntuple"].arrays(
                        snapshot_columns,
                        library="pd",
                    )

                # Add auxiliary columns to the data frame
                df_snapshot[output.name] = output_class_name
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
                    df_snapshot["m_x"] = float(
                        input_parameters_lookup["m_x"].fallback_value
                    )
                    df_snapshot["m_y"] = float(
                        input_parameters_lookup["m_y"].fallback_value
                    )

                dfs.append(df_snapshot)

        # Merge all data frames of this output class into a single object
        df = pd.concat(dfs, ignore_index=True)
        df.to_hdf(data_frame_dir / f"{output_class_name}.h5", key="ntuple")
