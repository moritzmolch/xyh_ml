import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd
import uproot


def _parse_arguments():
    parser = argparse.ArgumentParser(
        description="Preprocess snapshots and dump to HDF5 files."
    )
    parser.add_argument(
        "--data-config-file",
        type=Path,
        required=True,
        help="Path to the data configuration YAML file.",
    )
    parser.add_argument(
        "--training-tag",
        type=str,
        required=True,
        help="Training tag to identify the training run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save the preprocessed HDF5 files.",
    )
    parser.add_argument(
        "--campaign",
        type=str,
        required=True,
        help="Campaign to process.",
    )
    parser.add_argument(
        "--channel",
        type=str,
        required=True,
        help="Channel to process.",
    )
    parser.add_argument(
        "--category",
        type=str,
        required=True,
        help="Category to process.",
    )
    return vars(parser.parse_args())


def _load_data_config(data_config_file: Path) -> dict:
    """
    Load the data configuration from a YAML file.

    Parameters
    ----------
    data_config_file : Path
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Loaded data configuration as a dictionary.
    """
    import yaml

    with open(data_config_file, "r") as f:
        data_config = yaml.safe_load(f)

        # Get the output class configuration

    return data_config


def _build_snapshot_groups(
    base_dir: Path,
    output_classes: list[dict[str, Any]],
):

    # Get the snapshot file list
    snapshot_files = _get_snapshot_filelist(base_dir)

    # Group snapshot files by output classes
    snapshot_groups = {}
    for output_class in output_classes:
        class_name = output_class["name"]

        # Match snapshot file names with file name patterns from the data
        # configuration
        file_basenames = output_class["file_basenames"]
        for file_basename in file_basenames:
            for snapshot_file in snapshot_files:
                if re.match(file_basename, snapshot_file.name):
                    snapshot_groups.setdefault(class_name, []).append(
                        snapshot_file
                    )

    return snapshot_groups


def _load_snapshots(
    snapshot_files: list[Path],
    columns: set[str],
    output_class: dict[str, Any],
    input_parameters: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    Load a snapshot from a given path.

    Parameters
    ----------
    snapshot_files : list[Path]
        List of paths to the snapshot files.

    columns : list[str]
        List of columns to load from the snapshot.

    Returns
    -------
    pd.DataFrame
        Loaded snapshot as a pandas DataFrame.
    """

    # Create lookup table for input parameters to add to the data frame
    input_parameters_lookup = {
        param["name"]: param for param in input_parameters
    }

    # Load snapshot using uproot
    dfs = []
    for snapshot_file in snapshot_files:
        with uproot.open(snapshot_file) as f:
            # Load the snapshot as data frame
            df_snapshot = f["ntuple"].arrays(columns, library="pd")

            # Add attributes for class and signal/background flag
            df_snapshot["class"] = output_class["name"]
            df_snapshot["is_signal"] = output_class["is_signal"]

            if output_class["is_signal"] and (
                match := re.search(
                    r"(y2b_h2tau|y2tau_h2b)_mx(\d+)_my(\d+)",
                    snapshot_file.name,
                )
            ):
                # If this is a signal class, add the m_x, m_y, and decay_mode
                # columns
                df_snapshot["decay_mode"] = match.group(1)
                df_snapshot["m_x"] = float(match.group(2))
                df_snapshot["m_y"] = float(match.group(3))

            else:
                # If this is a background class, set default values for the
                # parameters
                df_snapshot["decay_mode"] = input_parameters_lookup[
                    "decay_mode"
                ]["fallback_value"]
                df_snapshot["m_x"] = input_parameters_lookup["m_x"][
                    "fallback_value"
                ]
                df_snapshot["m_y"] = input_parameters_lookup["m_y"][
                    "fallback_value"
                ]

            # Append the data frame to the list of data frames
            dfs.append(df_snapshot)

    # Concatenate all data frames into a single data frame
    df = pd.concat(dfs, ignore_index=True)

    return df


def _get_snapshot_filelist(snapshots_dir: Path):
    return list(snapshots_dir.glob("*.root"))


def main(**kwargs):
    # Get the training tag, campaign, channel, and category
    training_tag = kwargs["training_tag"]
    campaign = kwargs["campaign"]
    channel = kwargs["channel"]
    category = kwargs["category"]

    # Get the output directory
    output_dir = (
        kwargs["output_dir"]
        / training_tag
        / campaign
        / f"{channel}__{category}"
    )

    # Load the data configuration
    data_config = _load_data_config(kwargs["data_config_file"])

    # Get the snapshot directory from the config
    base_dir = Path(data_config["data_source"]["base_dir"])

    # Get the input parameters configuration
    input_parameters = data_config["input_parameters"]

    # Build groups of snapshots belonging to the same output class
    snapshot_groups = _build_snapshot_groups(
        base_dir, data_config["output_classes"]
    )

    # Get columns to load from the data configuration. Always load the 'event'
    # column, which is needed for cross-validation splitting.
    base_columns = {c["name"] for c in (data_config["input_variables"])} | {
        "event",
        "__total_weight",
    }

    # Create lookup table for output classes
    output_classes_lookup = {
        output_class["name"]: output_class
        for output_class in data_config["output_classes"]
    }

    # Load snapshots for each output class
    for class_name, snapshot_files in snapshot_groups.items():
        # Load the output class
        output_class = output_classes_lookup[class_name]

        # Add special columns for this output class
        columns = base_columns | set(output_class.get("columns", []))

        # Add special columns
        df = _load_snapshots(
            snapshot_files,
            columns,
            output_class,
            input_parameters,
        )
        print(
            f"Loaded {len(snapshot_files)} snapshots for class {class_name} with {len(df)} events"
        )

        # Create output directory if it does not exist
        if not output_dir.exists():
            output_dir.mkdir(parents=True)

        # Dump data frame to HDF5 output file
        output_file = output_dir / f"{class_name}.h5"
        df.to_hdf(output_file, key="ntuple", mode="w")
        print(f"Dumped data frame to {output_file}")


if __name__ == "__main__":
    kwargs = _parse_arguments()
    main(**kwargs)
