import logging
from functools import reduce
from operator import or_
from pathlib import Path
from time import time
from typing import Literal

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from xyh_ml.config.dataclasses import OutputClass, Variable
from xyh_ml.constants import TRAIN_TEST_SPLIT_SEED

# Set up logging
logger = logging.getLogger(__name__)


def load_data_frame(
    data_frame_dir: Path,
    decay_mode: str,
    output_classes: list[OutputClass],
    n_folds: int | None = None,
    fold: int | None = None,
    fold_selection: Literal["train", "eval"] | None = None,
    channel: str | None = None,
) -> DataFrame:
    """
    Load the data frame from the preprocessed HDF5 files.

    For signal events, the data frame is filtered for the given `decay_mode`.
    If `fold_selection` is set, the events are assigned to `n_folds` folds
    based on their event number (`event % n_folds`) and filtered for either the
    training folds or the held-out evaluation fold. If `channel` is set, the
    data frame is filtered for that analysis channel. The full data frame is
    loaded otherwise.

    The data frame is shuffled before being returned.

    Parameters
    ----------
    data_frame_dir : Path
        Path to the directory containing the preprocessed HDF5 files.

    decay_mode : str
        Decay mode to filter the signal events for. Must be either `y2b_h2tau`
        or `y2tau_h2b`.

    output_classes : list[OutputClass]
        List of output classes to load the data frames for.

    n_folds : int | None, optional
        Number of folds the events are split into. Required if
        `fold_selection` is set. Default is `None`.

    fold : int | None, optional
        Index of the held-out fold, between 0 and `n_folds - 1`. Required if
        `fold_selection` is set. Default is `None`.

    fold_selection : {"train", "eval"} | None, optional
        If `"train"`, keep the events of all folds except `fold`. If `"eval"`,
        keep only the events of the held-out fold `fold`. If `None`, no
        filtering is applied. Default is `None`.

    channel : str | None, optional
        Analysis channel (`et`, `mt` or `tt`) to filter the events for. If
        `None`, no filtering is applied. Default is `None`.

    Returns
    -------
    DataFrame
        Concatenated data frame containing the events for all output classes.

    Raises
    ------
    ValueError
        - If `decay_mode` is not `y2b_h2tau` or `y2tau_h2b`.
        - If `fold_selection` is not `train`, `eval` or `None`.
        - If `fold_selection` is set and `n_folds` or `fold` are invalid.
        - If `channel` is not `et`, `mt`, `tt` or `None`.
    """

    # Raise exception if decay mode is not `y2b_h2tau` or `y2tau_h2b`
    if decay_mode not in ["y2b_h2tau", "y2tau_h2b"]:
        msg = f"Decay mode must be either 'y2b_h2tau' or 'y2tau_h2b', got {decay_mode}"
        logger.critical(msg)
        raise ValueError(msg)

    # Raise exception if the fold selection is invalid
    if fold_selection not in ["train", "eval", None]:
        msg = f"Fold selection must be 'train', 'eval' or `None`, got {fold_selection}"
        logger.critical(msg)
        raise ValueError(msg)
    if fold_selection is not None and not (
        n_folds is not None
        and fold is not None
        and n_folds >= 2
        and 0 <= fold < n_folds
    ):
        msg = f"Need n_folds >= 2 and 0 <= fold < n_folds, got n_folds={n_folds}, fold={fold}"
        logger.critical(msg)
        raise ValueError(msg)

    # List of data frames which are going to be concatenated
    dfs: list[DataFrame] = []

    # Raise exception if channel is not `et`, `mt`, `tt` or `None`
    if channel not in ["et", "mt", "tt", None]:
        msg = (
            f"Channel must be either 'et', 'mt', 'tt' or `None`, got {channel}"
        )
        logger.critical(msg)
        raise ValueError(msg)

    # Track time for loading the data frames
    start = time()

    for output_class in output_classes:
        # Load the data frame from the preprocessed HDF5 file
        df = pd.read_hdf(
            data_frame_dir / f"{output_class.name}.h5",
            key="ntuple",
        )

        # If this is a signal output class, filter the data frame for the given
        # decay mode
        if output_class.is_signal:
            logger.debug(
                f"Select signal events for decay mode {decay_mode} in data "
                + f" frame for class {output_class.name}"
            )
            df = df.loc[df["decay_mode"] == decay_mode]

        # Filter for the training folds or the held-out fold if it has been set
        # in the function call
        if fold_selection is not None:
            logger.debug(
                f"Select {fold_selection} events for fold {fold} of {n_folds} "
                + f"in data frame for class {output_class.name}"
            )
            in_fold = df["event"] % n_folds == fold
            df = df.loc[in_fold if fold_selection == "eval" else ~in_fold]

        # Filter for analysis channel if it has been set in the function call
        if channel is not None:
            logger.debug(
                f"Filter events for channel {channel} in data frame "
                + f"for class {output_class.name}"
            )
            df = df.loc[df["channel"] == channel]

        dfs.append(df)

        # Log information about the loaded data frame
        logger.info(f"Loaded data frame for output class {output_class.name}")

    # Concatenate the data frames and shuffle the events
    df = pd.concat(dfs, axis=0).sample(frac=1.0).reset_index(drop=True)

    # Stop time tracking
    stop = time()

    # Log information about data frame loader
    n_events = len(df)
    mem = round(df.memory_usage().sum() / (1024**2), 2)
    delta = round(stop - start, 2)
    logger.debug(
        f"Loaded data frame with {n_events} events of size {mem} MB in "
        + f"{delta} s"
    )

    return df


def train_val_split(
    df: DataFrame,
    train_frac: float,
) -> dict[str, DataFrame]:
    """
    Split the full dataset into a training and validation set.

    Parameters
    ----------
    df : DataFrame
        Full dataset to split into training and validation sets.

    train_frac : float
        Fraction of the full dataset to use for training. Must be between 0 and
        1.

    Returns
    -------
    dict[str, DataFrame]
        Dictionary containing the training and validation data frames with
        keys "train" and "val", respectively.

    Raises
    ------
    ValueError
        If `train_frac` is not between 0 and 1.
    """

    # Raise an exception if training fraction is not between 0 and 1
    if train_frac < 0 or train_frac > 1:
        msg = f"Training fraction must be between 0 and 1, got {train_frac}"
        logger.critical(msg)
        raise ValueError(msg)

    # Use corresponding sklearn function to split the data frame into a
    # training and validation data frame
    df_train, df_val = train_test_split(
        df,
        train_size=train_frac,
        shuffle=True,
        random_state=TRAIN_TEST_SPLIT_SEED,
    )
    logger.debug(
        "Split into training and validation data frames with "
        + f"{len(df_train)} and {len(df_val)} events, respectively"
    )

    return {
        "train": df_train,
        "val": df_val,
    }


def get_unique_input_parameter_combinations(
    df: DataFrame,
    parameter_variables: list[Variable],
) -> DataFrame:
    """
    Get unique combinations of input parameter values from the data frame.

    Parameters
    ----------

    df : DataFrame
        Data frame containing the input parameter values.

    parameter_variables : list[Variable]
        List of input parameter variables to get unique values for.

    Returns
    -------
    DataFrame
        Data frame containing unique combinations of input parameter values.
    """

    # Create list of parameter names
    parameter_names = [p.name for p in parameter_variables]

    logger.debug(
        "Getting unique combinations of input parameter values for parameters "
        + f"{parameter_names}"
    )

    # Get unique values of the parameters in the data frame
    unique_parameters = df[parameter_names].drop_duplicates()

    # Remove values uniquely composed of fallback values
    mask = reduce(
        or_,
        [
            unique_parameters.iloc[:, i] != p.fallback_value
            for i, p in enumerate(parameter_variables)
        ],
    )
    unique_parameters = unique_parameters.loc[mask].reset_index(drop=True)
    logger.debug(
        f"Found {len(unique_parameters)} unique combinations of input parameter values"
    )

    return unique_parameters


def prepare_parameter_values(
    data_frame: pd.DataFrame,
    parameter_values: pd.DataFrame,
):
    """
    Prepare the data frame for evaluation with multiple parameter values.

    All rows in the original data frame are replicated `len(parameter_values)`
    times and unique parameter value combinations are added to the data frame.

    Parameters
    ----------
    data_frame : pd.DataFrame
        Data frame to prepare for evaluation with multiple parameter values.

    parameter_values : pd.DataFrame
        Data frame containing the unique parameter value combinations to add
        to the data frame.
    """
    # Get names of the parameters
    parameter_columns = parameter_values.columns

    # Create a set with all unique parameter value combinations for each event.
    # The output of this operation is an array with shape (n_events *
    # n_unique_parameters, n_parameters).
    n_events = data_frame.shape[0]
    n_parameter_values = parameter_values.shape[0]
    n_parameters = parameter_values.shape[1]
    parameter_values_reshaped = np.reshape(
        parameter_values.to_numpy(),
        (1, n_parameter_values, n_parameters),
    )
    parameter_values_repeated = np.reshape(
        np.repeat(parameter_values_reshaped, n_events, axis=0),
        (n_events * n_parameter_values, n_parameters),
    )
    parameter_values_index = np.reshape(
        np.repeat(
            np.reshape(
                np.arange(n_parameter_values),
                (1, n_parameter_values),
                axis=0,
            ),
            n_events,
        ),
        (n_events * n_parameter_values,),
    )

    # Repeat events as often as there are unique parameter combinations. The
    # data frame will have n_events * n_unique_parameters rows after this
    # operation. Append a new index level "parameter_value" to be able to
    # track the order of events.
    data_frame = data_frame.loc[data_frame.index.repeat(n_parameter_values)]
    data_frame = data_frame.assign(
        "parameter_value",
        parameter_values_index,
    ).set_index("parameter_value", append=True)

    # Add the unique parameter value combinations to the background events data
    # frame
    data_frame[parameter_columns] = parameter_values_repeated

    return data_frame
