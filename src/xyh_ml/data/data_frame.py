import logging
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from xyh_ml.config.data import DataConfig
from xyh_ml.constants import DEFAULT_SEED

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# -----------------------------------------------------------------------------
# Dataset loading and splitting
# -----------------------------------------------------------------------------


def load_data_frame_from_config(data_config: DataConfig) -> pd.DataFrame:
    # Get the base output directory
    output_dir = data_config.data_source.base_dir

    # For each output class, concatenate the data frames from the specified
    # input files
    data_frames = []
    for output_class in data_config.output_classes:
        for basenames in output_class.file_basenames:
            # Load the data frame corresponding to the base name
            df = pd.read_hdf(
                os.path.join(output_dir, f"{basenames}.h5"),
                key="ntuple",
            )

            # Add attributes for class and signal/background flag
            df["class"] = output_class.name
            df["is_signal"] = output_class.is_signal

            # Append data frame to list of data frames
            data_frames.append(df)

    return pd.concat(data_frames, ignore_index=True)


def _split_even_odd(
    df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Split data frame into two data frames according to whether the event number
    is even or odd.
    """
    return {
        "even": df[df["event"] % 2 == 0],
        "odd": df[df["event"] % 2 == 1],
    }


def _split_train_val(
    df: pd.DataFrame,
    train_frac: float,
    seed: int = DEFAULT_SEED,
) -> dict[str, pd.DataFrame]:
    """
    Split the data frame into training, validation, and test samples according
    to the specified fractions.
    """

    if train_frac < 0.0 or train_frac > 1.0:
        raise ValueError(
            "Fraction of training events must be between 0 and 1."
        )

    # Split the data set into training, validation, and test samples
    df_train, df_val = train_test_split(
        df,
        train_size=train_frac,
        random_state=seed,
        shuffle=True,
    )

    return {
        "train": df_train,
        "val": df_val,
    }


def split_data(
    df: pd.DataFrame,
    train_frac: float,
    seed: int = DEFAULT_SEED,
) -> dict[str, pd.DataFrame]:
    """
    Split the full data frame into samples with even and odd event number
    parity, and then split each of these samples into training and validation
    samples.

    The output is a dictionary with keys of the form `(even_odd_key,
    train_val_key)`, with `even_odd_key` being either "even" or "odd", and
    `train_val_key` being either "train" or "val". The values are the
    corresponding data frames for each split.

    Parameters
    ----------
    df : pd.DataFrame
        The full data frame to be split.

    train_frac : float
        The fraction of the data to be used for training. The rest will be used
        for validation). Must be between 0 and 1.

    seed : int (optional)
        The random seed to be used for the splitting. The default seed is set
        by `DEFAULT_SEED` in `xyh_ml.constants`.

    Returns
    -------
    dict[tuple[str, str], pd.DataFrame]
        With the splitted subsets for training on even or odd events.
    """
    return {
        (even_odd_key, train_val_test_key): df
        for even_odd_key, dfs_even_odd in _split_even_odd(df).items()
        for train_val_test_key, df in _split_train_val(
            dfs_even_odd,
            train_frac,
            seed=seed,
        ).items()
    }
