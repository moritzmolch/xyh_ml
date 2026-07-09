from pathlib import Path
import pandas as pd
import logging
from time import time

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


THIS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = THIS_DIR.parent.parent.parent / "test_output"

EVENT_PARITY = 0

INPUT_DIR = Path("/ceph/mmolch/xyh-bbtautau-crown/bbtautau/data/output/TrainingDataFrames/shapes-2026-06-10/2024/mt_base_sr")

CATEGORIES = [
    {
        "name": "xyh",
        "file_basenames": [
            # "2024__mt__mt_base_sr__xyh_y2b_h2tau__Nominal.h5",
            "2024__mt__mt_base_sr__xyh_y2tau_h2b__Nominal.h5",
        ],
        "is_signal": True,
    },
    {
        "name": "t",
        "label": "Single t",
        "file_basenames": [
            "2024__mt__mt_base_sr__single_t__Nominal.h5",
        ],
        "is_signal": False,
    },
    {
        "name": "tt",
        "label": "\\text{t}$\\bar{\\text{t}}$",
        "file_basenames": [
            "2024__mt__mt_base_sr__tt_rem__Nominal.h5",
            "2024__mt__mt_base_sr__tt_tautau__Nominal.h5",
        ],
        "is_signal": False,
    },
    {
        "name": "z",
        "label": "Z($\\tau\\tau$)",
        "file_basenames": [
            "2024__mt__mt_base_sr__dy_tautau__Nominal.h5",
        ],
        "is_signal": False,
    },
    {
        "name": "jetfakes",
        "label": "$\\text{j} \\to \\tau_{\\text{h}}$",
        "file_basenames":
        [
            "2024__mt__mt_base_sr__jetfakes__tau_antiid_vs_jet.h5",
        ],
        "is_signal": False,
    },
    # {
    #     "name": "single_h",
    #     "label": "Single H",
    #     "file_basenames": [
    #         "2024__mt__mt_base_sr__single_h__Nominal.h5",
    #     ],
    #     "is_signal": False,
    # },
    # {
    #     "name": "diboson",
    #     "label": "Diboson",
    #     "file_basenames": [
    #         "2024__mt__mt_base_sr__vv__Nominal.h5",
    #     ],
    #     "is_signal": False,
    # },
]


def _prepare_output_dir(output_dir: Path):
    # Create the output directory if it does not exist
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        logger.debug(f"Created output directory {output_dir}")


def _prepare_input_files(input_dir: Path, categories: dict):
    # Create dictionary of input file lists, where keys correspond to the
    # output categories of the network
    input_files = {}

    for category in categories:
        for basename in category["file_basenames"]:
            # Get the category name
            category_name = category["name"]

            # Construct the absolute input file path
            input_file = input_dir / basename

            # Check if the input file exists
            if not input_file.exists():
                msg = f"Could not find input file {input_file}"
                logger.critical(msg)
                raise FileNotFoundError(msg)

            # Add input file to dictionary
            input_files.setdefault(category_name, []).append(input_file)
            logger.debug(
                f"Added {input_file} to output category {category_name}",
            )

    return input_files


def _load_data_frame(input_files: dict, categories: dict):

    # List of all data frames that are loaded. The data frames are concatenated
    # to a single data frame later.
    dfs = []

    # Start time tracking
    start = time()

    for category_name, input_file_list in input_files.items():

        # Load data frame from each file separately and add it to the list of
        # all data frames. The category name is added as column.
        for input_file in input_file_list:
            df = pd.read_hdf(input_file)
            df["category"] = category_name
            dfs.append(df)
            logger.debug(f"Loaded data frame from {input_file}")

    # Concatenate all data frames. Sample with 100% fraction to shuffle the
    # dataset
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


def _filter_event_parity(df: pd.DataFrame, event_parity: int):
    # Raise exception if parity is not 0 or 1
    if event_parity not in [0, 1]:
        msg = f"Event parity must be either 0 or 1, got {event_parity}"
        logger.critical(msg)
        raise ValueError(msg)

    # Filter events for given event parity
    n_events_before = len(df)
    df = df.loc[df["event"] % 2 == event_parity]

    # Log reduction of the data frame
    n_events_after = len(df)
    frac = round(n_events_after / n_events_before * 100, 2)
    logger.debug(
        f"Filtered data frame for events with parity {event_parity}, resulting "
        + f"in {n_events_after} selected events (efficiency: {frac} %)"
    )

    return df


def main():
    # Prepare the output directory
    _prepare_output_dir(OUTPUT_DIR)

    # Load the input files
    input_files = _prepare_input_files(INPUT_DIR, CATEGORIES)

    # Load the input files into a data frame
    data_frame = _load_data_frame(input_files, CATEGORIES)

    # Filter events according to the event parity
    data_frame = _filter_event_parity(data_frame, EVENT_PARITY)


if __name__ == "__main__":
    main()
