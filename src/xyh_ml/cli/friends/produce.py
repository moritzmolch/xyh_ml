import logging
from pathlib import Path

import hydra
import uproot

from xyh_ml import PROJECT_DIR
from xyh_ml.config.dataclasses import Config
from xyh_ml.friend_production.dataclasses import Sample
from xyh_ml.io import load_json

# Set up logging
logger = logging.getLogger(__name__)


def determine_chunks(
    sample: Sample,
    chunk_size: int = 1000,
):
    # Get number of events in each file from the main files
    files = []
    empty_files = []
    for i, main_file in enumerate(sample.main_files):
        empty = False
        with uproot.open(main_file) as f:
            if "ntuple" not in f:
                logger.debug(
                    f"File {main_file} does not contain 'ntuple' tree. Skipping."
                )
                empty = True
            elif f["ntuple"].num_entries == 0:
                logger.debug(
                    f"File {main_file} contains 'ntuple' tree but has no entries. Skipping."
                )
                empty = True

            # If we reach here, we have a valid ntuple tree with entries. Get the
            # number of events.
            num_events = f["ntuple"].num_entries
            empty = empty or num_events == 0

            # Add empty files to the corresponding list
            if empty:
                empty_files.append(main_file)
                continue
            else:
                files.append(
                    {
                        "main_file": main_file,
                        "friend_files": {
                            sample.friend_files[f.name][i]
                            for f in sample.friends
                        },
                        "num_events": num_events,
                    }
                )

    # Get the total number of events in the sample and form chunk indices
    total_events = sum(f["num_events"] for f in files)
    chunk_indices = list(range(0, total_events, chunk_size)) + [total_events]

    # Determine files that belong to each chunk based on the number of events in each file
    chunks = []


@hydra.main(
    config_path=str(PROJECT_DIR / "config"),
    config_name="scrape_metadata",
    version_base="1.3",
)
def main(cfg: Config):
    # Get sample metadata
    metadata_file = Path(cfg.metadata_file)
    samples = [Sample(**s) for s in load_json(metadata_file)]


if __name__ == "__main__":
    main()
