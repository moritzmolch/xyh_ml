import logging
from dataclasses import asdict
from pathlib import Path

import hydra
from XRootD.client import FileSystem

from xyh_ml import PROJECT_DIR
from xyh_ml.config import (
    Config,
)
from xyh_ml.friend_production.metadata_scraper import scrape_metadata
from xyh_ml.io import dump_json

# Set up logging
logger = logging.getLogger(__name__)


@hydra.main(
    config_path=str(PROJECT_DIR / "config"),
    config_name="scrape_metadata",
    version_base="1.3",
)
def main(cfg: Config):
    # Get ntuple store metadata
    xrootd_server = cfg.ntuples.xrootd_server
    base_dir = cfg.ntuples.base_dir
    ntuple_tag = cfg.ntuples.tag
    friends = cfg.ntuples.friends

    # Get information about what to process
    campaigns = cfg.ntuples.campaigns

    # Create the XRootD file system
    xrd_fs = FileSystem(xrootd_server)

    # Scrape metadata from the ntuple store
    samples = scrape_metadata(
        xrd_fs=xrd_fs,
        base_dir=base_dir,
        ntuple_tag=ntuple_tag,
        friends=friends,
        campaigns=campaigns,
        samples=cfg.ntuples.samples,
    )

    # Dump the sample metadata to a JSON file
    output_file = Path(cfg.metadata_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    dump_json(asdict(samples), output_file)


if __name__ == "__main__":
    main()
