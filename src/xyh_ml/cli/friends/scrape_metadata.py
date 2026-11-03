import logging
from dataclasses import asdict
from pathlib import Path

import hydra
from XRootD.client import FileSystem

from xyh_ml import PROJECT_DIR
from xyh_ml.config.dataclasses import Config, Friend
from xyh_ml.friend_production.scraper import scrape_metadata
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
    xrootd_server = cfg.ntuple.xrootd_server
    base_dir = cfg.ntuple.base_dir
    ntuple_tag = cfg.ntuple.tag
    friends = [Friend(**f) for f in cfg.ntuple.friends]
    samples = cfg.ntuple.samples

    # Get information about what to process
    campaigns = cfg.campaigns

    # Create the XRootD file system
    xrd_fs = FileSystem(xrootd_server)

    # Scrape metadata from the ntuple store
    samples = scrape_metadata(
        xrd_fs=xrd_fs,
        base_dir=base_dir,
        ntuple_tag=ntuple_tag,
        friends=friends,
        campaigns=campaigns,
        samples=samples,
    )

    # Dump the sample metadata to a JSON file
    output_file = Path(cfg.metadata_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    dump_json(output_file, [asdict(s) for s in samples])


if __name__ == "__main__":
    main()
