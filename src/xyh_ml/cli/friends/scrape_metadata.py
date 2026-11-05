"""Scrape the metadata of a list of samples, one output file per sample.

The metadata of a sample is written to
``<metadata_dir>/<era>/<sample>.json.gz``. Samples whose file exists are
skipped, so an interrupted or partly failed run can be continued by repeating
the command. No file is written for a sample with missing or incomplete
channels or friends. Only the directories of the selected eras, samples and
channels are listed.

Example (from the repo root; further arguments are Hydra overrides)::

    pixi run python -m xyh_ml.cli.friends.scrape_metadata \\
        --samples-file ../KingMaker/sample_lists/samples_2024_signal.txt \\
        --campaign 2024_nano_v15 -j 4

Use ``--overwrite`` after changing ``ntuple.friends`` or if samples got new
files, as skipping only checks that the metadata file exists.
"""

import argparse
import logging
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from XRootD.client import FileSystem

from xyh_ml import PROJECT_DIR
from xyh_ml.config.dataclasses import Friend
from xyh_ml.friend_production.metadata import read_samples_file
from xyh_ml.friend_production.paths import project_path, select_campaigns
from xyh_ml.friend_production.scraper import pending_samples, scrape_samples

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="Further arguments are Hydra overrides of the "
        + "scrape_metadata config.",
    )
    parser.add_argument(
        "--samples-file",
        action="append",
        required=True,
        type=Path,
        help="Text file with one sample name or DAS dataset per line "
        + "(repeatable), e.g. a KingMaker sample list.",
    )
    parser.add_argument(
        "--campaign",
        action="append",
        default=None,
        help="Only this campaign (repeatable; default: all).",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=4,
        help="Number of samples scraped at the same time (default: 4).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Scrape samples again whose metadata file exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list the samples to scrape, without accessing the server.",
    )
    parser.add_argument("overrides", nargs="*", help="Hydra overrides.")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be at least 1.")
    return args


def _compose(overrides: list[str]) -> DictConfig:
    with initialize_config_dir(
        str(PROJECT_DIR / "config"), version_base="1.3"
    ):
        return compose("scrape_metadata", overrides=overrides)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    )
    args = _parse_args(argv)
    cfg = _compose(args.overrides)

    names = [n for f in args.samples_file for n in read_samples_file(f)]
    metadata_dir = project_path(cfg.metadata_dir)
    channels = [channel.name for channel in cfg.channels]
    try:
        campaigns = select_campaigns(cfg.campaigns, args.campaign)
    except ValueError:
        return 2

    if args.dry_run:
        pending, existing = pending_samples(
            metadata_dir, campaigns, names, args.overwrite
        )
        for campaign, sample in pending:
            print(f"{campaign.era} {sample}")
        logger.info(
            f"{len(pending)} samples to scrape, {len(existing)} with "
            + f"existing metadata files in {metadata_dir}"
        )
        return 0

    result = scrape_samples(
        FileSystem(cfg.ntuple.xrootd_server),
        cfg.ntuple.base_dir,
        cfg.ntuple.tag,
        [Friend(**friend) for friend in cfg.ntuple.friends],
        campaigns,
        channels,
        names,
        metadata_dir,
        n_parallel=args.jobs,
        overwrite=args.overwrite,
    )

    logger.info(
        f"Metadata in {metadata_dir}: {len(result.written)} written, "
        + f"{len(result.skipped)} skipped (exist), "
        + f"{len(result.incomplete)} incomplete, "
        + f"{len(result.failed)} failed, "
        + f"{len(result.not_found)} not found"
    )
    if result.not_found:
        logger.warning(
            f"{len(result.not_found)} samples not found in any selected "
            + "era:\n  "
            + "\n  ".join(result.not_found)
        )
    if result.incomplete:
        logger.warning(
            f"{len(result.incomplete)} incomplete samples:\n  "
            + "\n  ".join(
                f"{era} {sample}: {reason}"
                for era, sample, reason in sorted(result.incomplete)
            )
        )
    if result.failed:
        logger.error(
            f"{len(result.failed)} samples failed:\n  "
            + "\n  ".join(f"{e} {s}" for e, s in sorted(result.failed))
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
