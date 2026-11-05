import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import uproot
from XRootD.client import FileSystem

from xyh_ml.config.dataclasses import Campaign, Friend
from xyh_ml.friend_production.dataclasses import Sample, Shift, Variable
from xyh_ml.friend_production.metadata import (
    metadata_path,
    write_sample_metadata,
)
from xyh_ml.friend_production.xrootd import (
    MAIN_NTUPLE_DIR_NAME,
    DirectoryNotFoundError,
    get_files,
    get_friend_files,
)

# Set up logging
logger = logging.getLogger(__name__)


class SampleNotFoundError(RuntimeError):
    """The sample has no selected channel in this era."""


class IncompleteSampleError(RuntimeError):
    """Channels or friends of the sample are missing or incomplete."""


def get_shifts(
    main_files: list[str],
    friend_files: dict[str, list[str]],
) -> list[Shift]:
    ntuple_keys = []
    for i, main_file in enumerate(main_files):
        with uproot.open(main_file) as f:
            # Check if the ntuple tree exists and if it has entries
            if "ntuple" not in f:
                logger.debug(
                    f"File {main_file} does not contain 'ntuple' tree. Skipping."
                )
                continue
            elif f["ntuple"].num_entries == 0:
                logger.debug(
                    f"File {main_file} contains 'ntuple' tree but has no entries. Skipping."
                )
                continue

            # If we reach here, we have a valid ntuple tree with entries. Store
            # keys.
            ntuple_keys = f["ntuple"].keys()

        # Get the corresponding friends and extract their keys as well
        for friend_file_list in friend_files.values():
            friend_file = friend_file_list[i]
            with uproot.open(friend_file) as f:
                # Check if the friend ntuple tree exists and if it has entries
                if "ntuple" not in f:
                    logger.debug(
                        f"Friend file {friend_file} does not contain 'ntuple' tree. Skipping."
                    )
                    continue
                elif f["ntuple"].num_entries == 0:
                    logger.debug(
                        f"Friend file {friend_file} contains 'ntuple' tree but has no entries. Skipping."
                    )
                    continue

                # If we reach here, we have a valid friend ntuple tree with
                # entries. Store keys.
                friend_keys = f["ntuple"].keys()

            # Send a warning if the friend file keys overlap with already
            # extracted keys
            if intersection := set(friend_keys).intersection(set(ntuple_keys)):
                logger.warning(
                    f"Friend file {friend_file} contains keys that overlap "
                    + f" with already extracted keys: {intersection}"
                )

            # Add the friend keys to the ntuple keys
            ntuple_keys.extend(friend_keys)

        # Break the loop after the first successfully opened file + friends
        break

    shift_map = {}
    for key in ntuple_keys:
        # Check if "__" appears at most once
        if key.count("__") > 1:
            logger.warning(
                f"Column name '{key}' contains more than one '__'. This "
                + "might obfuscate the evaluation of systematic uncertainties."
            )

        # Check if the key matches the pattern for a shifted variable
        m = re.match(r"(?P<variable>.+?)(__(?P<shift>.+(Up|Down)))?$", key)
        if not m:
            msg = (
                f"Column name '{key}' does not match the expected pattern for "
                + "a shifted variable."
            )
            logger.error(msg)
            raise ValueError(msg)

        # Extract the variable and shift from the match
        variable = m.group("variable")
        shift = m.group("shift") or "nominal"

        # Add the shift to the map for the corresponding variable
        shift_map.setdefault(shift, []).append(
            Variable(
                name=variable,
                column=key,
            ),
        )

    # Merge shift maps
    shifts = [
        Shift(
            name=shift,
            variables=variables,
        )
        for shift, variables in shift_map.items()
    ]

    return shifts


def friend_applies(
    friend: Friend,
    campaign_name: str,
    channel: str,
    sample: str,
) -> bool:
    """Whether ``friend`` is used for this campaign, channel and sample."""
    if (
        friend.include_channels and channel not in friend.include_channels
    ) or (friend.exclude_channels and channel in friend.exclude_channels):
        return False

    if (
        friend.include_campaigns
        and campaign_name not in friend.include_campaigns
    ) or (
        friend.exclude_campaigns and campaign_name in friend.exclude_campaigns
    ):
        return False

    if friend.include_samples and not any(
        re.search(pattern, sample) for pattern in friend.include_samples
    ):
        return False

    return not (
        friend.exclude_samples
        and any(
            re.search(pattern, sample) for pattern in friend.exclude_samples
        )
    )


def _friend_files(
    xrd_fs: FileSystem,
    base_dir: str,
    ntuple_tag: str,
    friends: list[Friend],
    campaign: Campaign,
    sample: str,
    channel: str,
    main_files: list[str],
) -> tuple[dict[str, list[str]], list[str]]:
    """Files of the friends that apply to this channel of the sample, and the
    problems found with them."""
    friend_files = {}
    problems = []
    for friend in friends:
        if not friend_applies(friend, campaign.name, channel, sample):
            logger.debug(
                f"Skipping friend {friend.name} for sample {sample} "
                + f"({campaign.era}, {channel})."
            )
            continue

        try:
            friend_files[friend.name] = get_friend_files(
                xrd_fs,
                base_dir,
                ntuple_tag,
                friend.name,
                campaign.era,
                sample,
                channel,
            )
        except DirectoryNotFoundError:
            problems.append(f"{friend.name} is missing")
            continue

        # Friend files are matched to the main files by their position, so
        # both need the same file names
        if _file_names(friend_files[friend.name]) != _file_names(main_files):
            problems.append(
                f"{friend.name} has {len(friend_files[friend.name])} files "
                + f"with names different from the {len(main_files)} main "
                + "files"
            )
    return friend_files, problems


def scrape_sample(
    xrd_fs: FileSystem,
    base_dir: str,
    ntuple_tag: str,
    friends: list[Friend],
    campaign: Campaign,
    sample: str,
    channels: list[str],
) -> list[Sample]:
    """Metadata of ``sample`` in the era of ``campaign``, one record per
    channel.

    Only the directories of the selected channels and their friends are
    listed. The paths are built from the arguments, so the era and sample
    directories are not listed to discover their content.

    Raises :class:`SampleNotFoundError` if no channel exists, and
    :class:`IncompleteSampleError` if a channel or a friend is missing or
    incomplete. A sample is complete or not scraped at all, as friend columns
    would be missing or misaligned otherwise, and a later run has to find the
    missing channels.
    """
    sample_dir = os.path.join(
        base_dir, ntuple_tag, MAIN_NTUPLE_DIR_NAME, campaign.era, sample
    )

    main_files = {}
    problems = []
    for channel in channels:
        try:
            main_files[channel] = get_files(
                xrd_fs, os.path.join(sample_dir, channel)
            )
        except DirectoryNotFoundError:
            problems.append(f"channel {channel} is missing")
            continue
        if not main_files[channel]:
            problems.append(f"channel {channel} has no files")

    if not main_files:
        msg = f"Sample {sample} has no channel {channels} in {campaign.era}."
        raise SampleNotFoundError(msg)

    friend_files = {}
    for channel, files in main_files.items():
        friend_files[channel], friend_problems = _friend_files(
            xrd_fs,
            base_dir,
            ntuple_tag,
            friends,
            campaign,
            sample,
            channel,
            files,
        )
        problems.extend(f"{channel}: {p}" for p in friend_problems)

    if problems:
        raise IncompleteSampleError("; ".join(problems))

    return [
        Sample(
            era=campaign.era,
            sample=sample,
            channel=channel,
            main_files=files,
            friend_files=friend_files[channel],
            shifts=get_shifts(files, friend_files[channel]),
        )
        for channel, files in main_files.items()
    ]


@dataclass
class ScrapeResult:
    """Outcome of :func:`scrape_samples` as ``(era, sample)`` pairs."""

    written: list[tuple[str, str]] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)
    # With the reason
    incomplete: list[tuple[str, str, str]] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    # Names of the samples not found in any of the selected eras
    not_found: list[str] = field(default_factory=list)


def pending_samples(
    metadata_dir: Path,
    campaigns: list[Campaign],
    sample_names: list[str],
    overwrite: bool = False,
) -> tuple[list[tuple[Campaign, str]], list[tuple[Campaign, str]]]:
    """Split all selected pairs of campaign and sample into those to scrape
    and those whose metadata file exists. Does not access the XRootD server.
    """
    pending = []
    existing = []
    for campaign in campaigns:
        for sample in dict.fromkeys(sample_names):
            exists = metadata_path(metadata_dir, campaign.era, sample).exists()
            if exists and not overwrite:
                existing.append((campaign, sample))
            else:
                pending.append((campaign, sample))
    return pending, existing


def scrape_samples(
    xrd_fs: FileSystem,
    base_dir: str,
    ntuple_tag: str,
    friends: list[Friend],
    campaigns: list[Campaign],
    channels: list[str],
    sample_names: list[str],
    metadata_dir: Path,
    n_parallel: int = 4,
    overwrite: bool = False,
) -> ScrapeResult:
    """Write one metadata file per sample in ``sample_names`` and era of
    ``campaigns``, skipping existing files unless ``overwrite`` is set.

    Samples are scraped in ``n_parallel`` threads. A sample that fails does
    not stop the others, and a later run retries it, as no file was written.
    """
    if n_parallel < 1:
        raise ValueError(f"n_parallel must be >= 1, got {n_parallel}.")

    pending, existing = pending_samples(
        metadata_dir, campaigns, sample_names, overwrite
    )
    result = ScrapeResult(skipped=[(c.era, s) for c, s in existing])
    logger.info(
        f"Scraping {len(pending)} samples, skipping {len(existing)} with "
        + "existing metadata files"
    )

    def scrape_and_write(campaign: Campaign, sample: str) -> None:
        records = scrape_sample(
            xrd_fs,
            base_dir,
            ntuple_tag,
            friends,
            campaign,
            sample,
            channels,
        )
        write_sample_metadata(metadata_dir, campaign.era, sample, records)

    found = {sample for _, sample in existing}
    with ThreadPoolExecutor(max_workers=n_parallel) as executor:
        futures = {
            executor.submit(scrape_and_write, campaign, sample): (
                campaign.era,
                sample,
            )
            for campaign, sample in pending
        }
        try:
            for future in as_completed(futures):
                era, sample = futures[future]
                try:
                    future.result()
                except SampleNotFoundError as e:
                    logger.debug(e)
                    continue
                except IncompleteSampleError as e:
                    logger.warning(f"Incomplete sample {sample} ({era}): {e}")
                    result.incomplete.append((era, sample, str(e)))
                except Exception:
                    logger.exception(f"Failed to scrape {sample} ({era})")
                    result.failed.append((era, sample))
                else:
                    result.written.append((era, sample))
                found.add(sample)
        except KeyboardInterrupt:
            logger.warning("Interrupted, not starting further samples")
            executor.shutdown(wait=False, cancel_futures=True)
            raise

    result.not_found = sorted(set(sample_names).difference(found))
    return result


def _file_names(files: list[str]) -> list[str]:
    """Sorted file names of ``files`` without their directories."""
    return sorted(f.rsplit("/", 1)[-1] for f in files)
