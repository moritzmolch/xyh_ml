import logging
import os
import re

import uproot
from XRootD.client import FileSystem

from xyh_ml.config.dataclasses import Campaign, Friend
from xyh_ml.friend_production.dataclasses import Sample, Shift, Variable
from xyh_ml.friend_production.xrootd import (
    get_channel_dirs,
    get_era_dirs,
    get_files,
    get_friend_files,
    get_sample_dirs,
)

# Set up logging
logger = logging.getLogger(__name__)


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
        for friend_name, friend_file_list in friend_files.items():
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

        # Break the loop after the first successfully opened files
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
        m = re.match(r"(?P<variable>.+)__?(?P<shift>(.+)(Up|Down))$", key)
        if not m:
            msg = (
                f"Column name '{key}' does not match the expected pattern for "
                + "a shifted variable."
            )
            logger.error(msg)
            raise ValueError(msg)

        # Extract the variable and shift from the match
        variable = m.group("variable")
        shift = m.group("shift")

        # Add the shift to the map for the corresponding variable
        shift_map.setdefault(shift, []).append(
            Variable(
                variable=variable,
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


def scrape_metadata(
    xrd_fs: FileSystem,
    base_dir: str,
    ntuple_tag: str,
    friends: list[Friend],
    campaigns: list[Campaign] = None,
    samples: list[str] = None,
):
    # Create a lookup for campaigns by era
    eras_lookup = {campaign.era: campaign for campaign in campaigns}

    # Container for sample specs
    samples = []

    # Iterate through the eras, samples, and channels to collect sample
    # metadata
    for era, era_dir in get_era_dirs(
        xrd_fs, os.path.join(base_dir, ntuple_tag)
    ):
        # Get the campaign
        campaign = eras_lookup[era]

        # Check if this campaign needs to be skipped
        if campaigns and campaign not in campaigns:
            # Skip era if not in the list of campaigns
            logger.info(
                f"Skipping era {era} as it is not in the list of campaigns to process."
            )
            continue

        for sample, sample_dir in get_sample_dirs(xrd_fs, era_dir):
            # Skip sample if not in the list of samples
            if samples and sample not in samples:
                logger.info(
                    f"Skipping sample {sample} as it is not in the list of samples to process."
                )
                continue

            for channel, channel_dir in get_channel_dirs(xrd_fs, sample_dir):
                # Get the main filelist for this sample
                main_files = get_files(xrd_fs, channel_dir)

                # Get the corresponding friend filelists for this sample
                # (if any exist)
                friend_files = {}
                for friend in friends:
                    # Check if the channel is skipped
                    if (
                        friend.include_channels
                        and channel not in friend.include_channels
                    ) or (
                        friend.exclude_channels
                        and channel in friend.exclude_channels
                    ):
                        logger.info(
                            f"Skipping friend {friend.name} for channel {channel}."
                        )
                        continue

                    # Check if campaign is skipped
                    if (
                        friend.include_campaigns
                        and campaign not in friend.include_campaigns
                    ) or (
                        friend.exclude_campaigns
                        and campaign in friend.exclude_campaigns
                    ):
                        logger.info(
                            f"Skipping friend {friend.name} for campaign {campaign}."
                        )
                        continue

                    # Check if sample is skipped
                    if (
                        friend.include_samples
                        and not any(
                            re.search(pattern, sample)
                            for pattern in friend.exclude_samples
                        )
                    ) or (
                        friend.exclude_samples
                        and any(
                            re.search(pattern, sample)
                            for pattern in friend.exclude_samples
                        )
                    ):
                        logger.info(
                            f"Skipping friend {friend.name} for sample {sample}."
                        )
                        continue

                    # Get the list of friend files for this sample
                    friend_files[friend.name] = get_friend_files(
                        xrd_fs,
                        base_dir,
                        friend.name,
                        era,
                        sample,
                        channel,
                    )

                # Get the shifts for this sample
                shifts = get_shifts(main_files, friend_files)

                # Create the sample object
                sample = Sample(
                    era=era,
                    sample=sample,
                    channel=channel,
                    main_files=main_files,
                    friend_files=friend_files,
                    shifts=shifts,
                )

                # Append the sample to the list of samples
                samples.append(sample)

    return samples
