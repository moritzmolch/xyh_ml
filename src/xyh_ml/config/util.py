"""
Utility functions for performing operations on configuration objects.
"""

from collections.abc import Generator
from itertools import product
from pathlib import Path

from xyh_ml.config.dataclasses import DataSource


def yield_base_dirs(
    data_source: DataSource,
) -> Generator[tuple[tuple[str, str], Path], None, None]:
    """
    Yield all base directories for a given data source.

    The base directories are constructed by iterating over all combinations of
    campaigns and channels defined in the `data_source` object. These
    parameters replace the placeholders `{channel}` and `{campaign}` in the
    `base_dir` string of the `data_source`.

    Parameters
    ----------
    data_source : DataSource
        The data source configuration containing the base directory template.

    Yields
    ------
    tuple[tuple[str, str], Path]
        A tuple containing:
        - A tuple of `(campaign, channel)` representing the current
          combination.
        - A concrete `Path` object representing the constructed base directory
          for the current combination of campaign and channel.
    """

    for campaign, channel in product(
        data_source.campaigns, data_source.channels
    ):
        yield (
            (campaign, channel),
            Path(
                data_source.base_dir.format(campaign=campaign, channel=channel)
            ),
        )
