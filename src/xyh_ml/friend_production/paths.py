"""Campaigns and output paths of the friend production.

This module does not import torch, so that planning jobs stays cheap.
"""

import logging
import os
import re
from pathlib import Path

from omegaconf import DictConfig
from XRootD.client import FileSystem

from xyh_ml import PROJECT_DIR
from xyh_ml.config.dataclasses import Campaign
from xyh_ml.friend_production.dataclasses import Sample
from xyh_ml.friend_production.xrootd import (
    FRIEND_NTUPLE_DIR_NAME,
    DirectoryNotFoundError,
    is_xrootd_url,
    split_xrootd_url,
    xrd_exists,
    xrd_ls,
    xrootd_url,
)

logger = logging.getLogger(__name__)


def project_path(path: str | Path) -> Path:
    """``path`` relative to the repo root, where the jobs run."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_DIR / path


def campaign_name(campaigns: list[Campaign], era: str) -> str:
    """Name of the campaign of ``era``, as used for the ``campaign`` input."""
    for campaign in campaigns:
        if campaign.era == era:
            return campaign.name
    msg = f"No campaign defined for era {era}."
    logger.error(msg)
    raise ValueError(msg)


def select_campaigns(
    campaigns: list[Campaign],
    names: list[str] | None,
) -> list[Campaign]:
    """The campaigns called ``names``, or all with ``names=None``."""
    if names is None:
        return list(campaigns)
    by_name = {campaign.name: campaign for campaign in campaigns}
    unknown = sorted(set(names).difference(by_name))
    if unknown:
        msg = (
            f"Unknown campaigns {unknown}. Defined campaigns: {list(by_name)}"
        )
        logger.error(msg)
        raise ValueError(msg)
    return [by_name[name] for name in dict.fromkeys(names)]


def default_output_file(output_dir: Path | str, sample: Sample) -> Path | str:
    """Output path mirroring the CROWN friend layout,
    ``<output_dir>/<era>/<sample>/<channel>/<file name of the main file>``.

    ``output_dir`` is a local directory or an XRootD URL (see
    :func:`resolve_output_dir`); the output path is of the same kind.
    """
    (main_file,) = sample.main_files
    file_name = main_file.rsplit("/", 1)[-1]
    parts = [sample.era, sample.sample, sample.channel, file_name]
    if is_xrootd_url(output_dir):
        return "/".join([output_dir.rstrip("/"), *parts])
    return Path(output_dir).joinpath(*parts)


def resolve_output_dir(cfg: DictConfig) -> Path | str:
    """Directory of the friend files.

    Without ``nn_friend_tag``, the local ``output_dir``. With it, the friend
    directory ``<nn_friend_tag>-<decay mode>`` next to the other friends of
    the ntuples on the XRootD server (as URL)::

        <xrootd_server>//<base_dir>/<ntuple tag>/CROWNFriend/<name>
    """
    tag = cfg.get("nn_friend_tag")
    if tag is None:
        return Path(cfg.output_dir)

    tag = str(tag)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", tag):
        msg = (
            f"Invalid nn_friend_tag {tag!r}: use letters, digits, '.', '_' "
            + "and '-' only."
        )
        logger.error(msg)
        raise ValueError(msg)

    friend_name = f"{tag}-{cfg.mode.decay_mode}"
    if friend_name in {f.name for f in cfg.ntuple.friends}:
        msg = (
            f"Friend {friend_name} is an input friend of the ntuples "
            + "(ntuple.friends); choose another nn_friend_tag."
        )
        logger.error(msg)
        raise ValueError(msg)

    return xrootd_url(
        cfg.ntuple.xrootd_server,
        "/".join(
            [
                cfg.ntuple.base_dir.rstrip("/"),
                cfg.ntuple.tag,
                FRIEND_NTUPLE_DIR_NAME,
                friend_name,
            ]
        ),
    )


def friend_exists(output_file: Path | str) -> bool:
    """Whether the local or remote (XRootD URL) friend file exists."""
    if is_xrootd_url(output_file):
        return xrd_exists(output_file)
    return Path(output_file).exists()


def list_directory(directory: Path | str) -> set[str]:
    """Names in the local or remote (XRootD URL) ``directory``, which is
    empty if it does not exist."""
    try:
        if is_xrootd_url(directory):
            server, path = split_xrootd_url(directory)
            return set(xrd_ls(FileSystem(server), path))
        return set(os.listdir(directory))
    except (DirectoryNotFoundError, FileNotFoundError):
        return set()
