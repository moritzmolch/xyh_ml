import logging
import os
import re
from pathlib import Path

from XRootD.client import CopyProcess, FileSystem
from XRootD.client.flags import DirListFlags

# Set up logging
logger = logging.getLogger(__name__)

# Names of the directories for main and friend files
MAIN_NTUPLE_DIR_NAME = "CROWNRun"
FRIEND_NTUPLE_DIR_NAME = "CROWNFriend"

# XRootD error number of a path that does not exist (kXR_NotFound)
XRD_NOT_FOUND = 3011


class DirectoryNotFoundError(RuntimeError):
    """The directory to list does not exist."""


def xrd_ls(xrd_fs: FileSystem, path: str, flags: DirListFlags | None = None):
    # If no special flags are provided, use the `NONE` flag
    flags = flags or DirListFlags.NONE

    # Perform the directory listing using the XRootD client
    status, listing = xrd_fs.dirlist(path, flags=flags)

    # Check query status
    if not status.ok:
        msg = f"Failed to list directory {path}: {status.message}"
        # A missing directory may be expected, so leave logging it to the
        # caller
        if status.errno == XRD_NOT_FOUND:
            raise DirectoryNotFoundError(msg)
        logger.error(msg)
        raise RuntimeError(msg)

    return [item.name for item in listing]


def get_era_dirs(
    xrd_fs: FileSystem,
    base_dir: str,
) -> dict[str, str]:
    # Construct main NTuple directory path
    main_ntuple_dir = os.path.join(base_dir, MAIN_NTUPLE_DIR_NAME)

    # Get available eras from the directory listing of the main NTuple directory
    listing = xrd_ls(xrd_fs, main_ntuple_dir)

    return {era: os.path.join(main_ntuple_dir, era) for era in listing}


def get_sample_dirs(
    xrd_fs: FileSystem,
    era_dir: str,
) -> dict[str, str]:
    # Get available samples from the directory listing of the era directory
    listing = xrd_ls(xrd_fs, era_dir)

    return {sample: os.path.join(era_dir, sample) for sample in listing}


def get_channel_dirs(
    xrd_fs: FileSystem,
    sample_dir: str,
) -> dict[str, str]:
    # Get available channels from the directory listing of the sample directory
    listing = xrd_ls(xrd_fs, sample_dir)

    return {channel: os.path.join(sample_dir, channel) for channel in listing}


def get_files(
    xrd_fs: FileSystem,
    channel_dir: str,
) -> list[str]:
    # Get available files from the directory listing of the channel directory
    listing = xrd_ls(xrd_fs, channel_dir)

    # Get sorted list of ROOT files (ntuple trees)
    ntuple_files = sorted(
        str(xrd_fs.url).rstrip("/")
        + "//"
        + os.path.join(channel_dir, f).lstrip("/")
        for f in listing
        if f.endswith(".root")
    )

    return ntuple_files


def get_friend_files(
    xrd_fs: FileSystem,
    base_dir: str,
    ntuple_tag: str,
    friend_name: str,
    era: str,
    sample: str,
    channel: str,
) -> list[str]:
    # Construct friend NTuple directory path
    friend_dir = os.path.join(
        base_dir,
        ntuple_tag,
        FRIEND_NTUPLE_DIR_NAME,
        friend_name,
        era,
        sample,
        channel,
    )

    # Get available files from the directory listing of the channel directory
    listing = xrd_ls(xrd_fs, friend_dir)

    # Get sorted list of ROOT files (ntuple trees)
    ntuple_files = sorted(
        str(xrd_fs.url).rstrip("/")
        + "//"
        + os.path.join(friend_dir, f).lstrip("/")
        for f in listing
        if f.endswith(".root")
    )

    return ntuple_files


def is_xrootd_url(path: str | Path) -> bool:
    """Whether ``path`` is an XRootD URL (``root://server//path``)."""
    return isinstance(path, str) and path.startswith(("root://", "xroot://"))


def xrootd_url(server: str, path: str) -> str:
    """URL of the absolute ``path`` on ``server``, e.g.
    ``root://server:1094//store/file.root``."""
    return f"{server.rstrip('/')}//{path.lstrip('/')}"


def split_xrootd_url(url: str) -> tuple[str, str]:
    """Server and absolute path of an XRootD URL."""
    m = re.fullmatch(r"(x?root://[^/]+)//?(/?[^/].*)", url)
    if not m:
        msg = f"Not an XRootD URL with a path: {url}"
        logger.error(msg)
        raise ValueError(msg)
    return m[1], "/" + m[2].lstrip("/")


def xrd_exists(url: str) -> bool:
    """Whether the file or directory ``url`` exists."""
    server, path = split_xrootd_url(url)
    status, _ = FileSystem(server).stat(path)
    if status.ok:
        return True
    if status.errno == XRD_NOT_FOUND:
        return False
    msg = f"Failed to check whether {url} exists: {status.message}"
    logger.error(msg)
    raise RuntimeError(msg)


def xrd_upload(local_file: Path, url: str):
    """Copy ``local_file`` to ``url``, replacing an existing file.

    Missing directories are created. The checksums (adler32) of source and
    target are compared, and the target is only kept if the copy succeeded
    (persist on successful close), so that no incomplete file is left.
    """
    process = CopyProcess()
    process.add_job(
        str(local_file),
        url,
        force=True,
        posc=True,
        mkdir=True,
        checksummode="end2end",
        checksumtype="adler32",
        rmBadCksum=True,
    )
    status = process.prepare()
    if status.ok:
        status, results = process.run()
        if status.ok and results and not results[0]["status"].ok:
            status = results[0]["status"]
    if not status.ok:
        msg = f"Failed to copy {local_file} to {url}: {status.message}"
        logger.error(msg)
        raise RuntimeError(msg)
