import logging
import os

from XRootD.client import FileSystem
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
