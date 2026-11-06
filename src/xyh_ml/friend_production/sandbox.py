"""Software and data for jobs at sites without the shared file systems.

Two files are stored on grid storage and fetched by every job:

- The environment pack, a self-extracting script made with ``pixi-pack``. It
  only changes with the lock file, so runs share it.
- The job bundle, with the code, the models and the metadata of a run.

Their names contain a hash of the content. Worker nodes can cache files by
name, so a file must never be replaced under the same name.

This module does not import torch, so that planning jobs stays cheap.
"""

import gzip
import hashlib
import io
import logging
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Iterable
from pathlib import Path

from xyh_ml.friend_production.condor import CondorError
from xyh_ml.friend_production.xrootd import xrd_exists, xrd_upload

logger = logging.getLogger(__name__)

ENVIRONMENT = "cpu"
PIXI_PACK_TIMEOUT_S = 1800

# Files of a training directory needed to evaluate a model
MODEL_FILES = ("transformations.pkl", "best_model.ckpt")

# Directories in the bundle, relative to where the job runs
MODELS_DIR = "models"
METADATA_DIR = "metadata"

_HASH_LENGTH = 16


class SandboxError(CondorError):
    """The files for the jobs can not be built or stored."""


def _digest(*contents: bytes) -> str:
    digest = hashlib.sha256()
    for content in contents:
        digest.update(content)
    return digest.hexdigest()[:_HASH_LENGTH]


def pixi_pack_binary() -> Path:
    """``pixi-pack`` of the running environment."""
    binary = Path(sys.executable).parent / "pixi-pack"
    if not binary.is_file():
        raise SandboxError(
            f"{binary} not found. Update the environment with: pixi install "
            + f"-e {ENVIRONMENT}"
        )
    return binary


def _run(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=PIXI_PACK_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SandboxError(f"{command[0]} failed: {e}") from e
    if result.returncode != 0:
        raise SandboxError(
            f"{' '.join(command)} failed:\n{result.stderr or result.stdout}"
        )
    return result.stdout


def environment_pack_name(project_dir: Path, binary: Path) -> str:
    """Name of the pack, which changes with the lock file and ``pixi-pack``."""
    try:
        lock = (project_dir / "pixi.lock").read_bytes()
    except OSError as e:
        raise SandboxError(f"Can not read the lock file: {e}") from e
    version = _run([str(binary), "--version"])
    return (
        f"env-{ENVIRONMENT}-"
        + _digest(lock, ENVIRONMENT.encode(), version.encode())
        + ".sh"
    )


def upload_if_missing(local_file: Path, url: str):
    """Copy ``local_file`` to ``url``, unless the name exists already."""
    try:
        if xrd_exists(url):
            logger.info(f"Using the existing {url}")
            return
        logger.info(f"Uploading {url}")
        xrd_upload(local_file, url)
    except RuntimeError as e:
        raise SandboxError(str(e)) from e


def environment_pack(sandbox_url: str, project_dir: Path) -> str:
    """URL of the environment pack, which is built if it does not exist."""
    binary = pixi_pack_binary()
    name = environment_pack_name(project_dir, binary)
    url = f"{sandbox_url.rstrip('/')}/{name}"
    try:
        if xrd_exists(url):
            logger.info(f"Using the existing {url}")
            return url
    except RuntimeError as e:
        raise SandboxError(str(e)) from e

    with tempfile.TemporaryDirectory(prefix="xyh_ml_env_") as directory:
        pack = Path(directory) / name
        logger.info(f"Building the environment pack {name}")
        _run(
            [
                str(binary),
                str(project_dir / "pyproject.toml"),
                "--environment",
                ENVIRONMENT,
                "--platform",
                "linux-64",
                "--create-executable",
                # The editable install of this package, whose code is in the
                # bundle
                "--ignore-pypi-non-wheel",
                "--output-file",
                str(pack),
            ]
        )
        upload_if_missing(pack, url)
    return url


def _add_file(archive: tarfile.TarFile, path: Path, name: str):
    """Add a file with fixed attributes, so that the hash only depends on the
    content."""
    info = tarfile.TarInfo(name)
    info.size = path.stat().st_size
    info.mode = 0o644
    with path.open("rb") as f:
        archive.addfile(info, f)


def _source_files(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            yield path


def bundle_content(
    project_dir: Path,
    model_dirs: dict[str, Path],
    metadata_files: dict[str, Path],
) -> dict[str, Path]:
    """Source file per path in the bundle.

    ``model_dirs`` holds the training directory per directory below
    ``models``, and ``metadata_files`` the metadata file per path below
    ``metadata``.
    """
    content = {}
    for directory in ("src", "config"):
        for path in _source_files(project_dir / directory):
            content[path.relative_to(project_dir).as_posix()] = path
    for name, directory in model_dirs.items():
        for file_name in MODEL_FILES:
            content[f"{MODELS_DIR}/{name}/{file_name}"] = directory / file_name
    for name, path in metadata_files.items():
        content[f"{METADATA_DIR}/{name}"] = path

    missing = sorted(str(p) for p in content.values() if not p.is_file())
    if missing:
        raise SandboxError(
            "Files for the jobs not found:\n  " + "\n  ".join(missing)
        )
    return content


def write_bundle(content: dict[str, Path], path: Path):
    """Write the bundle, whose bytes only depend on its content."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name in sorted(content):
            _add_file(archive, content[name], name)
    with (
        path.open("wb") as raw,
        gzip.GzipFile(
            filename="", fileobj=raw, mode="wb", mtime=0
        ) as compressed,
    ):
        compressed.write(buffer.getvalue())


def job_bundle(
    sandbox_url: str,
    project_dir: Path,
    model_dirs: dict[str, Path],
    metadata_files: dict[str, Path],
) -> str:
    """URL of the job bundle of a run, which is uploaded if it does not
    exist."""
    content = bundle_content(project_dir, model_dirs, metadata_files)
    with tempfile.TemporaryDirectory(prefix="xyh_ml_bundle_") as directory:
        bundle = Path(directory) / "bundle.tar.gz"
        write_bundle(content, bundle)
        name = f"bundle-{_digest(bundle.read_bytes())}.tar.gz"
        url = f"{sandbox_url.rstrip('/')}/{name}"
        upload_if_missing(bundle, url)
    return url
