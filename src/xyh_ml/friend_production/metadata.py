"""Sample metadata files, one per era and sample.

The metadata of a sample is stored in ``<metadata_dir>/<era>/<sample>.json.gz``
as a list of :class:`Sample` records, one per channel. A file only exists if
the metadata of the sample is complete.

This module does not import torch, so that planning jobs stays cheap.
"""

import dataclasses
import gzip
import json
import logging
import os
import re
import threading
from collections.abc import Iterable
from pathlib import Path

from xyh_ml.friend_production.dataclasses import Sample

logger = logging.getLogger(__name__)

METADATA_SUFFIX = ".json.gz"


def read_samples_file(path: Path) -> list[str]:
    """Sample names from a text file with one sample per line.

    Empty lines and lines starting with ``#`` are ignored. DAS dataset paths
    ``/<primary>/<processed>/<tier>`` are converted to the sample name
    ``<primary>_<processed>`` used for the ntuple directories.
    """
    names = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if m := re.fullmatch(r"/([^/]+)/([^/]+)/[^/]+", line):
            line = f"{m[1]}_{m[2]}"
        names.append(line)
    return names


def metadata_path(metadata_dir: Path, era: str, sample: str) -> Path:
    """Path of the metadata file of ``sample`` in ``era``."""
    return Path(metadata_dir) / era / f"{sample}{METADATA_SUFFIX}"


def write_sample_metadata(
    metadata_dir: Path,
    era: str,
    sample: str,
    records: list[Sample],
) -> Path:
    """Write the metadata file of ``sample`` in ``era``.

    The file is written to a temporary file first and then moved to its final
    path, so that an interrupted write never leaves a file that looks
    complete. The name of the temporary file does not end in
    :data:`METADATA_SUFFIX`, so it is not seen by :func:`available_samples`.
    """
    path = metadata_path(metadata_dir, era, sample)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with gzip.open(tmp_file, "wt", encoding="utf-8") as f:
            json.dump([dataclasses.asdict(r) for r in records], f)
        tmp_file.replace(path)
    finally:
        tmp_file.unlink(missing_ok=True)
    return path


def load_sample_metadata(
    metadata_dir: Path,
    era: str,
    sample: str,
) -> list[Sample]:
    """Load the records of ``sample`` in ``era``, one per channel."""
    path = metadata_path(metadata_dir, era, sample)
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return [Sample.from_dict(r) for r in json.load(f)]
    except FileNotFoundError:
        msg = (
            f"No metadata file {path} for sample {sample} ({era}). Create it "
            + "with xyh_ml.cli.friends.scrape_metadata."
        )
        logger.error(msg)
        raise ValueError(msg) from None


def available_samples(
    metadata_dir: Path,
    eras: Iterable[str],
) -> dict[str, list[str]]:
    """Names of the samples with a metadata file per era in ``eras``."""
    return {
        era: sorted(
            p.name.removesuffix(METADATA_SUFFIX)
            for p in (Path(metadata_dir) / era).glob(f"*{METADATA_SUFFIX}")
        )
        for era in eras
    }


def parse_input_path(input_file: str) -> tuple[str, str, str]:
    """Era, sample and channel of a main file, from the last components
    ``<era>/<sample>/<channel>/<file name>`` of its path."""
    parts = input_file.rstrip("/").split("/")
    if len(parts) < 4 or not all(parts[-4:]):
        msg = (
            f"Input file {input_file} does not have a path ending in "
            + "<era>/<sample>/<channel>/<file name>."
        )
        logger.error(msg)
        raise ValueError(msg)
    era, sample, channel = parts[-4:-1]
    return era, sample, channel


def single_file_sample(sample: Sample, i: int) -> Sample:
    """Copy of ``sample`` with only its ``i``-th main file and the
    corresponding friend files."""
    return dataclasses.replace(
        sample,
        main_files=[sample.main_files[i]],
        friend_files={
            name: [friend_file_list[i]]
            for name, friend_file_list in sample.friend_files.items()
        },
    )


def select_input_file(
    samples: list[Sample],
    input_file: str,
) -> Sample:
    """Find ``input_file`` among the main files of ``samples`` and return a
    sample that only holds this main file and its friend files.

    ``input_file`` is either the full path of a main file as stored in the
    metadata or a trailing part of it (e.g. the path without the XRootD
    server) that matches exactly one main file. Era, sample name and
    analysis channel of the returned sample are those of the sample the file
    belongs to.
    """
    exact = [
        (sample, i)
        for sample in samples
        for i, main_file in enumerate(sample.main_files)
        if main_file == input_file
    ]
    suffix = "/" + input_file.lstrip("/")
    matches = exact or [
        (sample, i)
        for sample in samples
        for i, main_file in enumerate(sample.main_files)
        if main_file.endswith(suffix)
    ]

    if not matches:
        msg = f"Input file {input_file} not found in the sample metadata."
        logger.error(msg)
        raise ValueError(msg)
    if len(matches) > 1:
        msg = (
            f"Input file {input_file} matches several main files: "
            + f"{[s.main_files[i] for s, i in matches]}."
        )
        logger.error(msg)
        raise ValueError(msg)

    sample, i = matches[0]
    return single_file_sample(sample, i)


def select_input_files(
    metadata_dir: Path,
    input_files: Iterable[str],
) -> list[Sample]:
    """Single-file samples of ``input_files``, in the same order.

    The metadata file of every sample that occurs is loaded once.
    """
    records: dict[tuple[str, str], list[Sample]] = {}
    selected = []
    for input_file in input_files:
        era, sample, _ = parse_input_path(input_file)
        if (era, sample) not in records:
            records[era, sample] = load_sample_metadata(
                metadata_dir, era, sample
            )
        selected.append(select_input_file(records[era, sample], input_file))
    return selected
