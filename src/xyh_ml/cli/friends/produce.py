import logging
from pathlib import Path

import hydra
import uproot

from xyh_ml import PROJECT_DIR
from xyh_ml.config.dataclasses import Config
from xyh_ml.friend_production.dataclasses import Chunk, FileSlice, Sample
from xyh_ml.io import load_json

# Set up logging
logger = logging.getLogger(__name__)


def _get_num_entries(file: str, tree: str = "ntuple") -> int:
    """Return the number of entries of ``tree`` in ``file``, or 0 if the tree
    does not exist."""
    with uproot.open(file) as f:
        if tree not in f:
            logger.debug(f"File {file} does not contain '{tree}' tree.")
            return 0
        return f[tree].num_entries


def determine_chunks(
    sample: Sample,
    chunk_size: int = 1000,
) -> list[Chunk]:
    """Split the events of a sample into chunks of ``chunk_size`` events.

    The events of all non-empty main files are concatenated (in the order of
    ``sample.main_files``) and cut into consecutive ranges of ``chunk_size``
    events; the last chunk holds the remainder. A chunk may span several
    files, so each chunk is a list of :class:`FileSlice` objects giving the
    entry range per main file together with the matching friend files.

    Files without an ``ntuple`` tree or without entries are skipped.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")

    # Friend file lists must be aligned with the main file list
    for friend_name, friend_file_list in sample.friend_files.items():
        if len(friend_file_list) != len(sample.main_files):
            msg = (
                f"Friend '{friend_name}' of sample {sample.sample} "
                + f"({sample.era}, {sample.channel}) has "
                + f"{len(friend_file_list)} files, but there are "
                + f"{len(sample.main_files)} main files."
            )
            logger.error(msg)
            raise ValueError(msg)

    # Get number of events in each file from the main files
    files = []
    empty_files = []
    for i, main_file in enumerate(sample.main_files):
        num_events = _get_num_entries(main_file)
        if num_events == 0:
            logger.debug(f"File {main_file} has no entries. Skipping.")
            empty_files.append(main_file)
            continue

        files.append(
            {
                "main_file": main_file,
                "friend_files": {
                    name: friend_file_list[i]
                    for name, friend_file_list in sample.friend_files.items()
                },
                "num_events": num_events,
            }
        )

    if empty_files:
        logger.info(
            f"Skipped {len(empty_files)} empty file(s) of sample "
            + f"{sample.sample} ({sample.era}, {sample.channel})."
        )

    # Get the total number of events in the sample and form chunk indices
    total_events = sum(f["num_events"] for f in files)
    chunk_indices = list(range(0, total_events, chunk_size)) + [total_events]

    # Determine files that belong to each chunk based on the number of events
    # in each file. Global event index ranges are mapped onto per-file entry
    # ranges by walking through the files in parallel to the chunks.
    chunks = []
    file_idx = 0
    file_offset = 0  # global index of the first event of files[file_idx]
    for chunk_idx, (start, stop) in enumerate(
        zip(chunk_indices[:-1], chunk_indices[1:])
    ):
        slices = []
        position = start
        while position < stop:
            file = files[file_idx]
            file_stop = file_offset + file["num_events"]
            slice_stop = min(stop, file_stop)
            slices.append(
                FileSlice(
                    main_file=file["main_file"],
                    friend_files=file["friend_files"],
                    entry_start=position - file_offset,
                    entry_stop=slice_stop - file_offset,
                )
            )
            position = slice_stop

            # Advance to the next file once the current one is exhausted
            if position == file_stop:
                file_idx += 1
                file_offset = file_stop

        chunks.append(Chunk(index=chunk_idx, slices=slices))

    logger.info(
        f"Split sample {sample.sample} ({sample.era}, {sample.channel}) with "
        + f"{total_events} events in {len(files)} file(s) into "
        + f"{len(chunks)} chunk(s) of up to {chunk_size} events."
    )

    return chunks


@hydra.main(
    config_path=str(PROJECT_DIR / "config"),
    config_name="scrape_metadata",
    version_base="1.3",
)
def main(cfg: Config):
    # Get sample metadata
    metadata_file = Path(cfg.metadata_file)
    samples = [Sample(**s) for s in load_json(metadata_file)]


if __name__ == "__main__":
    main()
