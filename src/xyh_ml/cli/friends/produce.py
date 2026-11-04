import dataclasses
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
import uproot
from omegaconf import DictConfig, OmegaConf

from xyh_ml import PROJECT_DIR
from xyh_ml.config.dataclasses import (
    Campaign,
    Config,
    Network,
    OutputClass,
    Variable,
)
from xyh_ml.data.data_frame import prepare_with_parameter_values
from xyh_ml.data.torch import create_torch_dataset_for_evaluation
from xyh_ml.friend_production.dataclasses import Chunk, FileSlice, Sample
from xyh_ml.friend_production.profiling import Profiler
from xyh_ml.io import load_json, load_transformations
from xyh_ml.nn.mlp import create_multi_layer_perceptron_model
from xyh_ml.nn.predictions import predict

# Set up logging
logger = logging.getLogger(__name__)

# Name of the event column
EVENT_COLUMN_NAME = "event"

# Index levels of the data frames returned by load_data_frame
INDEX_NAMES = ["chunk", "slice", "entry"]

# Columns evaluated per event and parameter value combination and written to
# the friend tree
SCORE_COLUMNS = ["max_score", "max_score_index"]


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


def _read_tree_slice(
    file: str,
    columns: list[str],
    file_slice: FileSlice,
    tree: str = "ntuple",
) -> pd.DataFrame:
    """Read those of ``columns`` that exist in ``tree`` of ``file`` for the
    entry range of ``file_slice`` into a data frame with a fresh
    ``RangeIndex``. Columns missing in the tree are not included."""
    with uproot.open(file) as f:
        if tree not in f:
            msg = f"File {file} does not contain '{tree}' tree."
            logger.error(msg)
            raise KeyError(msg)
        t = f[tree]

        # Friend trees are aligned entry by entry with the main tree, so the
        # entry range must be valid in every tree
        if t.num_entries < file_slice.entry_stop:
            msg = (
                f"Tree '{tree}' in {file} has {t.num_entries} entries, but "
                + f"entries up to {file_slice.entry_stop} of "
                + f"{file_slice.main_file} are requested."
            )
            logger.error(msg)
            raise ValueError(msg)

        keys = set(t.keys())
        available = [c for c in columns if c in keys]
        if not available:
            return pd.DataFrame(index=range(file_slice.num_entries))

        return t.arrays(
            available,
            entry_start=file_slice.entry_start,
            entry_stop=file_slice.entry_stop,
            library="pd",
        ).reset_index(drop=True)


def load_data_frame(
    chunk: Chunk,
    columns: list[str],
) -> pd.DataFrame:
    """Load a data frame for a :class:`Chunk` object.

    For each :class:`FileSlice` of the chunk, the requested columns are read
    from the main file and all friend files and joined side by side; only the
    entries in the range ``[entry_start, entry_stop)`` are included. Each
    column is taken from the first tree that holds it (main file first, then
    the friends in order). The slices are then concatenated into a single
    data frame with the index levels ``chunk`` (chunk index), ``slice``
    (index of the slice in ``chunk.slices``) and ``entry`` (entry number in
    the main file of the slice).
    """

    # Add data frame for each slice of the chunk to a list, then concatenate them
    data_frames = []

    for slice_index, slice in enumerate(chunk.slices):
        # Read each requested column from the first tree that contains it
        remaining = list(dict.fromkeys(columns))
        frames = []
        for file in [slice.main_file, *slice.friend_files.values()]:
            df = _read_tree_slice(file, remaining, slice)
            remaining = [c for c in remaining if c not in df.columns]
            frames.append(df)

        if remaining:
            msg = (
                f"Columns {remaining} not found in {slice.main_file} or its "
                + f"friend files {list(slice.friend_files.values())}."
            )
            logger.error(msg)
            raise KeyError(msg)

        # Join main and friend columns and restore the requested order
        df = pd.concat(frames, axis=1)[list(dict.fromkeys(columns))]

        # Entry numbers restart in every main file, so the slice index is
        # part of the index to identify events within a chunk uniquely
        n = df.shape[0]
        df.index = pd.MultiIndex.from_arrays(
            [
                np.full(n, chunk.index),
                np.full(n, slice_index),
                np.arange(slice.entry_start, slice.entry_stop),
            ],
            names=INDEX_NAMES,
        )

        data_frames.append(df)

    if not data_frames:
        return pd.DataFrame(
            columns=list(dict.fromkeys(columns)),
            index=pd.MultiIndex.from_arrays(
                [[]] * len(INDEX_NAMES), names=INDEX_NAMES
            ),
        )

    return pd.concat(data_frames)


def score_branch_names(parameter_values: pd.DataFrame) -> list[str]:
    """Names of the friend tree branches: for every parameter value
    combination (outer loop) the columns in ``SCORE_COLUMNS`` (inner loop),
    e.g. ``max_score_m_x_400_m_y_100`` and
    ``max_score_index_m_x_400_m_y_100``."""
    parameter_value_strs = [
        "_".join(f"{col}_{int(val)}" for col, val in row.items())
        for row in parameter_values.to_dict(orient="records")
    ]
    return [
        f"{score_column}_{parameter_value_str}"
        for parameter_value_str in parameter_value_strs
        for score_column in SCORE_COLUMNS
    ]


def score_branch_types(branch_names: list[str]) -> dict[str, type]:
    """Types of the friend tree branches: ``int32`` for ``max_score_index``
    branches, ``float32`` for ``max_score`` branches."""
    return {
        name: np.int32 if name.startswith("max_score_index") else np.float32
        for name in branch_names
    }


def _process_fold(
    data_frame: pd.DataFrame,
    fold: int,
    n_folds: int,
    input_variables: list[Variable],
    input_parameters: list[Variable],
    output_variable: Variable,
    output_classes: list[OutputClass],
    models: dict[int, torch.nn.Module],
    transformations: dict[int, dict[str, dict[str, Any]]],
    batch_size: int,
    profiler: Profiler | None = None,
) -> pd.DataFrame:
    """Evaluate the model of ``fold`` on the events of ``data_frame`` that
    were held out in its training (``event % n_folds == fold``).

    The time spent in the model evaluation is added to the stage
    ``predict`` of ``profiler``.

    Returns a copy of these events with the columns ``max_score`` (highest
    softmax score) and ``max_score_index`` (index of the class with the
    highest score in ``output_classes``).
    """

    # Only keep events that have not been used for training in this
    # fold. Copy to be able to add columns without modifying the input.
    fold_data_frame = data_frame[
        data_frame[EVENT_COLUMN_NAME] % n_folds == fold
    ].copy()

    # Nothing to evaluate if no event of the chunk belongs to this fold
    if fold_data_frame.empty:
        return fold_data_frame.assign(
            max_score=pd.Series(dtype=np.float32),
            max_score_index=pd.Series(dtype=np.int64),
        )

    # Get the fold transformations and the fold model
    fold_transformations = transformations[fold]
    fold_model = models[fold]

    # Prepare the torch dataset
    fold_dataset = create_torch_dataset_for_evaluation(
        fold_data_frame,
        input_variables,
        input_parameters,
        fold_transformations,
    )

    # Evaluate the model. The dataset is already in memory, so no data loader
    # worker processes are used; forking them from the multi-threaded process
    # for every fold and chunk is slow and can deadlock.
    profiler = profiler if profiler is not None else Profiler()
    with profiler.measure("predict"):
        predictions = predict(
            fold_model,
            fold_dataset,
            device=_device(),
            data_loader_num_workers=0,
            batch_size=batch_size,
        )

    # Reorder predictions such that they match the order of output
    # classes in the configuration file. The model outputs are ordered like
    # the categories of the ordinal encoder of the output variable.
    t = fold_transformations["ordinal_encoding"]["transformation"]
    i_category = t.feature_names_in_.tolist().index(output_variable.name)
    transformation_categories = t.categories_[i_category].tolist()
    correct_order = [
        transformation_categories.index(output_class.name)
        for output_class in output_classes
    ]
    predictions = predictions[:, correct_order]

    # Store the maximum score and the index of the class with the maximum
    # score of every event in the fold data frame
    max_score_index = np.argmax(predictions, axis=1)
    fold_data_frame["max_score"] = np.take_along_axis(
        predictions, max_score_index[:, np.newaxis], axis=1
    )[:, 0]
    fold_data_frame["max_score_index"] = max_score_index

    return fold_data_frame


def _flatten_scores(
    data_frame: pd.DataFrame,
    parameter_values: pd.DataFrame,
) -> pd.DataFrame:
    """Turn the per-parameter-value rows of ``data_frame`` (index level
    ``parameter_value``) into one row per event.

    The returned data frame has the remaining index levels of ``data_frame``
    (sorted) and the columns ``<score column>_<parameter values>`` (see
    :func:`score_branch_names`).
    """
    # One row per event, columns (score column, parameter value index)
    flat = data_frame[SCORE_COLUMNS].unstack("parameter_value")

    # Order columns by parameter value first, then by score column
    flat = flat.reorder_levels([1, 0], axis=1)
    flat = flat.reindex(
        columns=pd.MultiIndex.from_product(
            [range(parameter_values.shape[0]), SCORE_COLUMNS]
        )
    )

    # Construct column names for the flattened scores
    flat.columns = score_branch_names(parameter_values)

    return flat


def _process_chunk(
    chunk: Chunk,
    columns: list[str],
    campaign_name: str,
    input_variables: list[Variable],
    input_parameters: list[Variable],
    output_variable: Variable,
    output_classes: list[OutputClass],
    parameter_values: pd.DataFrame,
    models: dict[int, torch.nn.Module],
    transformations: dict[int, dict[str, dict[str, Any]]],
    n_folds: int,
    batch_size: int,
    profiler: Profiler | None = None,
) -> pd.DataFrame:
    """Load a chunk, evaluate the fold models for all parameter values and
    return the flattened scores (see :func:`_flatten_scores`) with one row
    per event, in the order of the events in the chunk.

    The stages ``load``, ``prepare``, ``evaluate`` (including ``predict``)
    and ``flatten`` are timed with ``profiler``, which also records the size
    of the loaded (``loaded``) and the expanded (``expanded``) data frame.
    """
    profiler = profiler if profiler is not None else Profiler()

    # Load the data frame for the chunk and add the campaign column
    logger.info(f"Chunk {chunk.index} has {chunk.num_entries} entries.")
    with profiler.measure_memory("load"):
        chunk_data_frame = load_data_frame(chunk, columns=columns)
        chunk_data_frame["campaign"] = campaign_name
    profiler.add_data_frame("loaded", chunk_data_frame)

    # Prepare data frames for evaluation of multiple input parameter values
    with profiler.measure_memory("prepare"):
        data_frame = prepare_with_parameter_values(
            chunk_data_frame,
            parameter_values,
        )
    profiler.add_data_frame("expanded", data_frame)

    # Split data frame into folds and evaluate the model for each fold
    with profiler.measure("evaluate"):
        fold_data_frames = [
            _process_fold(
                data_frame,
                i,
                n_folds,
                input_variables,
                input_parameters,
                output_variable,
                output_classes,
                models,
                transformations,
                batch_size,
                profiler=profiler,
            )
            for i in range(n_folds)
        ]

    # Merge fold data frames and flatten the scores to one row per event
    with profiler.measure("flatten"):
        data_frame_eval = pd.concat(fold_data_frames)
        scores = _flatten_scores(data_frame_eval, parameter_values)

    profiler.add_chunk(len(chunk_data_frame), len(data_frame))

    # Return the scores in the order of the events of the chunk
    return scores.reindex(chunk_data_frame.index)


def _device() -> str:
    """Device the models are evaluated on."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _warm_up_models(models: dict[int, torch.nn.Module], input_size: int):
    """Move the models to the evaluation device and evaluate them once.

    The first CUDA operation of the process initializes the CUDA context,
    which takes seconds and allocates a large amount of host memory. Doing
    this before the friend production keeps it out of the timing and memory
    profile of the evaluation.
    """
    device = _device()
    with torch.no_grad():
        for model in models.values():
            model.eval().to(device)(torch.zeros(1, input_size, device=device))
    if device == "cuda":
        torch.cuda.synchronize()


def _fold_training_dirs(cfg: DictConfig) -> dict[int, Path]:
    """Return the training directory of the model of each fold.

    ``artifacts.training_dir`` is interpolated from ``mode.fold``, so it is
    resolved again with ``mode.fold`` set to each fold.
    """
    return {
        fold: Path(
            OmegaConf.merge(
                cfg, {"mode": {"fold": fold}}
            ).artifacts.training_dir
        )
        for fold in range(cfg.mode.n_folds)
    }


def _load_fold_model(
    training_dir: Path,
    input_size: int,
    output_size: int,
    network: Network,
) -> tuple[torch.nn.Module, dict[str, dict[str, Any]]]:
    """Load the best model and the transformations of one fold from its
    training directory."""
    # Load transformations from the training artifacts
    transformations = load_transformations(
        training_dir / "transformations.pkl"
    )

    # Create the neural network model
    model = create_multi_layer_perceptron_model(
        input_size, output_size, network
    )

    # Load checkpoints from the training artifacts
    state_dict = torch.load(
        training_dir / "best_model.ckpt", map_location="cpu", weights_only=True
    )["state_dict"]

    # Rename keys in the state dict to match the bare model's keys if they
    # are prefixed with "model." (Lightning module attribute)
    state_dict = {
        key.removeprefix("model."): value for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict)
    model.eval()

    logger.info(f"Loaded model and transformations from {training_dir}")

    return model, transformations


def write_friend_file(
    output_file: Path,
    branch_types: dict[str, type],
    num_entries: int,
    data_frames: Iterable[pd.DataFrame],
    tree: str = "ntuple",
    profiler: Profiler | None = None,
) -> None:
    """Write a friend tree with ``num_entries`` entries to ``output_file``.

    ``data_frames`` yields the branch values in consecutive blocks of rows.
    Their ``entry`` index level must continue the entries written so far, so
    that the friend tree is aligned entry by entry with the main tree. The
    file is written to a temporary file first and only moved to
    ``output_file`` if all entries were written, so that no incomplete
    friend file is left behind. The time spent writing is added to the
    stage ``write`` of ``profiler``.
    """
    profiler = profiler if profiler is not None else Profiler()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = output_file.with_name(output_file.name + ".tmp")

    try:
        # Creating the tree with all branches and closing the file (which
        # writes the remaining baskets and metadata) count as writing, too
        with profiler.measure("write"):
            f = uproot.recreate(tmp_file)
            f.mktree(tree, branch_types)
        try:
            n_written = 0
            for df in data_frames:
                entries = df.index.get_level_values("entry").to_numpy()
                expected = np.arange(n_written, n_written + len(df))
                if not np.array_equal(entries, expected):
                    msg = (
                        f"Entries {entries[:1]}...{entries[-1:]} do not "
                        + f"continue the {n_written} entries written to "
                        + f"{output_file}."
                    )
                    logger.error(msg)
                    raise ValueError(msg)

                missing = set(branch_types) - set(df.columns)
                if missing:
                    msg = f"Branches {sorted(missing)} missing in data frame."
                    logger.error(msg)
                    raise KeyError(msg)

                if len(df):
                    with profiler.measure("write"):
                        f[tree].extend(
                            {
                                name: df[name].to_numpy(dtype=dtype)
                                for name, dtype in branch_types.items()
                            }
                        )
                n_written += len(df)
        finally:
            with profiler.measure("write"):
                f.close()

        if n_written != num_entries:
            msg = (
                f"Wrote {n_written} entries to {output_file}, but the main "
                + f"tree has {num_entries} entries."
            )
            logger.error(msg)
            raise ValueError(msg)

        tmp_file.replace(output_file)
    finally:
        tmp_file.unlink(missing_ok=True)

    logger.info(
        f"Wrote friend tree '{tree}' with {n_written} entries and "
        + f"{len(branch_types)} branches to {output_file}"
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
    return dataclasses.replace(
        sample,
        main_files=[sample.main_files[i]],
        friend_files={
            name: [friend_file_list[i]]
            for name, friend_file_list in sample.friend_files.items()
        },
    )


def default_output_file(output_dir: Path, sample: Sample) -> Path:
    """Output path mirroring the CROWN friend layout,
    ``<output_dir>/<era>/<sample>/<channel>/<file name of the main file>``.
    """
    (main_file,) = sample.main_files
    file_name = main_file.rsplit("/", 1)[-1]
    return output_dir / sample.era / sample.sample / sample.channel / file_name


def campaign_name(campaigns: list[Campaign], era: str) -> str:
    """Name of the campaign of ``era``, as used for the ``campaign`` input."""
    for campaign in campaigns:
        if campaign.era == era:
            return campaign.name
    msg = f"No campaign defined for era {era}."
    logger.error(msg)
    raise ValueError(msg)


def produce_friend(
    sample: Sample,
    output_file: Path,
    columns: list[str],
    campaign_name: str,
    input_variables: list[Variable],
    input_parameters: list[Variable],
    output_variable: Variable,
    output_classes: list[OutputClass],
    parameter_values: pd.DataFrame,
    models: dict[int, torch.nn.Module],
    transformations: dict[int, dict[str, dict[str, Any]]],
    n_folds: int,
    batch_size: int,
    chunk_size: int,
    profiler: Profiler | None = None,
) -> Profiler:
    """Evaluate the models for all events of the single main file of
    ``sample`` and write the scores to the friend file ``output_file``.

    The friend tree has one entry per entry of the main tree. For main files
    without entries (or without ``ntuple`` tree), an empty friend tree with
    all branches is written.

    Returns the profiler with the time per stage (``total`` and the stages
    of :func:`_process_chunk` and :func:`write_friend_file`) and the memory
    usage. If no ``profiler`` is given, a new one is created and its memory
    tracking is started here.
    """
    if profiler is None:
        profiler = Profiler()
        profiler.start_memory_tracking()

    (main_file,) = sample.main_files
    num_entries = _get_num_entries(main_file)
    chunks = determine_chunks(sample, chunk_size=chunk_size)
    logger.info(
        f"Producing friend for {main_file} ({num_entries} entries, "
        + f"{len(chunks)} chunk(s)) to {output_file}"
    )

    branch_names = score_branch_names(parameter_values)

    # Evaluate the chunks lazily while writing to keep the memory usage
    # limited to one chunk
    scores = (
        _process_chunk(
            chunk,
            columns,
            campaign_name,
            input_variables,
            input_parameters,
            output_variable,
            output_classes,
            parameter_values,
            models,
            transformations,
            n_folds,
            batch_size,
            profiler=profiler,
        )
        for chunk in chunks
    )
    with profiler.measure("total"):
        write_friend_file(
            output_file,
            score_branch_types(branch_names),
            num_entries,
            scores,
            profiler=profiler,
        )

    return profiler


@hydra.main(
    config_path=str(PROJECT_DIR / "config"),
    config_name="produce",
    version_base="1.3",
)
def main(cfg: Config):
    # Find the input file in the sample metadata. Done first to fail early.
    samples = [Sample.from_dict(s) for s in load_json(Path(cfg.metadata_file))]
    sample = select_input_file(samples, cfg.input_file)

    # The analysis channel is that of the input file. Set it in the
    # configuration, so that the models trained for this channel are loaded
    # (artifacts.training_dir is interpolated from mode.channel). mode.channel
    # always has a default value, so a different value is only logged.
    if cfg.mode.channel != sample.channel:
        logger.info(
            f"Using channel {sample.channel} of input file {cfg.input_file} "
            + f"instead of mode.channel={cfg.mode.channel}."
        )
    cfg = OmegaConf.merge(cfg, {"mode": {"channel": sample.channel}})
    logger.info(
        f"Producing friend for sample {sample.sample}, era {sample.era}, "
        + f"channel {sample.channel}"
    )

    # Output file, by default following the CROWN friend directory layout
    if cfg.output_file:
        output_file = Path(cfg.output_file)
    else:
        output_file = default_output_file(Path(cfg.output_dir), sample)

    # Only variables that are part of the ntuples can be read from the files;
    # the others (e.g. campaign) are added during processing
    columns = [EVENT_COLUMN_NAME] + [
        v.name
        for v in cfg.inputs.variables
        if v.get("is_part_of_snapshot", True)
    ]

    # Get the models and transformations for all folds
    models = {}
    transformations = {}
    for fold, training_dir in _fold_training_dirs(cfg).items():
        models[fold], transformations[fold] = _load_fold_model(
            training_dir,
            len(cfg.inputs.variables) + len(cfg.inputs.parameters),
            len(cfg.outputs.classes),
            cfg.network,
        )

    # Initialize the evaluation device before profiling
    _warm_up_models(
        models, len(cfg.inputs.variables) + len(cfg.inputs.parameters)
    )

    # Track the memory usage of the friend production only, not of the
    # loading of the configuration and the models
    profiler = Profiler()
    profiler.start_memory_tracking()

    produce_friend(
        sample,
        output_file,
        columns,
        campaign_name(cfg.campaigns, sample.era),
        cfg.inputs.variables,
        cfg.inputs.parameters,
        cfg.outputs.variables[0],
        cfg.outputs.classes,
        pd.DataFrame.from_records(cfg.parameter_values),
        models,
        transformations,
        cfg.mode.n_folds,
        cfg.training.batch_size,
        cfg.chunk_size,
        profiler=profiler,
    )

    # Report time per event and per row and the memory usage
    profiler.log_summary()
    if cfg.profile_file:
        profiler.dump(Path(cfg.profile_file))


if __name__ == "__main__":
    main()
