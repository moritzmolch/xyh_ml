"""Tests for the functions of the friend production CLI script."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import uproot
from omegaconf import OmegaConf

from xyh_ml.cli.friends import produce
from xyh_ml.config.dataclasses import (
    Campaign,
    Network,
    OutputClass,
    Variable,
)
from xyh_ml.data.data_frame import prepare_with_parameter_values
from xyh_ml.data.transformations import derive_transformations
from xyh_ml.io import dump_transformations
from xyh_ml.nn.mlp import create_multi_layer_perceptron_model
from xyh_ml.friend_production.dataclasses import (
    Chunk,
    FileSlice,
    Sample,
)


@pytest.fixture
def write_tree(tmp_path):
    """Factory writing a ROOT file with an ``ntuple`` tree from ``branches``.

    ``branches`` maps branch names to arrays. ``None`` writes a file without
    an ``ntuple`` tree, an empty dict an ``ntuple`` tree without entries.
    """

    def factory(name, branches):
        path = tmp_path / name
        with uproot.recreate(path) as f:
            if branches is None:
                f["other"] = {"x": np.arange(2, dtype="float64")}
            elif not branches:
                f.mktree("ntuple", {"pt": "float64"})
            else:
                f["ntuple"] = {
                    k: np.asarray(v, dtype="float64")
                    for k, v in branches.items()
                }
        return str(path)

    return factory


def _sample(main_files, friend_files=None):
    return Sample(
        era="2024",
        sample="X",
        channel="et",
        main_files=main_files,
        friend_files=friend_files or {},
        shifts=[],
    )


def _filled(write_tree, name, n, offset=0.0):
    """Main file with ``n`` entries, ``pt`` = offset + entry, ``event`` =
    entry."""
    return write_tree(
        name,
        {"pt": offset + np.arange(n), "event": np.arange(n)},
    )


def _slices(chunk):
    return [
        (s.main_file.rsplit("/", 1)[-1], s.entry_start, s.entry_stop)
        for s in chunk.slices
    ]


# ---------------------------------------------------------- _get_num_entries


def test_get_num_entries_filled(write_tree):
    assert produce._get_num_entries(_filled(write_tree, "a.root", 4)) == 4


def test_get_num_entries_empty_tree(write_tree):
    assert produce._get_num_entries(write_tree("a.root", {})) == 0


def test_get_num_entries_missing_tree(write_tree):
    assert produce._get_num_entries(write_tree("a.root", None)) == 0


# ---------------------------------------------------------- determine_chunks


def test_determine_chunks_spans_file_boundaries(write_tree):
    main = [
        _filled(write_tree, "a.root", 7),
        _filled(write_tree, "b.root", 12),
        _filled(write_tree, "c.root", 3),
    ]
    chunks = produce.determine_chunks(_sample(main), chunk_size=5)

    assert [c.index for c in chunks] == [0, 1, 2, 3, 4]
    assert [c.num_entries for c in chunks] == [5, 5, 5, 5, 2]
    assert _slices(chunks[0]) == [("a.root", 0, 5)]
    assert _slices(chunks[1]) == [("a.root", 5, 7), ("b.root", 0, 3)]
    assert _slices(chunks[2]) == [("b.root", 3, 8)]
    assert _slices(chunks[3]) == [("b.root", 8, 12), ("c.root", 0, 1)]
    assert _slices(chunks[4]) == [("c.root", 1, 3)]


def test_determine_chunks_exact_multiple(write_tree):
    main = [_filled(write_tree, "a.root", 10)]
    chunks = produce.determine_chunks(_sample(main), chunk_size=5)
    assert [c.num_entries for c in chunks] == [5, 5]


def test_determine_chunks_skips_empty_files_and_pairs_friends(write_tree):
    main = [
        _filled(write_tree, "a.root", 3),
        write_tree("b.root", {}),
        write_tree("c.root", None),
        _filled(write_tree, "d.root", 2),
    ]
    friends = [write_tree(f"f{i}.root", {"s": [0.0]}) for i in range(4)]
    chunks = produce.determine_chunks(
        _sample(main, {"nn": friends}), chunk_size=100
    )

    assert len(chunks) == 1
    assert _slices(chunks[0]) == [("a.root", 0, 3), ("d.root", 0, 2)]
    assert [s.friend_files for s in chunks[0].slices] == [
        {"nn": friends[0]},
        {"nn": friends[3]},
    ]


def test_determine_chunks_all_empty(write_tree):
    main = [write_tree("a.root", {}), write_tree("b.root", None)]
    assert produce.determine_chunks(_sample(main), chunk_size=5) == []


def test_determine_chunks_no_files():
    assert produce.determine_chunks(_sample([]), chunk_size=5) == []


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_determine_chunks_invalid_chunk_size(chunk_size):
    with pytest.raises(ValueError, match="chunk_size"):
        produce.determine_chunks(_sample([]), chunk_size=chunk_size)


def test_determine_chunks_friend_length_mismatch(write_tree):
    main = [_filled(write_tree, "a.root", 3)]
    with pytest.raises(ValueError, match="Friend 'nn'"):
        produce.determine_chunks(_sample(main, {"nn": []}), chunk_size=5)


# ---------------------------------------------------------- _read_tree_slice


def test_read_tree_slice_entry_range_and_missing_columns(write_tree):
    main = _filled(write_tree, "a.root", 6)
    file_slice = FileSlice(main, {}, entry_start=2, entry_stop=5)

    df = produce._read_tree_slice(main, ["pt", "nope"], file_slice)

    assert df.columns.tolist() == ["pt"]
    assert df["pt"].tolist() == [2.0, 3.0, 4.0]
    assert df.index.tolist() == [0, 1, 2]


def test_read_tree_slice_no_available_columns(write_tree):
    main = _filled(write_tree, "a.root", 6)
    file_slice = FileSlice(main, {}, entry_start=1, entry_stop=4)

    df = produce._read_tree_slice(main, ["nope"], file_slice)

    assert df.shape == (3, 0)


def test_read_tree_slice_short_friend(write_tree):
    main = _filled(write_tree, "a.root", 6)
    friend = write_tree("f.root", {"s": np.arange(3)})
    file_slice = FileSlice(main, {"nn": friend}, entry_start=2, entry_stop=5)

    with pytest.raises(ValueError, match="has 3 entries"):
        produce._read_tree_slice(friend, ["s"], file_slice)


def test_read_tree_slice_missing_tree(write_tree):
    path = write_tree("a.root", None)
    file_slice = FileSlice(path, {}, entry_start=0, entry_stop=1)

    with pytest.raises(KeyError, match="does not contain"):
        produce._read_tree_slice(path, ["pt"], file_slice)


# ----------------------------------------------------------- load_data_frame


@pytest.fixture
def main_and_friend(write_tree):
    """Two main files (3 and 4 entries) with one friend each."""
    main = [
        _filled(write_tree, "a.root", 3, offset=0.0),
        _filled(write_tree, "b.root", 4, offset=100.0),
    ]
    friends = [
        write_tree(
            "fa.root", {"score": 0.5 * np.arange(3), "pt": -np.ones(3)}
        ),
        write_tree(
            "fb.root", {"score": 50 + 0.5 * np.arange(4), "pt": -np.ones(4)}
        ),
    ]
    return _sample(main, {"nn": friends})


def test_load_data_frame_joins_friend_columns(main_and_friend):
    (chunk,) = produce.determine_chunks(main_and_friend, chunk_size=100)

    df = produce.load_data_frame(chunk, ["score", "pt"])

    # Friends add columns, not rows; requested order is kept
    assert df.columns.tolist() == ["score", "pt"]
    assert len(df) == 7
    assert df["score"].tolist() == [0.0, 0.5, 1.0, 50.0, 50.5, 51.0, 51.5]
    # Columns present in main and friend are taken from the main file
    assert df["pt"].tolist() == [0.0, 1.0, 2.0, 100.0, 101.0, 102.0, 103.0]


def test_load_data_frame_index_is_unique_across_files(main_and_friend):
    (chunk,) = produce.determine_chunks(main_and_friend, chunk_size=100)

    df = produce.load_data_frame(chunk, ["pt"])

    # Entry numbers restart in every main file, so the slice index is needed
    # to identify events uniquely
    assert df.index.names == ["chunk", "slice", "entry"]
    assert df.index.tolist() == [
        (0, 0, 0),
        (0, 0, 1),
        (0, 0, 2),
        (0, 1, 0),
        (0, 1, 1),
        (0, 1, 2),
        (0, 1, 3),
    ]
    assert df.index.is_unique


def test_load_data_frame_partial_slices(main_and_friend):
    chunks = produce.determine_chunks(main_and_friend, chunk_size=2)

    df = produce.load_data_frame(chunks[1], ["event", "score"])

    assert df.index.tolist() == [(1, 0, 2), (1, 1, 0)]
    assert df["event"].tolist() == [2.0, 0.0]
    assert df["score"].tolist() == [1.0, 50.0]


def test_load_data_frame_duplicate_columns_requested(main_and_friend):
    (chunk,) = produce.determine_chunks(main_and_friend, chunk_size=100)

    df = produce.load_data_frame(chunk, ["pt", "score", "pt"])

    assert df.columns.tolist() == ["pt", "score"]


def test_load_data_frame_unknown_column(main_and_friend):
    (chunk,) = produce.determine_chunks(main_and_friend, chunk_size=100)

    with pytest.raises(KeyError, match="nope"):
        produce.load_data_frame(chunk, ["pt", "nope"])


def test_load_data_frame_empty_chunk():
    df = produce.load_data_frame(Chunk(index=0, slices=[]), ["pt", "score"])

    assert df.empty
    assert df.columns.tolist() == ["pt", "score"]
    assert df.index.names == ["chunk", "slice", "entry"]


# ------------------------------------------------- model evaluation helpers

INPUT_VARIABLES = [
    Variable("pt", "standard_scaling"),
    Variable("campaign", "ordinal_encoding", is_part_of_snapshot=False),
]
INPUT_PARAMETERS = [
    Variable("m_x", "standard_scaling", is_part_of_snapshot=False),
]
OUTPUT_VARIABLE = Variable("class", "ordinal_encoding")

# Order in the configuration differs from the alphabetical order of the
# ordinal encoder categories (bkg, sig, tt)
OUTPUT_CLASSES = [
    OutputClass("sig", "Signal", [], True),
    OutputClass("tt", "ttbar", [], False),
    OutputClass("bkg", "Other", [], False),
]
SCORE_COLUMNS = ["score_sig", "score_tt", "score_bkg"]
N_FOLDS = 2

# Class that the model of each fold prefers (by name)
FOLD_PREFERRED_CLASS = {0: "tt", 1: "bkg"}


class ConstantModel(torch.nn.Module):
    """Model returning the same logits for every event."""

    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", torch.tensor(logits))

    def forward(self, x):
        return self.logits.expand(x.shape[0], -1)


def _encoder_logits(preferred):
    """Logits in ordinal encoder order (bkg, sig, tt) favouring a class."""
    return [3.0 if c == preferred else 0.0 for c in ["bkg", "sig", "tt"]]


def _expected_scores(preferred):
    """Softmax scores in configuration order for a preferred class."""
    e = np.exp([3.0 if c.name == preferred else 0.0 for c in OUTPUT_CLASSES])
    return e / e.sum()


@pytest.fixture
def models():
    return {
        fold: ConstantModel(_encoder_logits(preferred))
        for fold, preferred in FOLD_PREFERRED_CLASS.items()
    }


@pytest.fixture
def transformations():
    train = pd.DataFrame(
        {
            "pt": [10.0, 20.0, 30.0, 40.0],
            "campaign": ["2024_nano_v15", "2025_nano_v15"] * 2,
            "m_x": [400.0, 400.0, 1000.0, 1000.0],
            "class": ["sig", "tt", "bkg", "sig"],
        }
    )
    t = derive_transformations(
        train, INPUT_VARIABLES + INPUT_PARAMETERS + [OUTPUT_VARIABLE]
    )
    return {fold: t for fold in range(N_FOLDS)}


@pytest.fixture
def parameter_values():
    return pd.DataFrame({"m_x": [400, 1000, 2000]})


@pytest.fixture
def expanded(parameter_values):
    """Five events, expanded with three parameter values."""
    events = pd.DataFrame(
        {
            "event": [10, 11, 12, 13, 14],
            "pt": [12.0, 18.0, 25.0, 33.0, 39.0],
            "campaign": ["2024_nano_v15"] * 5,
        },
        index=pd.MultiIndex.from_tuples(
            [(0, 0, 3), (0, 0, 4), (0, 1, 0), (0, 1, 1), (0, 1, 2)],
            names=produce.INDEX_NAMES,
        ),
    )
    return prepare_with_parameter_values(events, parameter_values)


def _process_fold(data_frame, fold, models, transformations):
    return produce._process_fold(
        data_frame,
        fold,
        N_FOLDS,
        INPUT_VARIABLES,
        INPUT_PARAMETERS,
        OUTPUT_VARIABLE,
        OUTPUT_CLASSES,
        models,
        transformations,
        batch_size=4,
    )


# ------------------------------------------------------------- _process_fold


@pytest.mark.parametrize("fold", [0, 1])
def test_process_fold_selects_held_out_events(
    expanded, models, transformations, fold
):
    df = _process_fold(expanded, fold, models, transformations)

    assert len(df) == 3 * (3 if fold == 0 else 2)
    assert (df["event"] % N_FOLDS == fold).all()


@pytest.mark.parametrize("fold", [0, 1])
def test_process_fold_scores_in_config_order(
    expanded, models, transformations, fold
):
    df = _process_fold(expanded, fold, models, transformations)
    preferred = FOLD_PREFERRED_CLASS[fold]

    np.testing.assert_allclose(
        df[SCORE_COLUMNS].to_numpy(),
        np.tile(_expected_scores(preferred), (len(df), 1)),
        rtol=1e-6,
    )
    np.testing.assert_allclose(df[SCORE_COLUMNS].sum(axis=1), 1.0, rtol=1e-6)
    names = [c.name for c in OUTPUT_CLASSES]
    assert (df["max_score_index"] == names.index(preferred)).all()
    assert (df["max_score_name"] == preferred).all()


def test_process_fold_keeps_input(expanded, models, transformations):
    before = expanded.copy()

    _process_fold(expanded, 0, models, transformations)

    pd.testing.assert_frame_equal(expanded, before)


def test_process_fold_without_events(expanded, models, transformations):
    even = expanded[expanded["event"] % 2 == 0]

    df = _process_fold(even, 1, models, transformations)

    assert df.empty
    assert set(SCORE_COLUMNS + ["max_score_index"]).issubset(df.columns)


# ----------------------------------------------------------- _flatten_scores


def _evaluated(expanded, models, transformations):
    return pd.concat(
        [
            _process_fold(expanded, fold, models, transformations)
            for fold in range(N_FOLDS)
        ]
    )


def test_flatten_scores_one_row_per_event(
    expanded, models, transformations, parameter_values
):
    evaluated = _evaluated(expanded, models, transformations)

    flat = produce._flatten_scores(evaluated, parameter_values, OUTPUT_CLASSES)

    assert flat.index.names == produce.INDEX_NAMES
    assert flat.index.tolist() == [
        (0, 0, 3),
        (0, 0, 4),
        (0, 1, 0),
        (0, 1, 1),
        (0, 1, 2),
    ]
    assert flat.columns.tolist() == [
        f"{c}_m_x_{m}"
        for m in [400, 1000, 2000]
        for c in SCORE_COLUMNS + ["max_score_index"]
    ]


def test_flatten_scores_values(
    expanded, models, transformations, parameter_values
):
    # Row order of the input must not matter
    evaluated = _evaluated(expanded, models, transformations).sample(
        frac=1.0, random_state=1
    )

    flat = produce._flatten_scores(evaluated, parameter_values, OUTPUT_CLASSES)

    for (_, _, entry), row in flat.iterrows():
        event = {3: 10, 4: 11, 0: 12, 1: 13, 2: 14}[entry]
        expected = _expected_scores(FOLD_PREFERRED_CLASS[event % N_FOLDS])
        for m in [400, 1000, 2000]:
            np.testing.assert_allclose(
                [row[f"{c}_m_x_{m}"] for c in SCORE_COLUMNS],
                expected,
                rtol=1e-6,
            )


# ------------------------------------------------------------ _process_chunk


def test_process_chunk(write_tree, models, transformations, parameter_values):
    main = [
        write_tree(
            "a.root",
            {"event": [10, 11, 12], "pt": [12.0, 18.0, 25.0]},
        ),
        write_tree("b.root", {"event": [13, 14], "pt": [33.0, 39.0]}),
    ]
    sample = _sample(main)
    sample.era = "2024"
    (chunk,) = produce.determine_chunks(sample, chunk_size=100)

    df = produce._process_chunk(
        chunk,
        ["event", "pt"],
        sample,
        INPUT_VARIABLES,
        INPUT_PARAMETERS,
        OUTPUT_VARIABLE,
        OUTPUT_CLASSES,
        [Campaign("2024_nano_v15", "2024"), Campaign("2025_nano_v15", "2025")],
        parameter_values,
        models,
        transformations,
        N_FOLDS,
        batch_size=4,
    )

    # One row per event with the loaded columns, the campaign and the
    # flattened scores
    assert df.index.tolist() == [
        (0, 0, 0),
        (0, 0, 1),
        (0, 0, 2),
        (0, 1, 0),
        (0, 1, 1),
    ]
    assert df.columns[:3].tolist() == ["event", "pt", "campaign"]
    assert (df["campaign"] == "2024_nano_v15").all()
    assert len(df.columns) == 3 + 3 * 4
    for _, row in df.iterrows():
        expected = _expected_scores(
            FOLD_PREFERRED_CLASS[int(row["event"]) % N_FOLDS]
        )
        np.testing.assert_allclose(
            [row[f"{c}_m_x_2000"] for c in SCORE_COLUMNS],
            expected,
            rtol=1e-6,
        )


# ---------------------------------------------------- loading of the models


def test_fold_training_dirs_resolve_fold_interpolation():
    cfg = OmegaConf.create(
        {
            "mode": {"n_folds": 3, "fold": 1, "channel": "mt"},
            "artifacts": {
                "training_dir": (
                    "out/fold${mode.fold}of${mode.n_folds}/${mode.channel}"
                )
            },
        }
    )

    dirs = produce._fold_training_dirs(cfg)

    assert dirs == {
        0: Path("out/fold0of3/mt"),
        1: Path("out/fold1of3/mt"),
        2: Path("out/fold2of3/mt"),
    }
    # The configuration itself is not changed
    assert cfg.mode.fold == 1


def test_load_fold_model(tmp_path, transformations):
    network = Network(hidden_layers=[4], activation="relu", dropout_frac=0.0)
    torch.manual_seed(0)
    trained = create_multi_layer_perceptron_model(3, 2, network)

    # Checkpoint as written by the Lightning module, which holds the network
    # as attribute "model"
    torch.save(
        {
            "state_dict": {
                f"model.{k}": v for k, v in trained.state_dict().items()
            }
        },
        tmp_path / "best_model.ckpt",
    )
    dump_transformations(tmp_path / "transformations.pkl", transformations[0])

    model, loaded_transformations = produce._load_fold_model(
        tmp_path, 3, 2, network
    )

    assert not model.training
    for key, value in trained.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], value)
    assert loaded_transformations.keys() == transformations[0].keys()
    x = torch.rand(5, 3)
    torch.testing.assert_close(model(x), trained.eval()(x))


def test_load_fold_model_missing_artifacts(tmp_path):
    network = Network(hidden_layers=[4], activation="relu", dropout_frac=0.0)

    with pytest.raises(FileNotFoundError):
        produce._load_fold_model(tmp_path, 3, 2, network)
