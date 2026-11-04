"""Tests for the functions of the friend production CLI script."""

import dataclasses
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
CLASS_NAMES = [c.name for c in OUTPUT_CLASSES]
SCORE_COLUMNS = ["max_score", "max_score_index"]
N_FOLDS = 2

# Class that the model of each fold prefers (by name) and the logit of this
# class (all others are 0). The logits differ, so that the maximum scores of
# the folds differ as well.
FOLD_PREFERRED_CLASS = {0: "tt", 1: "bkg"}
FOLD_LOGIT = {0: 3.0, 1: 1.0}


class ConstantModel(torch.nn.Module):
    """Model returning the same logits for every event."""

    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", torch.tensor(logits))

    def forward(self, x):
        return self.logits.expand(x.shape[0], -1)


def _encoder_logits(fold):
    """Logits of the model of ``fold`` in ordinal encoder order (bkg, sig,
    tt)."""
    return [
        FOLD_LOGIT[fold] if c == FOLD_PREFERRED_CLASS[fold] else 0.0
        for c in ["bkg", "sig", "tt"]
    ]


def _expected(event):
    """Expected (max_score, max_score_index) of an event, evaluated by the
    model of fold ``event % N_FOLDS``; the index refers to the order of the
    classes in the configuration."""
    fold = event % N_FOLDS
    e = np.exp(FOLD_LOGIT[fold])
    max_score = e / (e + len(OUTPUT_CLASSES) - 1)
    return max_score, CLASS_NAMES.index(FOLD_PREFERRED_CLASS[fold])


@pytest.fixture
def models():
    return {fold: ConstantModel(_encoder_logits(fold)) for fold in FOLD_LOGIT}


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
def test_process_fold_max_score(expanded, models, transformations, fold):
    df = _process_fold(expanded, fold, models, transformations)
    max_score, max_score_index = _expected(fold)

    assert df.columns[-2:].tolist() == SCORE_COLUMNS
    assert df["max_score"].shape == (len(df),)
    np.testing.assert_allclose(df["max_score"], max_score, rtol=1e-6)
    # Index in the order of the configuration, not of the encoder categories
    assert (df["max_score_index"] == max_score_index).all()


def test_process_fold_max_score_per_event(transformations):
    """The maximum is taken per event (row) of the predictions."""
    events = pd.DataFrame(
        {
            "event": [0, 2, 4],
            "pt": [10.0, 20.0, 30.0],
            "campaign": ["2024_nano_v15"] * 3,
            "m_x": [400.0] * 3,
        }
    )

    class PtModel(torch.nn.Module):
        """Prefers sig (encoder index 1) for small pt, tt (index 2) for
        large pt, with a margin growing with pt."""

        def forward(self, x):
            pt = x[:, :1]
            return torch.cat([torch.zeros_like(pt), -pt, pt], dim=1)

    df = produce._process_fold(
        events,
        0,
        N_FOLDS,
        INPUT_VARIABLES,
        INPUT_PARAMETERS,
        OUTPUT_VARIABLE,
        OUTPUT_CLASSES,
        {0: PtModel()},
        transformations,
        batch_size=2,
    )

    # Expected softmax from the standardized pt (mean 25, std sqrt(125) of
    # the training data of the transformations fixture)
    pt = (events["pt"].to_numpy() - 25.0) / np.sqrt(125.0)
    logits = np.stack([np.zeros_like(pt), -pt, pt], axis=1)
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    np.testing.assert_allclose(df["max_score"], probs.max(axis=1), rtol=1e-5)
    # Encoder order (bkg, sig, tt) -> configuration order (sig, tt, bkg)
    to_config = {0: 2, 1: 0, 2: 1}
    assert df["max_score_index"].tolist() == [
        to_config[i] for i in probs.argmax(axis=1)
    ]


def test_process_fold_keeps_input(expanded, models, transformations):
    before = expanded.copy()

    _process_fold(expanded, 0, models, transformations)

    pd.testing.assert_frame_equal(expanded, before)


def test_process_fold_without_events(expanded, models, transformations):
    even = expanded[expanded["event"] % 2 == 0]

    df = _process_fold(even, 1, models, transformations)

    assert df.empty
    assert set(SCORE_COLUMNS).issubset(df.columns)


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

    flat = produce._flatten_scores(evaluated, parameter_values)

    assert flat.index.names == produce.INDEX_NAMES
    assert flat.index.tolist() == [
        (0, 0, 3),
        (0, 0, 4),
        (0, 1, 0),
        (0, 1, 1),
        (0, 1, 2),
    ]
    assert flat.columns.tolist() == [
        f"{c}_m_x_{m}" for m in [400, 1000, 2000] for c in SCORE_COLUMNS
    ]


def test_flatten_scores_values(
    expanded, models, transformations, parameter_values
):
    # Row order of the input must not matter
    evaluated = _evaluated(expanded, models, transformations).sample(
        frac=1.0, random_state=1
    )

    flat = produce._flatten_scores(evaluated, parameter_values)

    for (_, _, entry), row in flat.iterrows():
        event = {3: 10, 4: 11, 0: 12, 1: 13, 2: 14}[entry]
        max_score, max_score_index = _expected(event)
        for m in [400, 1000, 2000]:
            assert row[f"max_score_m_x_{m}"] == pytest.approx(max_score)
            assert row[f"max_score_index_m_x_{m}"] == max_score_index


# ------------------------------------------------------------ _process_chunk


def test_process_chunk(write_tree, models, transformations, parameter_values):
    main = [
        write_tree(
            "a.root",
            {"event": [10, 11, 12], "pt": [12.0, 18.0, 25.0]},
        ),
        write_tree("b.root", {"event": [13, 14], "pt": [33.0, 39.0]}),
    ]
    (chunk,) = produce.determine_chunks(_sample(main), chunk_size=100)

    df = produce._process_chunk(
        chunk,
        ["event", "pt"],
        "2024_nano_v15",
        INPUT_VARIABLES,
        INPUT_PARAMETERS,
        OUTPUT_VARIABLE,
        OUTPUT_CLASSES,
        parameter_values,
        models,
        transformations,
        N_FOLDS,
        batch_size=4,
    )

    # One row per event with only the flattened scores, in event order
    assert df.index.tolist() == [
        (0, 0, 0),
        (0, 0, 1),
        (0, 0, 2),
        (0, 1, 0),
        (0, 1, 1),
    ]
    assert df.columns.tolist() == produce.score_branch_names(parameter_values)
    events = [10, 11, 12, 13, 14]
    for event, (_, row) in zip(events, df.iterrows()):
        max_score, max_score_index = _expected(event)
        assert row["max_score_m_x_2000"] == pytest.approx(max_score)
        assert row["max_score_index_m_x_2000"] == max_score_index


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


# ----------------------------------------------------------- output writing


def test_score_branch_names():
    parameter_values = pd.DataFrame({"m_x": [400, 1000], "m_y": [100, 300]})

    names = produce.score_branch_names(parameter_values)

    assert names == [
        "max_score_m_x_400_m_y_100",
        "max_score_index_m_x_400_m_y_100",
        "max_score_m_x_1000_m_y_300",
        "max_score_index_m_x_1000_m_y_300",
    ]
    assert produce.score_branch_types(names) == {
        "max_score_m_x_400_m_y_100": np.float32,
        "max_score_index_m_x_400_m_y_100": np.int32,
        "max_score_m_x_1000_m_y_300": np.float32,
        "max_score_index_m_x_1000_m_y_300": np.int32,
    }


BRANCH_TYPES = {"score_a": np.float32, "max_score_index": np.int32}


def _block(chunk, entries, offset=0.0):
    entries = list(entries)
    return pd.DataFrame(
        {
            "score_a": [offset + e / 10 for e in entries],
            "max_score_index": [e % 3 for e in entries],
            "extra": 0.0,
        },
        index=pd.MultiIndex.from_arrays(
            [[chunk] * len(entries), [0] * len(entries), entries],
            names=produce.INDEX_NAMES,
        ),
    )


def test_write_friend_file(tmp_path):
    output_file = tmp_path / "sub" / "dir" / "friend.root"

    produce.write_friend_file(
        output_file,
        BRANCH_TYPES,
        5,
        iter([_block(0, range(3)), _block(1, range(3, 5))]),
    )

    assert output_file.exists()
    assert list(tmp_path.glob("sub/dir/*")) == [output_file]
    with uproot.open(output_file) as f:
        tree = f["ntuple"]
        assert tree.num_entries == 5
        assert sorted(tree.keys()) == sorted(BRANCH_TYPES)
        arrays = tree.arrays(library="np")
    assert arrays["score_a"].dtype == np.float32
    assert arrays["max_score_index"].dtype == np.int32
    np.testing.assert_allclose(arrays["score_a"], [0.0, 0.1, 0.2, 0.3, 0.4])
    np.testing.assert_array_equal(arrays["max_score_index"], [0, 1, 2, 0, 1])


def test_write_friend_file_empty(tmp_path):
    output_file = tmp_path / "friend.root"

    produce.write_friend_file(output_file, BRANCH_TYPES, 0, [])

    with uproot.open(output_file) as f:
        assert f["ntuple"].num_entries == 0
        assert sorted(f["ntuple"].keys()) == sorted(BRANCH_TYPES)


def test_write_friend_file_overwrites(tmp_path):
    output_file = tmp_path / "friend.root"
    produce.write_friend_file(output_file, BRANCH_TYPES, 1, [_block(0, [0])])

    produce.write_friend_file(
        output_file, BRANCH_TYPES, 2, [_block(0, [0, 1])]
    )

    with uproot.open(output_file) as f:
        assert f["ntuple"].num_entries == 2


@pytest.mark.parametrize(
    "blocks, num_entries, error, match",
    [
        # Gap between blocks
        ([_block(0, [0, 1]), _block(1, [3])], 4, ValueError, "continue"),
        # Not starting at entry 0
        ([_block(0, [1, 2])], 3, ValueError, "continue"),
        # Fewer entries than the main tree
        ([_block(0, [0, 1])], 3, ValueError, "main tree has 3"),
        # Branch missing
        (
            [_block(0, [0]).drop(columns="score_a")],
            1,
            KeyError,
            "score_a",
        ),
    ],
)
def test_write_friend_file_invalid(
    tmp_path, blocks, num_entries, error, match
):
    output_file = tmp_path / "friend.root"

    with pytest.raises(error, match=match):
        produce.write_friend_file(
            output_file, BRANCH_TYPES, num_entries, blocks
        )

    # Neither the output file nor the temporary file is left behind
    assert list(tmp_path.iterdir()) == []


def test_write_friend_file_failure_keeps_previous_output(tmp_path):
    output_file = tmp_path / "friend.root"
    produce.write_friend_file(output_file, BRANCH_TYPES, 1, [_block(0, [0])])

    with pytest.raises(ValueError):
        produce.write_friend_file(
            output_file, BRANCH_TYPES, 2, [_block(0, [0])]
        )

    with uproot.open(output_file) as f:
        assert f["ntuple"].num_entries == 1


# ------------------------------------------------------- input file selection

SERVER = "root://server:1094/"


def _metadata_samples():
    def files(prefix, era, sample, channel, n):
        return [
            f"{SERVER}/store/{prefix}/{era}/{sample}/{channel}/f_{i}.root"
            for i in range(n)
        ]

    return [
        Sample(
            era=era,
            sample=sample,
            channel=channel,
            main_files=files("ntuples", era, sample, channel, 2),
            friend_files={
                "kinfit": files("friends/kinfit", era, sample, channel, 2),
                "fastmtt": files("friends/fastmtt", era, sample, channel, 2),
            },
            shifts=[],
        )
        for era in ["2024", "2025"]
        for sample in ["DY", "TT"]
        for channel in ["et", "mt"]
    ]


def test_select_input_file_full_path():
    samples = _metadata_samples()
    input_file = f"{SERVER}/store/ntuples/2025/TT/mt/f_1.root"

    sample = produce.select_input_file(samples, input_file)

    assert (sample.era, sample.sample, sample.channel) == ("2025", "TT", "mt")
    assert sample.main_files == [input_file]
    assert sample.friend_files == {
        "kinfit": [f"{SERVER}/store/friends/kinfit/2025/TT/mt/f_1.root"],
        "fastmtt": [f"{SERVER}/store/friends/fastmtt/2025/TT/mt/f_1.root"],
    }


@pytest.mark.parametrize(
    "input_file",
    ["/store/ntuples/2024/DY/et/f_0.root", "ntuples/2024/DY/et/f_0.root"],
)
def test_select_input_file_trailing_path(input_file):
    sample = produce.select_input_file(_metadata_samples(), input_file)

    assert sample.main_files == [f"{SERVER}/store/ntuples/2024/DY/et/f_0.root"]


def test_select_input_file_keeps_metadata():
    samples = _metadata_samples()
    before = [dataclasses.replace(s) for s in samples]

    produce.select_input_file(samples, "ntuples/2024/DY/et/f_0.root")

    assert samples == before


def test_select_input_file_not_found():
    with pytest.raises(ValueError, match="not found"):
        produce.select_input_file(
            _metadata_samples(), "ntuples/2024/DY/et/f_9.root"
        )


def test_select_input_file_ambiguous():
    with pytest.raises(ValueError, match="matches several"):
        produce.select_input_file(_metadata_samples(), "f_0.root")


def test_select_input_file_partial_file_name_does_not_match():
    with pytest.raises(ValueError, match="not found"):
        produce.select_input_file(
            _metadata_samples(), "ntuples/2024/DY/et/_0.root"
        )


@pytest.mark.parametrize("channel", ["et", "mt"])
def test_select_input_file_channel_from_file(channel):
    sample = produce.select_input_file(
        _metadata_samples(), f"ntuples/2024/DY/{channel}/f_0.root"
    )

    assert sample.channel == channel
    assert sample.friend_files["kinfit"] == [
        f"{SERVER}/store/friends/kinfit/2024/DY/{channel}/f_0.root"
    ]


def test_select_sample_files_all_eras_and_channels():
    files = produce.select_sample_files(_metadata_samples(), "TT")

    assert [s.main_files[0] for s in files] == [
        f"{SERVER}/store/ntuples/{era}/TT/{channel}/f_{i}.root"
        for era in ["2024", "2025"]
        for channel in ["et", "mt"]
        for i in range(2)
    ]
    for s in files:
        assert s.sample == "TT"
        (main_file,) = s.main_files
        assert s.friend_files == {
            name: [main_file.replace("ntuples", f"friends/{name}")]
            for name in ["kinfit", "fastmtt"]
        }


@pytest.mark.parametrize(
    "eras, channels, expected",
    [
        (["2025"], None, [("2025", "et"), ("2025", "mt")]),
        (None, ["mt"], [("2024", "mt"), ("2025", "mt")]),
        (["2024"], ["et"], [("2024", "et")]),
        (["2024", "2025"], ["et"], [("2024", "et"), ("2025", "et")]),
        (["2025"], ["et", "mt"], [("2025", "et"), ("2025", "mt")]),
    ],
)
def test_select_sample_files_restricted(eras, channels, expected):
    files = produce.select_sample_files(
        _metadata_samples(), "DY", eras=eras, channels=channels
    )

    assert sorted({(s.era, s.channel) for s in files}) == expected
    assert len(files) == 2 * len(expected)


def test_select_sample_files_not_found():
    with pytest.raises(
        ValueError, match="Available samples: \\['DY', 'TT'\\]"
    ):
        produce.select_sample_files(_metadata_samples(), "WJets")
    with pytest.raises(ValueError, match="not found"):
        produce.select_sample_files(_metadata_samples(), "DY", eras=["2023"])
    with pytest.raises(ValueError, match="not found"):
        produce.select_sample_files(_metadata_samples(), "DY", channels=["tt"])


CAMPAIGNS = [
    Campaign(name="2024_nano_v15", era="2024"),
    Campaign(name="2025_nano_v15", era="2025"),
]


def test_campaign_eras():
    assert produce.campaign_eras(CAMPAIGNS, ["2025_nano_v15"]) == ["2025"]
    assert produce.campaign_eras(
        CAMPAIGNS, ["2025_nano_v15", "2024_nano_v15"]
    ) == ["2025", "2024"]
    assert produce.campaign_eras(CAMPAIGNS, []) == []


def test_campaign_eras_unknown():
    with pytest.raises(
        ValueError, match="Unknown campaigns \\['2024'\\]. Defined"
    ):
        produce.campaign_eras(CAMPAIGNS, ["2024_nano_v15", "2024"])


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        ("et", ["et"]),
        (["et", "mt"], ["et", "mt"]),
        (OmegaConf.create({"x": ["et"]}).x, ["et"]),
        (OmegaConf.create({"x": [2024]}).x, ["2024"]),
    ],
)
def test_as_list(value, expected):
    assert produce._as_list(value) == expected


def test_default_output_file():
    sample = produce.select_input_file(
        _metadata_samples(), "ntuples/2025/TT/et/f_1.root"
    )

    output_file = produce.default_output_file(Path("out/friend"), sample)

    assert output_file == Path("out/friend/2025/TT/et/f_1.root")


def test_campaign_name():
    campaigns = [
        Campaign("2024_nano_v15", "2024"),
        Campaign("2025_nano_v15", "2025"),
    ]

    assert produce.campaign_name(campaigns, "2025") == "2025_nano_v15"
    with pytest.raises(ValueError, match="era 2023"):
        produce.campaign_name(campaigns, "2023")


# ------------------------------------------------------------- produce_friend


def _produce_friend(sample, output_file, models, transformations, **kwargs):
    return produce.produce_friend(
        sample,
        output_file,
        ["event", "pt"],
        "2024_nano_v15",
        INPUT_VARIABLES,
        INPUT_PARAMETERS,
        OUTPUT_VARIABLE,
        OUTPUT_CLASSES,
        pd.DataFrame({"m_x": [400, 1000]}),
        models,
        transformations,
        N_FOLDS,
        batch_size=4,
        **kwargs,
    )


def test_produce_friend(tmp_path, write_tree, models, transformations):
    events = [10, 11, 12, 13, 14, 15, 16]
    main = write_tree(
        "main.root",
        {"event": events, "pt": np.linspace(10.0, 40.0, len(events))},
    )
    output_file = tmp_path / "out" / "friend.root"

    _produce_friend(
        _sample([main]), output_file, models, transformations, chunk_size=3
    )

    with uproot.open(output_file) as f:
        tree = f["ntuple"]
        assert tree.num_entries == len(events)
        assert tree.keys() == [
            f"{c}_m_x_{m}" for m in [400, 1000] for c in SCORE_COLUMNS
        ]
        arrays = tree.arrays(library="np")

    for entry, event in enumerate(events):
        max_score, max_score_index = _expected(event)
        for m in [400, 1000]:
            assert arrays[f"max_score_m_x_{m}"].dtype == np.float32
            assert arrays[f"max_score_index_m_x_{m}"].dtype == np.int32
            assert arrays[f"max_score_m_x_{m}"][entry] == pytest.approx(
                max_score, rel=1e-6
            )
            assert arrays[f"max_score_index_m_x_{m}"][entry] == max_score_index


def test_produce_friend_reads_inputs_from_friends(
    tmp_path, write_tree, models, transformations
):
    main = write_tree("main.root", {"event": [10, 11, 12]})
    friend = write_tree("kinfit.root", {"pt": [12.0, 18.0, 25.0]})
    output_file = tmp_path / "friend.root"

    _produce_friend(
        _sample([main], {"kinfit": [friend]}),
        output_file,
        models,
        transformations,
        chunk_size=2,
    )

    with uproot.open(output_file) as f:
        assert f["ntuple"].num_entries == 3


@pytest.mark.parametrize("branches", [{}, None])
def test_produce_friend_empty_main_file(
    tmp_path, write_tree, models, transformations, branches
):
    main = write_tree("main.root", branches)
    output_file = tmp_path / "friend.root"

    _produce_friend(
        _sample([main]), output_file, models, transformations, chunk_size=2
    )

    with uproot.open(output_file) as f:
        assert f["ntuple"].num_entries == 0
        assert len(f["ntuple"].keys()) == 2 * len(SCORE_COLUMNS)


def test_produce_friend_profile(tmp_path, write_tree, models, transformations):
    events = list(range(10, 17))
    main = write_tree(
        "main.root",
        {"event": events, "pt": np.linspace(10.0, 40.0, len(events))},
    )

    profiler = _produce_friend(
        _sample([main]),
        tmp_path / "friend.root",
        models,
        transformations,
        chunk_size=3,
    )
    summary = profiler.summary()

    # Two parameter values per event
    assert (summary["n_chunks"], summary["n_events"], summary["n_rows"]) == (
        3,
        7,
        14,
    )
    assert set(summary["stages"]) == {
        "total",
        "load",
        "prepare",
        "evaluate",
        "predict",
        "flatten",
        "write",
    }
    stages = summary["stages"]
    assert stages["predict"]["total_s"] <= stages["evaluate"]["total_s"]
    assert stages["evaluate"]["total_s"] <= stages["total"]["total_s"]
    for stage in stages.values():
        assert stage["per_row_s"] == pytest.approx(stage["total_s"] / 14)
        assert stage["per_event_s"] == pytest.approx(stage["total_s"] / 7)

    memory = summary["memory"]
    # The largest loaded chunk has 3 events, expanded to 6 rows
    assert 0 < memory["loaded_data_frame_max_bytes"]
    assert (
        memory["loaded_data_frame_max_bytes"]
        < memory["expanded_data_frame_max_bytes"]
    )
    assert memory["rss_peak_bytes"] >= memory["rss_start_bytes"]


def test_produce_friend_profile_given_profiler(
    tmp_path, write_tree, models, transformations
):
    main = write_tree("main.root", {"event": [10, 11], "pt": [12.0, 18.0]})
    profiler = produce.Profiler()

    returned = _produce_friend(
        _sample([main]),
        tmp_path / "friend.root",
        models,
        transformations,
        chunk_size=5,
        profiler=profiler,
    )

    assert returned is profiler
    assert profiler.n_events == 2


def test_warm_up_models_moves_models_to_device_and_evaluates():
    calls = []

    class Recorder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(3, 2)

        def forward(self, x):
            calls.append(tuple(x.shape))
            return self.layer(x)

    models = {0: Recorder().train(), 1: Recorder().train()}

    produce._warm_up_models(models, input_size=3)

    assert calls == [(1, 3), (1, 3)]
    for model in models.values():
        assert not model.training
        assert next(model.parameters()).device.type == produce._device()
