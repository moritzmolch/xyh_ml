"""Tests for the functions of the friend production CLI script."""

import numpy as np
import pytest
import uproot

from xyh_ml.cli.friends import produce
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
