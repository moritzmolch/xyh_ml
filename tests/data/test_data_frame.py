"""Tests for the data frame helpers in ``xyh_ml.data.data_frame``."""

import numpy as np
import pandas as pd
import pytest

from xyh_ml.data.data_frame import prepare_with_parameter_values


@pytest.fixture
def events():
    """Three events with a (chunk, slice, entry) index."""
    return pd.DataFrame(
        {"event": [10, 11, 12], "pt": [1.0, 2.0, 3.0]},
        index=pd.MultiIndex.from_tuples(
            [(0, 0, 5), (0, 1, 0), (0, 1, 1)],
            names=["chunk", "slice", "entry"],
        ),
    )


@pytest.fixture
def parameter_values():
    return pd.DataFrame({"m_x": [400, 1000], "m_y": [100, 300]})


def test_prepare_with_parameter_values_rows(events, parameter_values):
    df = prepare_with_parameter_values(events, parameter_values)

    # Every event is repeated once per parameter value combination, keeping
    # the rows of one event together
    assert len(df) == 6
    assert df["event"].tolist() == [10, 10, 11, 11, 12, 12]
    assert df["pt"].tolist() == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    assert df["m_x"].tolist() == [400, 1000] * 3
    assert df["m_y"].tolist() == [100, 300] * 3


def test_prepare_with_parameter_values_index(events, parameter_values):
    df = prepare_with_parameter_values(events, parameter_values)

    assert df.index.names == ["chunk", "slice", "entry", "parameter_value"]
    assert df.index.tolist() == [
        (0, 0, 5, 0),
        (0, 0, 5, 1),
        (0, 1, 0, 0),
        (0, 1, 0, 1),
        (0, 1, 1, 0),
        (0, 1, 1, 1),
    ]
    assert df.index.is_unique


def test_prepare_with_parameter_values_keeps_input(events, parameter_values):
    before = events.copy()

    prepare_with_parameter_values(events, parameter_values)

    pd.testing.assert_frame_equal(events, before)


def test_prepare_with_parameter_values_single_index():
    events = pd.DataFrame({"pt": [1.0, 2.0]})
    parameter_values = pd.DataFrame({"m_x": [400, 1000, 2000]})

    df = prepare_with_parameter_values(events, parameter_values)

    assert df.index.names == [None, "parameter_value"]
    assert df["m_x"].tolist() == [400, 1000, 2000] * 2
    np.testing.assert_array_equal(
        df.index.get_level_values("parameter_value"), [0, 1, 2] * 2
    )


def test_prepare_with_parameter_values_no_events(parameter_values):
    events = pd.DataFrame(
        {"pt": pd.Series([], dtype=float)},
        index=pd.MultiIndex.from_arrays(
            [[], [], []], names=["chunk", "slice", "entry"]
        ),
    )

    df = prepare_with_parameter_values(events, parameter_values)

    assert df.empty
    assert df.columns.tolist() == ["pt", "m_x", "m_y"]
