"""Tests for the torch dataset helpers in ``xyh_ml.data.torch``."""

import numpy as np
import pandas as pd
import pytest

from xyh_ml.config.dataclasses import Variable
from xyh_ml.data.torch import create_torch_dataset_for_evaluation
from xyh_ml.data.transformations import derive_transformations

INPUT_VARIABLES = [
    Variable("pt", "standard_scaling"),
    Variable("n_jets", "none"),
    Variable("campaign", "ordinal_encoding", is_part_of_snapshot=False),
]
INPUT_PARAMETERS = [
    Variable("m_x", "standard_scaling", is_part_of_snapshot=False),
]
OUTPUT_VARIABLES = [Variable("class", "ordinal_encoding")]


@pytest.fixture
def transformations():
    """Transformations fitted like in training, i.e. on inputs, parameters
    and the output variable."""
    train = pd.DataFrame(
        {
            "pt": [10.0, 20.0, 30.0, 40.0],
            "n_jets": [1, 2, 3, 4],
            "campaign": ["2024", "2025", "2024", "2025"],
            "m_x": [400.0, 400.0, 1000.0, 1000.0],
            "class": ["sig", "bkg", "sig", "bkg"],
        }
    )
    return derive_transformations(
        train, INPUT_VARIABLES + INPUT_PARAMETERS + OUTPUT_VARIABLES
    )


@pytest.fixture
def events():
    """Events to evaluate; no output variable column is available."""
    return pd.DataFrame(
        {
            "event": [1, 2],
            "pt": [25.0, 35.0],
            "n_jets": [2, 5],
            "campaign": ["2025", "2024"],
            "m_x": [400.0, 1000.0],
        }
    )


def test_create_torch_dataset_for_evaluation_without_output_column(
    transformations, events
):
    dataset = create_torch_dataset_for_evaluation(
        events, INPUT_VARIABLES, INPUT_PARAMETERS, transformations
    )
    x, p = dataset.tensors

    # pt: mean 25, std sqrt(125); n_jets untransformed; campaign encoded
    # with the categories fitted in training (2024 -> 0, 2025 -> 1)
    expected_x = np.array(
        [
            [0.0, 2.0, 1.0],
            [10.0 / np.sqrt(125.0), 5.0, 0.0],
        ],
        dtype=np.float32,
    )
    # m_x: mean 700, std 300
    expected_p = np.array([[-1.0], [1.0]], dtype=np.float32)

    np.testing.assert_allclose(x.numpy(), expected_x, rtol=1e-6)
    np.testing.assert_allclose(p.numpy(), expected_p, rtol=1e-6)


def test_create_torch_dataset_for_evaluation_keeps_input(
    transformations, events
):
    before = events.copy()

    create_torch_dataset_for_evaluation(
        events, INPUT_VARIABLES, INPUT_PARAMETERS, transformations
    )

    pd.testing.assert_frame_equal(events, before)


def test_create_torch_dataset_for_evaluation_with_output_column(
    transformations, events
):
    # An output column in the data frame must not change the result
    with_class = events.assign(**{"class": ["bkg", "sig"]})

    x_with, p_with = create_torch_dataset_for_evaluation(
        with_class, INPUT_VARIABLES, INPUT_PARAMETERS, transformations
    ).tensors
    x, p = create_torch_dataset_for_evaluation(
        events, INPUT_VARIABLES, INPUT_PARAMETERS, transformations
    ).tensors

    np.testing.assert_array_equal(x_with.numpy(), x.numpy())
    np.testing.assert_array_equal(p_with.numpy(), p.numpy())
