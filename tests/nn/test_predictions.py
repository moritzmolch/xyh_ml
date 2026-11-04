"""Tests for the model predictions in ``xyh_ml.nn.predictions``."""

import numpy as np
import pytest
import torch
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, TensorDataset

from xyh_ml.config.dataclasses import Network
from xyh_ml.nn.mlp import create_multi_layer_perceptron_model
from xyh_ml.nn.predictions import predict

N_X, N_P, N_CLASSES = 5, 2, 4

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _reference_predict(model, dataset, batch_size):
    """Previous implementation of predict with a DataLoader."""
    model = model.eval()
    pred = []
    for x, p in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        with torch.no_grad():
            pred.append(softmax(model(torch.cat((x, p), dim=-1)), dim=-1))
    return torch.cat(pred, dim=0).numpy()


@pytest.fixture
def model():
    torch.manual_seed(0)
    # Dropout makes outputs in training mode random, which checks that the
    # model is evaluated in evaluation mode
    return create_multi_layer_perceptron_model(
        N_X + N_P,
        N_CLASSES,
        Network(hidden_layers=[16, 16], activation="relu", dropout_frac=0.5),
    ).train()


def _dataset(n):
    generator = torch.Generator().manual_seed(1)
    return TensorDataset(
        torch.rand(n, N_X, generator=generator),
        torch.rand(n, N_P, generator=generator),
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "n, batch_size", [(10, 3), (10, 10), (10, 64), (1, 1)]
)
def test_predict_matches_data_loader_implementation(
    model, device, n, batch_size
):
    dataset = _dataset(n)

    pred = predict(model, dataset, device, batch_size=batch_size)

    expected = _reference_predict(model.cpu(), dataset, batch_size)
    assert isinstance(pred, np.ndarray)
    assert pred.shape == (n, N_CLASSES)
    np.testing.assert_allclose(pred, expected, rtol=1e-5, atol=1e-6)


def test_predict_does_not_depend_on_batch_size(model):
    dataset = _dataset(25)

    pred_small = predict(model, dataset, "cpu", batch_size=4)
    pred_large = predict(model, dataset, "cpu", batch_size=1000)

    np.testing.assert_allclose(pred_small, pred_large, rtol=1e-5, atol=1e-6)


def test_predict_softmax_and_evaluation_mode(model):
    pred = predict(model, _dataset(20), "cpu", batch_size=7)

    assert not model.training
    np.testing.assert_allclose(pred.sum(axis=1), 1.0, rtol=1e-6)
    assert np.all(pred >= 0.0)


def test_predict_order_of_events(model):
    """Each row of the output belongs to the same row of the input."""
    dataset = _dataset(9)

    pred = predict(model, dataset, "cpu", batch_size=2)

    x, p = dataset.tensors
    for i in [0, 4, 8]:
        single = TensorDataset(x[i : i + 1], p[i : i + 1])
        np.testing.assert_allclose(
            pred[i : i + 1],
            predict(model, single, "cpu", batch_size=1),
            rtol=1e-5,
            atol=1e-6,
        )


def test_predict_empty_dataset(model):
    pred = predict(model, _dataset(0), "cpu", batch_size=4)

    assert pred.shape == (0, N_CLASSES)


@pytest.mark.parametrize("batch_size", [0, -1])
def test_predict_invalid_batch_size(model, batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        predict(model, _dataset(3), "cpu", batch_size=batch_size)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_predict_on_cuda_returns_numpy_on_host(model):
    pred = predict(model, _dataset(5), "cuda", batch_size=2)

    assert isinstance(pred, np.ndarray)
    assert next(model.parameters()).device.type == "cuda"
