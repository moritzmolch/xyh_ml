import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from xyh_ml.cli.evaluate_test_sample import (
    filter_and_set_evaluation_parameters,
)
from xyh_ml.nn.mlp import MultiLayerPerceptron
from xyh_ml.taylor import (
    compute_taylor_coefficients,
    drop_tc_columns_with_inputs,
    get_taylor_coefficients_file,
    get_tc_column,
    get_tc_columns,
    get_tc_index_list,
    get_tc_label,
    parse_tc_column,
    select_benchmark_mass_points,
)

# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 3, 5])
def test_get_tc_index_list_sizes(n):
    assert len(get_tc_index_list(n, 1)) == n
    assert len(get_tc_index_list(n, 2)) == n + n * (n + 1) // 2


def test_get_tc_index_list_ordering():
    assert get_tc_index_list(3, 1) == [(0,), (1,), (2,)]
    assert get_tc_index_list(2, 2) == [(0,), (1,), (0, 0), (0, 1), (1, 1)]


def test_get_tc_index_list_default_is_second_order():
    assert get_tc_index_list(2) == get_tc_index_list(2, 2)


@pytest.mark.parametrize("max_order", [0, 3, -1])
def test_get_tc_index_list_invalid_order_raises(max_order):
    with pytest.raises(ValueError, match="max_order"):
        get_tc_index_list(2, max_order)


@pytest.mark.parametrize(
    "index, column",
    [
        ((0,), "tc__xyh__pt"),
        ((1, 1), "tc__xyh__eta__eta"),
        ((0, 1), "tc__xyh__pt__eta"),
    ],
)
def test_tc_column_round_trip(index, column):
    names = ["pt", "eta"]
    assert get_tc_column("xyh", index, names) == column
    assert parse_tc_column(column) == (
        "xyh",
        tuple(names[i] for i in index),
    )


@pytest.mark.parametrize("column", ["score_xyh", "tc__xyh", "foo__xyh__pt"])
def test_parse_tc_column_non_tc_raises(column):
    with pytest.raises(ValueError, match="not a Taylor coefficient"):
        parse_tc_column(column)


def test_get_tc_columns_selects_single_node_without_prefix_collision():
    names = ["a", "b"]
    columns = ["class", "score_t"]
    for node in ("t", "t_tt"):
        columns += [
            get_tc_column(node, i, names) for i in get_tc_index_list(2, 2)
        ]

    selected_t = get_tc_columns(columns, "t")
    selected_t_tt = get_tc_columns(columns, "t_tt")

    assert len(selected_t) == len(selected_t_tt) == 5
    assert all(parse_tc_column(c)[0] == "t" for c in selected_t)
    assert all(parse_tc_column(c)[0] == "t_tt" for c in selected_t_tt)
    assert get_tc_columns(columns, "missing") == []


def test_get_tc_label_plain_text():
    assert get_tc_label(("pt_1",)) == "pt_1"
    assert get_tc_label(("pt_1", "m_x")) == "pt_1 \u00d7 m_x"


def test_get_taylor_coefficients_file(tmp_path):
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "artifacts": {"taylor_dir": str(tmp_path)},
            "mode": {"decay_mode": "y2b_h2tau"},
        }
    )
    assert (
        get_taylor_coefficients_file(cfg)
        == tmp_path / "taylor_coefficients_y2b_h2tau.h5"
    )


# ---------------------------------------------------------------------------
# compute_taylor_coefficients
# ---------------------------------------------------------------------------

N_X, N_P, N_CLASSES = 2, 1, 3
INPUT_NAMES = ["a", "b", "p"]
CLASS_NAMES = ["c0", "c1", "c2"]


class TanhNet(nn.Module):
    """Smooth network with nontrivial second derivatives."""

    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(N_X + N_P, 5)
        self.l2 = nn.Linear(5, N_CLASSES)

    def forward(self, x):
        return self.l2(torch.tanh(self.l1(x)))


def _dataset(n=7, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, N_X, generator=g)
    p = torch.randn(n, N_P, generator=g)
    y = torch.zeros(n, dtype=torch.long)
    w = torch.ones(n)
    r = torch.ones(n)
    return TensorDataset(x, p, y, w, r)


def _tanh_model():
    torch.manual_seed(0)
    return TanhNet().double().float()


def _reference(model, dataset, node):
    """Jacobian / hessian of softmax node per event (float64)."""
    model = model.double()

    def f(z):
        return torch.softmax(model(z), dim=-1)[node]

    jac, hes = [], []
    for x, p, *_ in dataset:
        z = torch.cat((x, p)).double()
        jac.append(torch.autograd.functional.jacobian(f, z))
        hes.append(torch.autograd.functional.hessian(f, z))
    return torch.stack(jac).numpy(), torch.stack(hes).numpy()


def test_compute_taylor_coefficients_matches_autograd():
    model = _tanh_model()
    dataset = _dataset()

    result = compute_taylor_coefficients(
        model, dataset, INPUT_NAMES, CLASS_NAMES, "cpu"
    )
    ref = {}
    for node, name in enumerate(CLASS_NAMES):
        jac, hes = _reference(_tanh_model(), dataset, node)
        for idx in get_tc_index_list(3, 2):
            col = get_tc_column(name, idx, INPUT_NAMES)
            if len(idx) == 1:
                ref[col] = jac[:, idx[0]]
            elif idx[0] == idx[1]:
                ref[col] = hes[:, idx[0], idx[0]] / 2
            else:
                ref[col] = hes[:, idx[0], idx[1]]

    assert set(result) == set(ref)
    assert len(result) == 3 * (3 + 6)
    for col, expected in ref.items():
        np.testing.assert_allclose(
            result[col], expected, atol=1e-5, err_msg=col
        )
    # Guard against trivially vanishing hessians
    assert np.abs(result["tc__c0__a__b"]).max() > 1e-4


def test_compute_taylor_coefficients_mlp_matches_autograd():
    torch.manual_seed(3)
    model = MultiLayerPerceptron(3, N_CLASSES, [6, 4], "relu", 0.0)
    dataset = _dataset()

    result = compute_taylor_coefficients(
        model, dataset, INPUT_NAMES, CLASS_NAMES, "cpu"
    )

    for node, name in enumerate(CLASS_NAMES):
        jac, hes = _reference(model, dataset, node)
        for i, n in enumerate(INPUT_NAMES):
            np.testing.assert_allclose(
                result[f"tc__{name}__{n}"], jac[:, i], atol=1e-5
            )
        np.testing.assert_allclose(
            result[f"tc__{name}__a__b"], hes[:, 0, 1], atol=1e-5
        )
        np.testing.assert_allclose(
            result[f"tc__{name}__p__p"], hes[:, 2, 2] / 2, atol=1e-5
        )


@pytest.mark.parametrize("batch_size", [1, 3, 7, 100])
def test_compute_taylor_coefficients_batch_size_invariant(batch_size):
    model = _tanh_model()
    dataset = _dataset()

    full = compute_taylor_coefficients(
        model,
        dataset,
        INPUT_NAMES,
        CLASS_NAMES,
        "cpu",
        batch_size=len(dataset),
    )
    batched = compute_taylor_coefficients(
        model, dataset, INPUT_NAMES, CLASS_NAMES, "cpu", batch_size=batch_size
    )

    assert full.keys() == batched.keys()
    for col in full:
        np.testing.assert_allclose(batched[col], full[col], atol=1e-6)


def test_compute_taylor_coefficients_max_order_one():
    result = compute_taylor_coefficients(
        _tanh_model(),
        _dataset(),
        INPUT_NAMES,
        CLASS_NAMES,
        "cpu",
        max_order=1,
    )

    assert len(result) == 3 * 3
    assert all(len(parse_tc_column(c)[1]) == 1 for c in result)


def test_compute_taylor_coefficients_dtype_and_length():
    dataset = _dataset(n=5)
    result = compute_taylor_coefficients(
        _tanh_model(), dataset, INPUT_NAMES, CLASS_NAMES, "cpu", batch_size=2
    )

    for values in result.values():
        assert values.dtype == np.float32
        assert values.shape == (len(dataset),)


def test_compute_taylor_coefficients_dropout_is_deterministic():
    torch.manual_seed(5)
    model = MultiLayerPerceptron(3, N_CLASSES, [16, 16], "relu", 0.5)
    model.train()
    dataset = _dataset()

    first = compute_taylor_coefficients(
        model, dataset, INPUT_NAMES, CLASS_NAMES, "cpu", max_order=1
    )
    second = compute_taylor_coefficients(
        model, dataset, INPUT_NAMES, CLASS_NAMES, "cpu", max_order=1
    )

    for col in first:
        np.testing.assert_array_equal(first[col], second[col])


def test_compute_taylor_coefficients_softmax_derivatives_sum_to_zero():
    # Softmax outputs sum to one, so the first-order TCs of all nodes cancel
    result = compute_taylor_coefficients(
        _tanh_model(), _dataset(), INPUT_NAMES, CLASS_NAMES, "cpu", max_order=1
    )
    for n in INPUT_NAMES:
        total = sum(result[f"tc__{c}__{n}"] for c in CLASS_NAMES)
        np.testing.assert_allclose(total, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# select_benchmark_mass_points
# ---------------------------------------------------------------------------

SIGNAL_POINTS = [(300.0, 60.0), (300.0, 150.0), (500.0, 60.0), (500.0, 150.0)]


@pytest.fixture
def events():
    rows = []
    for k, (mx, my) in enumerate(SIGNAL_POINTS):
        for i in range(3):
            rows.append(("xyh", True, mx, my, 10 * k + i))
    for i in range(4):
        rows.append(("z2tau", False, -10.0, -10.0, 100 + i))
    return pd.DataFrame(
        rows, columns=["class", "is_signal", "m_x", "m_y", "event"]
    )


@pytest.fixture
def unique_parameters():
    return pd.DataFrame(SIGNAL_POINTS, columns=["m_x", "m_y"])


def test_select_benchmark_mass_points_pairwise(events, unique_parameters):
    benchmarks = [[300, 150], [500, 60]]

    df, bench = select_benchmark_mass_points(
        events, unique_parameters, benchmarks
    )

    signal = df[df["is_signal"]]
    assert set(map(tuple, signal[["m_x", "m_y"]].to_numpy())) == {
        (300.0, 150.0),
        (500.0, 60.0),
    }
    assert len(signal) == 6
    assert (~df["is_signal"]).sum() == 4
    assert bench.to_numpy().tolist() == [[300, 150], [500, 60]]
    assert list(bench.columns) == ["m_x", "m_y"]
    # input is not modified
    assert len(events) == 16


def test_select_benchmark_mass_points_unavailable_raises(
    events, unique_parameters
):
    with pytest.raises(ValueError, match="not available"):
        select_benchmark_mass_points(
            events, unique_parameters, [[300, 60], [301, 60]]
        )


def test_select_benchmark_mass_points_pair_not_available_raises(
    events, unique_parameters
):
    # Both values exist individually but not as a pair
    unique = unique_parameters.iloc[[0, 3]]
    with pytest.raises(ValueError, match="not available"):
        select_benchmark_mass_points(events, unique, [[300, 150]])


def test_select_benchmark_mass_points_wrong_parameters_raises(events):
    unique = pd.DataFrame({"m_x": [300.0], "other": [1.0]})
    with pytest.raises(ValueError, match="m_x and m_y"):
        select_benchmark_mass_points(events, unique, [[300, 1]])


def test_benchmarks_combined_with_filter_and_set_evaluation_parameters(
    events, unique_parameters
):
    benchmarks = [[300, 150], [500, 60]]
    df, bench = select_benchmark_mass_points(
        events, unique_parameters, benchmarks
    )

    out = filter_and_set_evaluation_parameters(df, bench, n_folds=3)

    signal = out[out["is_signal"]]
    assert set(map(tuple, signal[["m_x", "m_y"]].to_numpy())) == {
        (300.0, 150.0),
        (500.0, 60.0),
    }
    background = out[~out["is_signal"]]
    assert len(background) == 4 * len(benchmarks)
    for mx, my in benchmarks:
        sel = background[(background["m_x"] == mx) & (background["m_y"] == my)]
        assert sorted(sel["event"]) == [100, 101, 102, 103]


def test_filter_and_set_evaluation_parameters_matches_pairs(
    events, unique_parameters
):
    # Regression test: m_x = 300 and m_y = 60 both occur in the parameter
    # combinations below, but the pair (300, 60) does not, so signal at
    # (300, 60) (and (500, 150)) must be dropped.
    combinations = pd.DataFrame({"m_x": [300.0, 500.0], "m_y": [150.0, 60.0]})
    assert {(300.0, 60.0), (500.0, 150.0)} <= set(
        map(tuple, events.loc[events["is_signal"], ["m_x", "m_y"]].to_numpy())
    )

    out = filter_and_set_evaluation_parameters(events, combinations, n_folds=3)

    signal = out[out["is_signal"]]
    assert set(map(tuple, signal[["m_x", "m_y"]].to_numpy())) == {
        (300.0, 150.0),
        (500.0, 60.0),
    }
    # Signal events of the kept pairs are all retained
    expected = events[
        events["is_signal"]
        & (
            ((events["m_x"] == 300.0) & (events["m_y"] == 150.0))
            | ((events["m_x"] == 500.0) & (events["m_y"] == 60.0))
        )
    ]
    assert len(signal) == len(expected)


# ---------------------------------------------------------------------------
# Exclusion of the input parameters
# ---------------------------------------------------------------------------


def test_get_tc_index_list_selected_inputs():
    assert get_tc_index_list(4, 2, selected=[0, 2]) == [
        (0,),
        (2,),
        (0, 0),
        (0, 2),
        (2, 2),
    ]
    assert get_tc_index_list(4, 1, selected=[3, 1]) == [(1,), (3,)]
    with pytest.raises(ValueError, match="out of range"):
        get_tc_index_list(2, 2, selected=[2])


def test_compute_taylor_coefficients_subset_of_inputs():
    # TCs w.r.t. the variables only are the corresponding subset of all TCs;
    # the network is still evaluated with the parameter "p"
    model, dataset = _tanh_model(), _dataset()
    full = compute_taylor_coefficients(
        model, dataset, INPUT_NAMES, CLASS_NAMES, "cpu"
    )
    subset = compute_taylor_coefficients(
        model,
        dataset,
        INPUT_NAMES,
        CLASS_NAMES,
        "cpu",
        tc_input_names=["a", "b"],
    )

    assert set(subset) == set(drop_tc_columns_with_inputs(list(full), ["p"]))
    assert len(subset) == len(CLASS_NAMES) * (2 + 3)
    assert not any("p" in parse_tc_column(c)[1] for c in subset)
    for column, values in subset.items():
        np.testing.assert_allclose(values, full[column], rtol=1e-6)


def test_compute_taylor_coefficients_unknown_tc_input_raises():
    with pytest.raises(ValueError, match="Unknown TC inputs"):
        compute_taylor_coefficients(
            _tanh_model(),
            _dataset(),
            INPUT_NAMES,
            CLASS_NAMES,
            "cpu",
            tc_input_names=["a", "m_x"],
        )


def test_drop_tc_columns_with_inputs():
    columns = [
        "tc__xyh__pt_1",
        "tc__xyh__m_x",
        "tc__xyh__pt_1__m_y",
        "tc__xyh__m_x__m_y",
        "tc__xyh__m_vis__m_fastmtt",
        "tc__xyh__m_xx",
    ]
    assert drop_tc_columns_with_inputs(columns, ["m_x", "m_y"]) == [
        "tc__xyh__pt_1",
        "tc__xyh__m_vis__m_fastmtt",
        "tc__xyh__m_xx",
    ]
