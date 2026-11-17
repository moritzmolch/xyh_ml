"""
Helpers for the Taylor coefficient (TC) analysis of the trained classifier.

The TCs of the softmax output node of class `c` with respect to the inputs
`x_i` (input variables, and the input parameters only with
`taylor.include_parameters`, transformed as they are fed to the network) are
stored per event in columns

- `tc__<c>__<x_i>` (first order, d y_c / d x_i), and
- `tc__<c>__<x_i>__<x_j>` with i <= j (second order,
  1 / (n_i! n_j!) d^2 y_c / d x_i d x_j, i.e. with a factor 1/2 for i == j).
"""

import logging
from pathlib import Path
from time import time

import numpy as np
import torch
from pandas import DataFrame, MultiIndex
from tayloranalysis import extend_model

logger = logging.getLogger(__name__)

TC_PREFIX = "tc"
TC_SEPARATOR = "__"


def get_tc_index_list(
    n_inputs: int,
    max_order: int = 2,
    selected: list[int] | None = None,
) -> list[tuple]:
    """
    Get the input index combinations of all TCs up to `max_order`: first all
    `(i,)`, then all `(i, j)` with `i <= j`. If `selected` is given, only the
    inputs with these indices are used (e.g. to skip the input parameters).
    """
    if max_order not in (1, 2):
        raise ValueError(f"max_order must be 1 or 2, got {max_order}")
    if selected is None:
        selected = list(range(n_inputs))
    selected = sorted(set(selected))
    if any(i < 0 or i >= n_inputs for i in selected):
        raise ValueError(f"Selected inputs {selected} out of range")
    indices = [(i,) for i in selected]
    if max_order >= 2:
        indices += [
            (i, j) for k, i in enumerate(selected) for j in selected[k:]
        ]
    return indices


def get_tc_column(
    class_name: str, index: tuple, input_names: list[str]
) -> str:
    """
    Get the data frame column name of the TC of the output node `class_name`
    for the input index combination `index`.
    """
    return TC_SEPARATOR.join(
        [TC_PREFIX, class_name] + [input_names[i] for i in index]
    )


def parse_tc_column(column: str) -> tuple[str, tuple[str, ...]]:
    """
    Inverse of `get_tc_column`: return the output class name and the tuple of
    input names of a TC column.
    """
    parts = column.split(TC_SEPARATOR)
    if len(parts) < 3 or parts[0] != TC_PREFIX:
        raise ValueError(f"'{column}' is not a Taylor coefficient column")
    return parts[1], tuple(parts[2:])


def get_tc_columns(columns, class_name: str) -> list[str]:
    """
    Select the TC columns of the output node `class_name` from `columns`.
    """
    prefix = f"{TC_PREFIX}{TC_SEPARATOR}{class_name}{TC_SEPARATOR}"
    return [c for c in columns if c.startswith(prefix)]


def drop_tc_columns_with_inputs(
    columns: list[str], input_names: list[str]
) -> list[str]:
    """
    Remove the TC columns from `columns` that involve any of `input_names`
    (e.g. the input parameters m_x and m_y).
    """
    excluded = set(input_names)
    return [c for c in columns if not excluded & set(parse_tc_column(c)[1])]


def get_tc_label(input_names: tuple[str, ...]) -> str:
    """
    Get a plain-text tick label of a TC: the input name for first order,
    e.g. `m_vis`, and the input names joined by `×` for second order, e.g.
    `m_vis × m_fastmtt`.
    """
    return " \u00d7 ".join(input_names)


def get_taylor_coefficients_file(cfg) -> Path:
    """
    Get the path of the HDF5 file with the TCs of the configuration `cfg`.
    """
    return (
        Path(cfg.artifacts.taylor_dir)
        / f"taylor_coefficients_{cfg.mode.decay_mode}.h5"
    )


def compute_taylor_coefficients(
    model: torch.nn.Module,
    dataset: torch.utils.data.TensorDataset,
    input_names: list[str],
    class_names: list[str],
    device: str,
    batch_size: int = 4096,
    max_order: int = 2,
    tc_input_names: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """
    Compute the per-event TCs of the softmax outputs of `model`.

    Parameters
    ----------
    model
        Network returning logits for inputs `cat((x, p), -1)`.
    dataset
        Dataset yielding `(x, p, y, w, r)`.
    input_names
        Names of the input variables followed by the input parameters.
    class_names
        Names of the output nodes.
    device
        Device on which the gradients are computed.
    batch_size
        Number of events per batch.
    max_order
        Maximum order of the coefficients (1 or 2).
    tc_input_names
        Subset of `input_names` the TCs are computed for (default: all). The
        network is still evaluated on all inputs.

    Returns
    -------
    dict[str, np.ndarray]
        Float32 arrays of length `len(dataset)` keyed by the TC column names.
    """
    n_inputs = len(input_names)
    selected = None
    if tc_input_names is not None:
        unknown = set(tc_input_names) - set(input_names)
        if unknown:
            raise ValueError(f"Unknown TC inputs: {sorted(unknown)}")
        selected = [input_names.index(n) for n in tc_input_names]
    tc_index_list = get_tc_index_list(n_inputs, max_order, selected)
    wrapped = torch.nn.Sequential(model, torch.nn.Softmax(dim=-1))
    wrapped = extend_model(wrapped).eval().to(device)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    chunks = {
        get_tc_column(c, idx, input_names): []
        for c in class_names
        for idx in tc_index_list
    }
    start = time()
    for x, p, _, _, _ in data_loader:
        base = torch.cat((x, p), dim=-1).to(device)
        for i, class_name in enumerate(class_names):
            # get_tc sets requires_grad on the input, so use a fresh tensor
            tcs = wrapped.get_tc(
                "input",
                {"input": base.clone()},
                tc_index_list,
                selected_output_node=i,
                eval_max_output_node_only=False,
                reduce_func=lambda t: t.detach().cpu(),
            )
            for idx in tc_index_list:
                tc = tcs[idx]
                column = get_tc_column(class_name, idx, input_names)
                chunks[column].append(tc.reshape(-1).float().numpy())

    result = {
        k: np.concatenate(v).astype(np.float32) for k, v in chunks.items()
    }
    delta = round(time() - start, 2)
    logger.debug(
        f"Computed {len(result)} TCs for {len(dataset)} events in {delta} s"
    )
    return result


def select_benchmark_mass_points(
    data_frame: DataFrame,
    unique_parameter_values: DataFrame,
    mass_points: list,
) -> tuple[DataFrame, DataFrame]:
    """
    Restrict the evaluation to benchmark mass points.

    Signal events whose `(m_x, m_y)` is not a benchmark are dropped (pair-wise
    match), background events are kept.

    Parameters
    ----------
    data_frame
        Events with columns `is_signal`, `m_x` and `m_y`.
    unique_parameter_values
        Available parameter combinations; columns must be `m_x` and `m_y`.
    mass_points
        List of `[m_x, m_y]` benchmarks.

    Returns
    -------
    tuple[DataFrame, DataFrame]
        The filtered data frame and the benchmark parameter combinations.

    Raises
    ------
    ValueError
        If the parameters are not `m_x`, `m_y` or benchmarks are unavailable.
    """
    if set(unique_parameter_values.columns) != {"m_x", "m_y"}:
        raise ValueError(
            "Expected parameters m_x and m_y, got "
            f"{list(unique_parameter_values.columns)}"
        )
    benchmarks = DataFrame(
        np.asarray(mass_points).reshape(-1, 2), columns=["m_x", "m_y"]
    ).astype(unique_parameter_values[["m_x", "m_y"]].dtypes.to_dict())

    available = set(
        map(tuple, unique_parameter_values[["m_x", "m_y"]].to_numpy())
    )
    missing = [
        tuple(m) for m in benchmarks.to_numpy() if tuple(m) not in available
    ]
    if missing:
        raise ValueError(f"Benchmark mass points not available: {missing}")

    is_benchmark = MultiIndex.from_frame(data_frame[["m_x", "m_y"]]).isin(
        list(map(tuple, benchmarks.to_numpy()))
    )
    keep = ~data_frame["is_signal"].to_numpy(dtype=bool) | is_benchmark
    return data_frame.loc[keep].copy(), benchmarks


def iter_diagonal_sets(
    data_frame: DataFrame,
    class_names: list[str],
    excluded_inputs: list[str] | None = None,
):
    """
    Iterate over the "diagonal" TC sets of `data_frame`: for each channel,
    mass point and class `c`, the events of true and predicted class `c` and
    the TC columns of the output node `c`.

    Parameters
    ----------
    data_frame
        Events with the columns `channel`, `m_x`, `m_y`, `true_class_index`,
        `predicted_class_index` (see `xyh_ml.plotting.add_class_indices`,
        indices in the order of `class_names`) and the TC columns.
    class_names
        Output class names in the order of the class indices.
    excluded_inputs
        Drop the TC columns that involve any of these inputs (e.g. m_x, m_y).

    Yields
    ------
    tuple
        `(channel, m_x, m_y, class_name, events, columns)`; sets without TC
        columns or without events are skipped.
    """
    for (channel, m_x, m_y), df_point in data_frame.groupby(
        ["channel", "m_x", "m_y"]
    ):
        point = f"{channel}, m_x={m_x:g}, m_y={m_y:g}"
        for i, class_name in enumerate(class_names):
            columns = get_tc_columns(df_point.columns, class_name)
            if excluded_inputs:
                columns = drop_tc_columns_with_inputs(columns, excluded_inputs)
            if not columns:
                logger.info(f"No coefficients of node {class_name}, skipping")
                continue
            events = df_point[
                (df_point["true_class_index"] == i)
                & (df_point["predicted_class_index"] == i)
            ]
            if len(events) == 0:
                logger.info(
                    f"No events of true and predicted class {class_name} "
                    f"for {point}, skipping"
                )
                continue
            yield channel, m_x, m_y, class_name, events, columns
