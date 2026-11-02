import logging
from typing import Any

import numpy as np
import torch
from pandas import DataFrame
from torch.utils.data import TensorDataset

from xyh_ml.config.dataclasses import Variable
from xyh_ml.data.transformations import apply_transformations

# Set up logging
logger = logging.getLogger(__name__)


def create_torch_dataset(
    df: DataFrame,
    input_variables: list[Variable],
    input_parameters: list[Variable],
    output_variable: Variable,
    weight: str,
    transformations: dict[str, dict[str, Any]],
) -> TensorDataset:
    """
    Prepare the training dataset by applying transformations and creating torch
    tensors from the data frame. The dataset is returned as a `TensorDataset`
    object, which can be used for efficient training and validation of `torch`
    models.

    Transformations are applied to input and output variables according to the
    `transformations` dictionary.

    The torch dataset has the following attributes, which are returned in the
    order specified below:
    - `x`: Two-dimensional tensor of input variables
    - `p`: Two-dimensional tensor of input parameters
    - `y`: Two-dimensional tensor of output targets
    - `w`: One-dimensional tensor of event weights
    - `r`: One-dimensional tensor of boolean values indicating whether the
           event is a background event (`True`).

    Parameters
    ----------
    df : DataFrame
        Data frame to prepare the dataset from.

    input_variables : list[Variable]
        List of input variables to use for training/evaluation.

    input_parameters : list[Variable]
        List of input parameters to use for training/evaluation.

    output_variable : Variable
        Output target variable to use for training/evaluation.

    weight : str
        Name of the event weight column.

    transformations : dict
        Dictionary containing the fitted transformation objects and the
        corresponding variable names. For more information, see the
        `derive_transformations` function.

    Returns
    -------
    TensorDataset
        Torch dataset containing the input variables, input parameters, output
        targets, event weights, and background event indicators as torch
        tensors.
    """

    # Get column names
    input_variable_names = [v.name for v in input_variables]
    input_parameter_names = [v.name for v in input_parameters]
    output_variable_name = output_variable.name
    logger.debug(
        f"Preparing torch dataset with input variables {input_variable_names}, "
        + f"input parameters {input_parameter_names}, output variable "
        + f"{output_variable_name}, and weight {weight}"
    )

    # Transform input and output variables for training dataset creation, leave
    # the original data frame unchanged
    df_trans = df.copy()
    df_trans = apply_transformations(df_trans, transformations)

    # Create torch tensors from transformed data frame
    x = torch.from_numpy(
        df_trans.loc[:, input_variable_names]
        .astype(np.float32)
        .to_numpy(copy=True)
    ).float()
    p = torch.from_numpy(
        df_trans.loc[:, input_parameter_names]
        .astype(np.float32)
        .to_numpy(copy=True)
    ).float()
    y = torch.from_numpy(
        df_trans.loc[:, [output_variable_name]]
        .astype(np.int64)
        .to_numpy(copy=True)
    ).long()
    w = torch.from_numpy(
        df_trans.loc[:, weight].astype(np.float32).to_numpy(copy=True)
    ).float()
    r = torch.from_numpy(
        ~df_trans.loc[:, "is_signal"].astype(np.bool_).to_numpy(copy=True)
    ).bool()

    # Create the tensor dataset from the torch tensors
    dataset = TensorDataset(x, p, y, w, r)
    logger.debug("Created dataset with torch tensor objects from data frame")

    return dataset
