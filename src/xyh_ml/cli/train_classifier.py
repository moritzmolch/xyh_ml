import copy
import json
import logging
import pickle
from functools import reduce
from operator import or_
from pathlib import Path
from time import time
from typing import Any, Literal

import hydra
import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from torch import Tensor
from torch.nn import Module
from torch.utils.data import TensorDataset

from xyh_ml import PROJECT_DIR
from xyh_ml.cli.model import MultiLayerPerceptron
from xyh_ml.config import (
    Config,
    Network,
    OutputClass,
    Training,
    Variable,
)

# Set up logging
logger = logging.getLogger(__name__)

# Seeds
TRAIN_TEST_SPLIT_SEED = 4712
PARAMETER_SAMPLING_SEED = 583


def prepare_output_dir(
    output_dir: Path,
):
    # Create the output directory if it does not exist
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        logger.debug(f"Created output directory {output_dir}")


def load_data_frame(
    data_frame_dir: Path,
    decay_mode: str,
    output_classes: list[OutputClass],
    n_folds: int | None = None,
    fold: int | None = None,
    fold_selection: Literal["train", "eval"] | None = None,
    channel: str | None = None,
) -> DataFrame:
    """
    Load the data frame from the preprocessed HDF5 files.

    For signal events, the data frame is filtered for the given `decay_mode`.
    If `fold_selection` is set, the events are assigned to `n_folds` folds
    based on their event number (`event % n_folds`) and filtered for either
    the training folds or the held-out evaluation fold. If `channel` is set, the data frame is filtered for that
    analysis channel. The full data frame is loaded otherwise.

    The data frame is shuffled before being returned.

    Parameters
    ----------
    data_frame_dir : Path
        Path to the directory containing the preprocessed HDF5 files.

    decay_mode : str
        Decay mode to filter the signal events for. Must be either `y2b_h2tau`
        or `y2tau_h2b`.

    output_classes : list[OutputClass]
        List of output classes to load the data frames for.

    n_folds : int | None, optional
        Number of folds the events are split into. Required if
        `fold_selection` is set. Default is `None`.

    fold : int | None, optional
        Index of the held-out fold, between 0 and `n_folds - 1`. Required if
        `fold_selection` is set. Default is `None`.

    fold_selection : {"train", "eval"} | None, optional
        If `"train"`, keep the events of all folds except `fold`. If `"eval"`,
        keep only the events of the held-out fold `fold`. If `None`, no
        filtering is applied. Default is `None`.

    channel : str | None, optional
        Analysis channel (`et`, `mt` or `tt`) to filter the events for. If
        `None`, no filtering is applied. Default is `None`.

    Returns
    -------
    DataFrame
        Concatenated data frame containing the events for all output classes.

    Raises
    ------
    ValueError
        - If `decay_mode` is not `y2b_h2tau` or `y2tau_h2b`.
        - If `fold_selection` is not `train`, `eval` or `None`.
        - If `fold_selection` is set and `n_folds` or `fold` are invalid.
        - If `channel` is not `et`, `mt`, `tt` or `None`.
    """

    # Raise exception if decay mode is not `y2b_h2tau` or `y2tau_h2b`
    if decay_mode not in ["y2b_h2tau", "y2tau_h2b"]:
        msg = f"Decay mode must be either 'y2b_h2tau' or 'y2tau_h2b', got {decay_mode}"
        logger.critical(msg)
        raise ValueError(msg)

    # Raise exception if the fold selection is invalid
    if fold_selection not in ["train", "eval", None]:
        msg = f"Fold selection must be 'train', 'eval' or `None`, got {fold_selection}"
        logger.critical(msg)
        raise ValueError(msg)
    if fold_selection is not None and not (
        n_folds is not None
        and fold is not None
        and n_folds >= 2
        and 0 <= fold < n_folds
    ):
        msg = f"Need n_folds >= 2 and 0 <= fold < n_folds, got n_folds={n_folds}, fold={fold}"
        logger.critical(msg)
        raise ValueError(msg)

    # List of data frames which are going to be concatenated
    dfs: list[DataFrame] = []

    # Raise exception if channel is not `et`, `mt`, `tt` or `None`
    if channel not in ["et", "mt", "tt", None]:
        msg = (
            f"Channel must be either 'et', 'mt', 'tt' or `None`, got {channel}"
        )
        logger.critical(msg)
        raise ValueError(msg)

    # Track time for loading the data frames
    start = time()

    for output_class in output_classes:
        # Load the data frame from the preprocessed HDF5 file
        df = pd.read_hdf(
            data_frame_dir / f"{output_class.name}.h5",
            key="ntuple",
        )

        # If this is a signal output class, filter the data frame for the given
        # decay mode
        if output_class.is_signal:
            logger.debug(
                f"Select signal events for decay mode {decay_mode} in data "
                + f" frame for class {output_class.name}"
            )
            df = df.loc[df["decay_mode"] == decay_mode]

        # Filter for the training folds or the held-out fold if it has been set
        # in the function call
        if fold_selection is not None:
            logger.debug(
                f"Select {fold_selection} events for fold {fold} of {n_folds} "
                + f"in data frame for class {output_class.name}"
            )
            in_fold = df["event"] % n_folds == fold
            df = df.loc[in_fold if fold_selection == "eval" else ~in_fold]

        # Filter for analysis channel if it has been set in the function call
        if channel is not None:
            logger.debug(
                f"Filter events for channel {channel} in data frame "
                + f"for class {output_class.name}"
            )
            df = df.loc[df["channel"] == channel]

        dfs.append(df)

        # Log information about the loaded data frame
        logger.info(f"Loaded data frame for output class {output_class.name}")

    # Concatenate the data frames and shuffle the events
    df = pd.concat(dfs, axis=0).sample(frac=1.0).reset_index(drop=True)

    # Stop time tracking
    stop = time()

    # Log information about data frame loader
    n_events = len(df)
    mem = round(df.memory_usage().sum() / (1024**2), 2)
    delta = round(stop - start, 2)
    logger.debug(
        f"Loaded data frame with {n_events} events of size {mem} MB in "
        + f"{delta} s"
    )

    return df


def train_val_split(
    df: DataFrame,
    train_frac: float,
) -> dict[str, DataFrame]:
    """
    Split the full dataset into a training and validation set.

    Parameters
    ----------
    df : DataFrame
        Full dataset to split into training and validation sets.

    train_frac : float
        Fraction of the full dataset to use for training. Must be between 0 and
        1.

    Returns
    -------
    dict[str, DataFrame]
        Dictionary containing the training and validation data frames with
        keys "train" and "val", respectively.

    Raises
    ------
    ValueError
        If `train_frac` is not between 0 and 1.
    """

    # Raise an exception if training fraction is not between 0 and 1
    if train_frac < 0 or train_frac > 1:
        msg = f"Training fraction must be between 0 and 1, got {train_frac}"
        logger.critical(msg)
        raise ValueError(msg)

    # Use corresponding sklearn function to split the data frame into a
    # training and validation data frame
    df_train, df_val = train_test_split(
        df,
        train_size=train_frac,
        shuffle=True,
        random_state=TRAIN_TEST_SPLIT_SEED,
    )
    logger.debug(
        "Split into training and validation data frames with "
        + f"{len(df_train)} and {len(df_val)} events, respectively"
    )

    return {
        "train": df_train,
        "val": df_val,
    }


def reweight_by_group(
    df: DataFrame,
    group: str | list[str],
    weight_name: str,
    mask: np.ndarray | None = None,
) -> DataFrame:
    """
    Reweight events in the data frame such that the sum of weights for each
    group is equal to the mean sum of weights over all groups.

    The groups are defined by the `group` parameter, which is a column or a
    list of columns. The data frame is then split into groups based on the
    unique values of the `group` columns.

    The `weight_name` parameter specifies the weight column that is going to be
    altered. It must already be present in the data frame.

    The `mask` parameter can be used to select a subset of events to reweight.
    If `mask` is `None`, all events are used.

    Parameters
    ----------
    df : DataFrame
        Data frame containing the events to reweight.

    group : str | list[str]
        Column or list of columns to group the events by.

    weight_name : str
        Name of the weight column to reweight.

    mask : np.ndarray | None, optional
        Boolean mask to select a subset of events to reweight. If `None`, all
        events are used. Default is `None`.

    Returns
    -------
    DataFrame
        Data frame with the modified `weight_name` column.
    """

    # Create dummy mask that accepts all events if the mask has not been set
    # in the function call
    mask = mask if mask is not None else np.ones(df.shape[0], dtype=bool)

    # Create the grouper
    df_groups = df.loc[mask].groupby(by=group)

    # Calculate sum of weights for each group and get the average value over
    # all individual group values
    sum_of_weights = df_groups.agg({weight_name: "sum"})
    mean_sum_of_weights = sum_of_weights.mean().to_numpy()[0]
    logger.debug(
        f"Sum of weights per group before reweighting:\n{sum_of_weights.to_string()}"
    )
    logger.debug(f"Average sum of weights:\n{mean_sum_of_weights}")

    # Calculate the scale factor for each group, which scales the sum of weights
    # of all groups to the mean sum of weights
    scale_factor = mean_sum_of_weights / sum_of_weights
    logger.debug(f"Scale factors for reweighting:\n{scale_factor.to_string()}")

    # Apply the scale factor
    for group_values, df_group in df_groups:
        df.loc[df_group.index, weight_name] = (
            scale_factor.loc[group_values].iloc[0]
            * df.loc[df_group.index, weight_name]
        )

    # Calculate sum of weights after reweighting (for logging purposes)
    sum_of_weights = df.loc[mask].groupby(by=group).agg({weight_name: "sum"})
    logger.debug(
        f"Sum of weights per group after reweighting:\n{sum_of_weights.to_string()}"
    )

    return df


def derive_transformations(
    df: DataFrame,
    variables: list[Variable],
) -> dict[str, dict[str, Any]]:
    """
    Derive variable transformations on an input data frame and return a
    dictionary of the fitted transformation objects.

    Currently, `StandardScaler` and `OrdinalEncoder` transformations are
    supported. Transformations are declared in the `Variable` objects with the
    `transformation` attribute, which must be empty or set to either
    `standard_scaling` or `ordinal_encoding`.

    Parameters
    ----------
    df : DataFrame
        Data frame to fit the transformations on.

    variables : list[Variable]
        List of variables to fit the transformations for.

    Returns
    -------
    dict[str, Any]
        Dictionary containing the fitted transformation objects and the
        corresponding variable names. The keys are the transformation names and
        the values are dictionaries with the following keys:
        - `"transformation"`: The fitted transformation object. The value is
          either a `StandardScaler` or an `OrdinalEncoder` object.
        - `"variables"`: List of variable names that the transformation is
          applied to.
    """

    # Set up the transformations with sklearn preprocessing objects
    transformations: dict[str, dict[str, Any]] = {
        "standard_scaling": {
            "transformation": StandardScaler(),
        },
        "ordinal_encoding": {
            "transformation": OrdinalEncoder(),
        },
    }

    # Add information about transformed variables to the transformations
    # dictionary
    for transformation, transformation_dict in transformations.items():
        # Get the transformation object
        t = transformation_dict["transformation"]

        # Get the variable names that are going to be transformed with this
        # transformation
        v = [v.name for v in variables if v.transformation == transformation]
        transformation_dict.update({"variables": v})

        # Fit the transformation
        t.fit(df[v])
        logger.debug(
            f"Fitted transformation '{transformation}' for variables {v}"
        )

    return transformations


def apply_transformations(
    df: DataFrame,
    transformations: dict[str, dict[str, Any]],
) -> DataFrame:
    """
    Apply the fitted transformations to the input data frame and return the
    transformed data frame.

    Parameters
    ----------
    df : DataFrame
        Data frame to apply the transformations to.

    transformations : dict
        Dictionary containing the fitted transformation objects and the
        corresponding variable names. For more information, see the
        `derive_transformations` function.

    Returns
    -------
    DataFrame
        Transformed data frame with the same columns as the input data frame.
    """

    for transformation, transformation_dict in transformations.items():
        # Get the transformation object and the variables to transform
        t = transformation_dict["transformation"]
        v = transformation_dict["variables"]

        # Apply transform to the columns
        df[v] = t.transform(df[v])
        logger.debug(
            f"Applied transformation {transformation} for variables {v}"
        )

    return df


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
    logging.debug(
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


def create_multi_layer_perceptron_model(
    input_size: int,
    output_size: int,
    network: Network,
) -> MultiLayerPerceptron:
    """
    Create a simple multi-layer perceptron model.

    Parameters
    ----------
    input_size : int
        Number of input features.

    output_size : int
        Number of output classes.

    network : Network
        Network configuration containing hyperparameter settings of the network
        architecture.

    Returns
    -------
    MultiLayerPerceptron
        Multi-layer perceptron model with the specified architecture and
        hyperparameters.
    """

    # Create the model
    model = MultiLayerPerceptron(
        input_size,
        output_size,
        network.hidden_layers,
        network.activation,
        network.dropout_frac,
        activation_kwargs=network.activation_kwargs,
    )

    return model


def cross_entropy_loss(
    prediction: Tensor,
    target: Tensor,
    weight: Tensor,
) -> Tensor:
    """
    Custom cross-entropy loss function that calculates the weighted mean of
    cross-entropy losses for a batch of events.

    Parameters
    ----------
    prediction : Tensor
        Predicted class probabilities for each event in the batch. Shape is
        (batch_size, num_classes).

    target : Tensor
        True class labels for each event in the batch. Shape is (batch_size,).

    weight : Tensor
        Event weights for each event in the batch. Shape is (batch_size,).

    Returns
    -------
    Tensor
        Weighted mean of cross-entropy losses for the batch.
    """

    # Get the single losses for each event
    elements = torch.nn.functional.cross_entropy(
        prediction,
        target,
        reduction="none",
    )

    # Calculate weighted sum
    loss = torch.sum(weight * elements) / torch.sum(weight)

    return loss


def get_unique_input_parameter_combinations(
    df: DataFrame,
    parameter_variables: list[Variable],
) -> DataFrame:
    """
    Get unique combinations of input parameter values from the data frame.

    Parameters
    ----------

    df : DataFrame
        Data frame containing the input parameter values.

    parameter_variables : list[Variable]
        List of input parameter variables to get unique values for.

    Returns
    -------
    DataFrame
        Data frame containing unique combinations of input parameter values.
    """

    # Create list of parameter names
    parameter_names = [p.name for p in parameter_variables]

    logger.debug(
        "Getting unique combinations of input parameter values for parameters "
        + f"{parameter_names}"
    )

    # Get unique values of the parameters in the data frame
    unique_parameters = df[parameter_names].drop_duplicates()

    # Remove values uniquely composed of fallback values
    mask = reduce(
        or_,
        [
            unique_parameters.iloc[:, i] != p.fallback_value
            for i, p in enumerate(parameter_variables)
        ],
    )
    unique_parameters = unique_parameters.loc[mask].reset_index(drop=True)
    logger.debug(
        f"Found {len(unique_parameters)} unique combinations of input parameter values"
    )

    return unique_parameters


def apply_transformations_to_unique_input_parameters(
    unique_parameters: pd.DataFrame,
    transformations: dict[str, dict[str, Any]],
) -> DataFrame:
    """
    Apply transformations to the unique parameter combinations.

    Parameters
    ----------
    unique_parameters : DataFrame
        Data frame containing unique combinations of input parameter values.

    transformations : dict
        Dictionary containing the fitted transformation objects and the
        corresponding variable names. For more information, see the
        `derive_transformations` function.

    Returns
    -------
    DataFrame
        Transformed data frame with the same columns as the input data frame.
    """

    # Remember original sort order of parameters in data frame
    parameter_names = unique_parameters.columns.tolist()

    for transformation, transformation_dict in transformations.items():
        # Get the transformation object and the variables to transform
        t = transformation_dict["transformation"]
        v = transformation_dict["variables"]

        # Determine overlap between transformation's variable list and the
        # columns of the unique_parameters data frame
        v_overlap = list(set(v).intersection(set(unique_parameters.columns)))

        # Continue if there is no overlap
        if len(v_overlap) == 0:
            continue

        # TODO this is a hotfix, improve the implementation

        # Create temporary transformation that only contains the overlapping
        # variables
        t_tmp = None
        if transformation == "ordinal_encoding":
            feature_names_in = t.get_feature_names_out().tolist()
            categories = [
                t.categories_[feature_names_in.index(vo)] for vo in v_overlap
            ]
            t_tmp = copy.deepcopy(t)
            t_tmp.categories_ = categories
            t_tmp.feature_names_in_ = np.array(v_overlap)
            t_tmp.n_features_in_ = len(v_overlap)
        else:
            feature_names_out = t.get_feature_names_out().tolist()
            mean = np.array(
                [t.mean_[feature_names_out.index(vo)] for vo in v_overlap]
            )
            scale = np.array(
                [t.scale_[feature_names_out.index(vo)] for vo in v_overlap]
            )
            t_tmp = StandardScaler()
            t_tmp.mean_, t_tmp.scale_ = mean, scale

        # Apply transform to the columns
        unique_parameters[v_overlap] = t_tmp.transform(
            unique_parameters[v_overlap]
        )
        logger.debug(
            f"Applied transformation {transformation} for variables "
            + f"{v_overlap}"
        )

    # Reorder the columns of the unique_parameters data frame to match the
    # original order
    unique_parameters = unique_parameters.loc[:, parameter_names]

    return unique_parameters


def resample_input_parameters(
    parameters: Tensor,
    resample_mask: Tensor,
    unique_parameters: Tensor,
    rng: torch.Generator | None = None,
) -> Tensor:
    """
    Resample input parameter values for events that are marked for resampling
    in the `resample_mask`.

    Parameters
    ----------
    parameters : Tensor
        Tensor containing the input parameter values for each event. Shape is
        (num_events, num_parameters).

    resample_mask : Tensor
        Boolean tensor indicating which events should have their input parameter
        values resampled. Shape is (num_events,).

    unique_parameters : Tensor
        Tensor containing unique combinations of input parameter values. Shape
        is (num_unique_combinations, num_parameters).

    rng : torch.Generator | None, optional
        Random number generator to use for sampling. If `None`, the default
        generator is used. Default is `None`.

    Returns
    -------
    Tensor
        Tensor containing the input parameter values for each event after
        resampling. Shape is (num_events, num_parameters).
    """

    # Sample a random index in the unique_parameters list for each event that
    # is marked for resampling
    unique_parameters_index = torch.randint(
        0,
        unique_parameters.size(0),
        size=(torch.sum(resample_mask),),
        generator=rng,
        device=parameters.device,
    )

    # Change values of parameters that have been declared for resampling
    index = torch.where(resample_mask)[0]
    parameters[index] = unique_parameters[unique_parameters_index]

    return parameters


def train_and_validate(
    model: Module,
    dataset_train: TensorDataset,
    dataset_val: TensorDataset,
    unique_parameters: Tensor,
    training: Training,
    device: str,
    n_steps_logging: int = 25,
) -> dict[str, Any]:
    """
    Training and validation loop for the model.

    The function returns a dictionary with the training results. The dictionary
    contains the following keys:

    - `"checkpoints"`: List of model checkpoints after each epoch. Each
        checkpoint is a dictionary containing the following keys:
        - `"epoch"`: Epoch number.
        - `"step"`: Global step number.
        - `"model_state_dict"`: State dictionary of the model.
        - `"optimizer_state_dict"`: State dictionary of the optimizer.

    - `"metrics"`: Dictionary containing the tracked metrics during training
      and validation. The keys are the metric names and the values are lists of
      metric values for each step.

    Parameters
    ----------

    model : Module
        PyTorch model to train and validate.

    dataset_train : TensorDataset
        Training dataset. For more information, see the `create_torch_dataset`
        function.

    dataset_val : TensorDataset
        Validation dataset. For more information, see the
        `create_torch_dataset` function.

    unique_parameters : Tensor
        Tensor containing unique combinations of input parameter values. Shape
        is (num_unique_combinations, num_parameters).

    training : Training
        Training configuration containing hyperparameter settings for the
        training loop.

    device : str
        Device to use for training and validation. Must be either `"cpu"` or
        `"cuda"`.

    n_steps_logging : int, optional
        Number of training steps between logging of training metrics. Default
        is 25.

    Returns
    -------

    dict[str, Any]
        Dictionary containing the training results. For more information, see
        the function description.
    """

    # Get training hyperparameters
    max_epochs = training.max_epochs
    batch_size = training.batch_size
    optimizer_learning_rate = training.optimizer_learning_rate
    optimizer_weight_decay = training.optimizer_weight_decay
    data_loader_num_workers = training.data_loader_num_workers
    data_loader_pin_memory = training.data_loader_pin_memory

    # Create the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=optimizer_learning_rate,
        weight_decay=optimizer_weight_decay,
    )

    # Create the data loaders for training and validation
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_loader_num_workers,
        pin_memory=data_loader_pin_memory,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=len(dataset_val),
        shuffle=False,
        num_workers=data_loader_num_workers,
        pin_memory=data_loader_pin_memory,
    )

    # Move the model to the training device
    model = model.to(device)

    # Create the generator for parameter sampling
    sampling_rng = torch.Generator(device=device)
    sampling_rng.manual_seed(PARAMETER_SAMPLING_SEED)

    # Move parameters to the training device
    unique_parameters = unique_parameters.to(device)

    # Metrics container
    metrics = {}

    # Checkpoints container
    checkpoints = []

    # Global step counter
    step = 0

    for epoch in range(max_epochs):
        # Log training epoch
        logger.debug(f"Start training epoch {epoch}")

        # Use model in training mode
        model = model.train()

        for batch, (x, p, y, w, r) in enumerate(data_loader_train):
            # Clear gradients from previous iteration
            optimizer.zero_grad()

            # Move tensors to training device
            x = x.to(device)
            p = p.to(device)
            y = y.to(device)
            w = w.to(device)
            r = r.to(device)

            # Resample parameters for events that are marked for resampling
            p = resample_input_parameters(
                p, r, unique_parameters, rng=sampling_rng
            )

            # Get the network prediction and calculate the loss
            pred = model(torch.cat((x, p), dim=-1))
            loss = cross_entropy_loss(pred, y.view(-1), w)

            # Perform backpropagation
            loss.backward()
            optimizer.step()

            # Track training metrics
            metrics.setdefault("train_step", []).append(step)
            metrics.setdefault("train_epoch", []).append(epoch)
            metrics.setdefault("train_batch", []).append(batch)
            metrics.setdefault("train_count", []).append(w.size(0))
            metrics.setdefault("train_weight_sum", []).append(w.sum().item())
            metrics.setdefault("train_loss", []).append(loss.item())

            # Log results
            if batch % n_steps_logging == 0:
                logger.debug(
                    "Training metrics "
                    + "  /  ".join(
                        [
                            f"{name}: {metric_list[-1]}"
                            for name, metric_list in metrics.items()
                            if name.startswith("train")
                        ]
                    )
                )

            # Increment step counter
            step += 1

        # Use model in evaluation mode
        model = model.eval()

        for batch, (x, p, y, w, r) in enumerate(data_loader_val):
            # Move tensors to training device
            x = x.to(device)
            p = p.to(device)
            y = y.to(device)
            w = w.to(device)
            r = r.to(device)

            # Resample parameters for events that are marked for resampling
            p = resample_input_parameters(
                p, r, unique_parameters, rng=sampling_rng
            )

            # Get the network prediction and calculate the loss
            with torch.no_grad():
                pred = model(torch.cat((x, p), dim=-1))
                loss = cross_entropy_loss(pred, y.view(-1), w)

            # Track validation metrics
            metrics.setdefault("val_step", []).append(step)
            metrics.setdefault("val_epoch", []).append(epoch)
            metrics.setdefault("val_count", []).append(w.size(0))
            metrics.setdefault("val_weight_sum", []).append(w.sum().item())
            metrics.setdefault("val_loss", []).append(loss.item())

            # Log results
            logger.debug(
                "Validation metrics "
                + "  /  ".join(
                    [
                        f"{name}: {metric_list[-1]}"
                        for name, metric_list in metrics.items()
                        if name.startswith("val")
                    ]
                )
            )

        # Add model checkpoint after epoch
        checkpoints.append(
            {
                "epoch": epoch,
                "step": step,
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
            }
        )

    return {
        "checkpoints": checkpoints,
        "metrics": metrics,
    }


def dump_config(
    output_file: Path,
    config: Config,
):
    """
    Serialize training configuration to a JSON file.

    Parameters
    ----------
    output_file : Path
        Path to the output JSON file.

    config : Config
        Training configuration to serialize. Must be an instance of the
        `Config` dataclass.
    """

    # Serialize training configuration in a JSON file
    with output_file.open(mode="w") as f:
        json.dump(config, f, default=str, indent=4)
    logger.debug(f"Serialized configuration to {output_file} in JSON format")


def dump_transformations(
    output_file: Path,
    transformations: dict[str, dict[str, Any]],
):
    """
    Serialize fitted transformations to a pickle file.

    Parameters
    ----------
    output_file : Path
        Path to the output pickle file.

    transformations : dict
        Dictionary containing the fitted transformation objects and the
        corresponding variable names. For more information, see the
        `derive_transformations` function.
    """

    # Serialize transformation objects in a pickle file
    with output_file.open(mode="wb") as f:
        pickle.dump(transformations, f)
    logger.debug(
        f"Serialized transformations to {output_file} in pickle format"
    )


def dump_metrics(
    output_file: Path,
    metrics: list[dict[str, Any]],
):
    """
    Serialize tracked metrics to a JSON file.

    Parameters
    ----------
    output_file : Path
        Path to the output JSON file.

    metrics : dict
        Dictionary containing the tracked metrics.
    """

    # Serialize tracked metrics in a JSON file
    with output_file.open(mode="w") as f:
        json.dump(metrics, f)
    logger.debug(f"Serialized metrics to {output_file} in JSON format")


def dump_checkpoints(
    output_file: Path,
    checkpoints: list[dict[str, Any]],
):
    """
    Serialize tracked model checkpoints to a torch file.

    Parameters
    ----------
    output_file : Path
        Path to the output torch file.

    checkpoints : list
        List of model checkpoints after each epoch.
    """

    # Serialize tracked checkpoints with the torch.save function
    torch.save(checkpoints, output_file)
    logger.debug(
        f"Serialized model checkpoints to {output_file} in torch format"
    )


@hydra.main(
    config_path=str(PROJECT_DIR / "config"),
    config_name="train_classifier",
    version_base="1.3",
)
def main(cfg: Config):
    # Create output directory if it does not exist
    training_dir = Path(cfg.artifacts.training_dir)
    if not training_dir.exists():
        training_dir.mkdir(parents=True)
        logger.debug(f"Created output directory {training_dir}")

    # Load the input files into a data frame
    data_frame = load_data_frame(
        Path(cfg.artifacts.data_frame_dir),
        cfg.mode.decay_mode,
        cfg.outputs.classes,
        cfg.mode.n_folds,
        cfg.mode.fold,
        "train",
        cfg.mode.channel,
    )

    # Get unique parameter combinations
    unique_parameters = get_unique_input_parameter_combinations(
        data_frame,
        cfg.inputs.parameters,
    )

    # Define training weight column and turn negative weights into positive
    # weights
    train_weight = "train_weight"
    data_frame.loc[:, train_weight] = np.abs(
        data_frame.loc[:, "__total_weight"]
    )

    # Split into a training and validation data frame
    data_frames = train_val_split(
        data_frame,
        train_frac=cfg.dataset.train_frac,
    )

    # Reweight events so that all campaigns have the same sum of weights
    for split in data_frames:
        data_frames[split] = reweight_by_group(
            data_frames[split],
            "campaign",
            train_weight,
        )

    # Reweight events so that all signal hypotheses have the same sum of
    # weights
    for split in data_frames:
        data_frames[split] = reweight_by_group(
            data_frames[split],
            ["m_x", "m_y"],
            train_weight,
            mask=data_frames[split]["is_signal"],
        )

    # Reweight events so that all categories have the same sum of weights
    for split in data_frames:
        data_frames[split] = reweight_by_group(
            data_frames[split],
            "class",
            train_weight,
        )

    # Fit transformations to preprocess input and output variables on the
    # training dataset
    transformations = derive_transformations(
        data_frames["train"],
        cfg.inputs.variables + cfg.inputs.parameters + cfg.outputs.variables,
    )

    # Prepare training dataset by applying transformations and creating torch
    # tensors from the data frame. Create separate datasets for training and
    # validation.
    datasets = {}
    for split, data_frame in data_frames.items():
        datasets[split] = create_torch_dataset(
            data_frame,
            cfg.inputs.variables,
            cfg.inputs.parameters,
            cfg.outputs.variables[0],
            train_weight,
            transformations,
        )

    # Apply transformations to the unique parameter combinations and create a
    # torch tensor from the data frame
    unique_parameters_transformed = torch.from_numpy(
        (
            apply_transformations_to_unique_input_parameters(
                unique_parameters, transformations
            )
            .astype(np.float32)
            .to_numpy()
        )
    )

    # Create the neural network model
    model = create_multi_layer_perceptron_model(
        len(cfg.inputs.variables) + len(cfg.inputs.parameters),
        len(cfg.outputs.classes),
        cfg.network,
    )

    # Run the training
    training_results = train_and_validate(
        model,
        datasets["train"],
        datasets["val"],
        unique_parameters_transformed,
        cfg.training,
        "cuda" if torch.cuda.is_available() else "cpu",
    )

    # Serialize transformations, checkpoints, and metrics
    output_dir = Path(cfg.artifacts.training_dir)
    dump_config(output_dir / "config.json", cfg)
    dump_transformations(output_dir / "transformations.pkl", transformations)
    dump_metrics(output_dir / "metrics.json", training_results["metrics"])
    dump_checkpoints(
        output_dir / "checkpoints.pt", training_results["checkpoints"]
    )


if __name__ == "__main__":
    main()
