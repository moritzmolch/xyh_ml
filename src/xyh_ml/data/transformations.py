import copy
import logging
from typing import Any

import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from xyh_ml.config.dataclasses import Variable

# Set up logging
logger = logging.getLogger(__name__)


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


def apply_transformations_to_unique_input_parameters(
    unique_parameters: DataFrame,
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
