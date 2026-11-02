import logging
from typing import Any

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
