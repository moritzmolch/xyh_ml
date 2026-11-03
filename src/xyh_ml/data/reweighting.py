import logging

import numpy as np
from pandas import DataFrame

# Set up logging
logger = logging.getLogger(__name__)


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
