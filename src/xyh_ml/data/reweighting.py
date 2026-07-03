import pandas as pd

# -----------------------------------------------------------------------------
# Dataset reweighting
# -----------------------------------------------------------------------------


def reweight_sample_with_negative_weights(
    df: pd.DataFrame,
    process: str,
    training_weight_name: str,
):
    """
    Reweight a sample with negative weights to positive weights while
    preserving the normalization of the sample.

    Parameters
    ----------

    df : pd.DataFrame
        The input data frame containing the events to be reweighted.

    process : str
        The name of the process class for which the reweighting should be
        applied.

    Returns
    -------
    pd.DataFrame
        The data frame with the reweighted `weight` column.
    """

    # Mask for process class that the reweighting should be applied to
    mask = df["class"] == process

    # Calculate the sum of weights before reweighting for each class
    sum_of_weights_before = df.loc[mask, training_weight_name].sum()

    # Calculate the sum of weights after reweighting for each class
    sum_of_weights_after = df.loc[mask, training_weight_name].abs().sum()

    # Rescale the weights so that sum of weights is preserved after reweighting
    scale_factor = (
        sum_of_weights_before / sum_of_weights_after
        if sum_of_weights_after > 0
        else 1.0
    )

    # Apply the weights to the events of the considered class
    df.loc[mask, training_weight_name] = (
        scale_factor * df.loc[mask, training_weight_name].abs()
    )

    return df


def reweight_signal_hypotheses(
    df: pd.DataFrame,
    process: str,
    training_weight_name: str,
):
    """
    Reweight mass hypotheses of a given signal process to have the same sum of
    weights.

    Parameters
    ----------
    df : pd.DataFrame
        The input data frame containing the events to be reweighted.

    process : str
        The name of the process class for which the reweighting should be
        applied.

    Returns
    -------
    pd.DataFrame
        The data frame with the reweighted `weight` column.
    """

    # Mask for process class that the reweighting should be applied to
    mask = df["class"] == process

    # Calculate sum of weights for each signal mass hypothesis
    masses = df.loc[mask, ["m_x", "m_y"]].apply(tuple, axis=1).unique()
    weights_before = {}
    for m_x, m_y in masses:
        weights_before[(m_x, m_y)] = df.loc[
            mask & (df["m_x"] == m_x) & (df["m_y"] == m_y),
            training_weight_name,
        ].sum()

    # Calculate average sum of weights across all mass hypotheses
    average_weight = sum(weights_before.values()) / len(weights_before)

    # Rescale the weights so that all mass hypotheses are balanced equally
    for m_x, m_y in masses:
        scale_factor = (
            average_weight / weights_before[(m_x, m_y)]
            if weights_before[(m_x, m_y)] > 0
            else 1.0
        )
        df.loc[
            mask & (df["m_x"] == m_x) & (df["m_y"] == m_y),
            training_weight_name,
        ] *= scale_factor

    return df


def reweight_processes(
    df: pd.DataFrame,
    training_weight_name: str,
):
    """
    Reweight the process classes to have the same sum of weights.

    Parameters
    ----------
    df : pd.DataFrame
        The input data frame containing the events to be reweighted.

    Returns
    -------
    pd.DataFrame
        The data frame with the reweighted `weight` column.
    """

    # Calculate sum of weights for each process class
    processes = df["class"].unique()
    weights_before = {}
    for process in processes:
        weights_before[process] = df.loc[
            df["class"] == process, training_weight_name
        ].sum()

    # Calculate average sum of weights across all process classes
    average_weight = sum(weights_before.values()) / len(weights_before)

    # Rescale the weights so that all process classes are balanced equally
    for process in processes:
        scale_factor = (
            average_weight / weights_before[process]
            if weights_before[process] > 0
            else 1.0
        )
        df.loc[df["class"] == process, training_weight_name] *= scale_factor

    return df


def reweight(
    dfs: dict[tuple[str, str], pd.DataFrame],
    weight_name: str,
    training_weight_name: str,
    signal_processes: list[str],
):
    """
    Calculate event weights for the training for different purposes.

    - The first step consists in reweighting single mass hypotheses of the
      signal processes to have the same sum of weights.
    - In the second step, the two inclusive signal classes, as well as events
      from the single background classes are reweighted to have the same sum of
      weights across all classes.

    Parameters
    ----------
    dfs : dict[tuple[str, str], pd.DataFrame]
        A dictionary mapping split keys to data frames.

    weight_name : str
        The name of the column containing the original event weights.

    training_weight_name : str
        The name of the new column with the training weight.

    signal_processes : list[str]
        The list of signal process classes for which the first step of the
        reweighting should be applied.

    Returns
    -------
    dict[tuple[str, str], pd.DataFrame]
        The dictionary with the reweighted data frames.
    """
    for key in dfs:
        # Get the data frame for the current split
        df = dfs[key]

        # Create the training weight, start with the "physics weight" of the
        # event
        df[training_weight_name] = df[weight_name]

        # Reweight all mass hypotheses within a signal process to have the same
        # sum of weights
        for process in signal_processes:
            df = reweight_signal_hypotheses(df, process, training_weight_name)

        # Reweight processes so that the sum of weights is the same for all
        # processes is equal
        df = reweight_processes(df, training_weight_name)

        # Put the reweighted data frame back into the dictionary
        dfs[key] = df

    return dfs
