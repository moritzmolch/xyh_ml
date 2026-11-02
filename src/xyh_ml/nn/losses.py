from functools import partial
from typing import Any

import torch
from torch import Tensor


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


def focal_loss(
    prediction: Tensor,
    target: Tensor,
    weight: Tensor,
    *,
    gamma: float,
) -> Tensor:
    """
    Custom focal loss function that calculates the weighted mean of
    focal losses for a batch of events.

    Parameters
    ----------
    prediction : Tensor
        Predicted class probabilities for each event in the batch. Shape is
        (batch_size, num_classes).

    target : Tensor
        True class labels for each event in the batch. Shape is (batch_size,).

    weight : Tensor
        Event weights for each event in the batch. Shape is (batch_size,).

    gamma : float
        Focusing parameter for the focal loss. Must be non-negative.

    Returns
    -------
    Tensor
        Weighted mean of focal losses for the batch.
    """

    # Get the single losses for each event
    ce_elements = torch.nn.functional.cross_entropy(
        prediction,
        target,
        reduction="none",
    )
    one_minus_p_t = -torch.expm1(-ce_elements)
    elements = torch.pow(one_minus_p_t, gamma) * ce_elements

    # Calculate weighted sum
    loss = torch.sum(weight * elements) / torch.sum(weight)

    return loss


def get_loss_fn(loss_fn: str, loss_kwargs: dict[str, Any]):
    """
    Get the loss function based on its name and keyword arguments.

    Parameters
    ----------
    loss_fn : str
        Name of the loss function.

    loss_kwargs : dict[str, Any]
        Keyword arguments for the loss function.

    Returns
    -------
    callable
        The loss function.
    """
    if loss_fn == "cross_entropy":
        return cross_entropy_loss
    elif loss_fn == "focal":
        required = {"gamma"}
        if set(loss_kwargs.keys()) != required:
            raise ValueError(
                f"Focal loss needs the following keyword arguments: {required}"
            )
        return partial(focal_loss, **loss_kwargs)
    else:
        raise ValueError(f"Unknown loss function: {loss_fn}")
