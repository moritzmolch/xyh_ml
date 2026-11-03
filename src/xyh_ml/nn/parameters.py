import torch
from torch import Generator, Tensor


def resample_input_parameters(
    parameters: Tensor,
    resample_mask: Tensor,
    unique_parameters: Tensor,
    rng: Generator | None = None,
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
