import torch


def weighted_cross_entropy_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    # Compute the weighted cross-entropy loss using log_softmax for numerical
    # stability. This prevents NaN values from log(0) or underflow in softmax.
    log_probs = torch.log_softmax(output, dim=-1)
    elements = -torch.sum(target * log_probs, dim=-1)
    return torch.sum(weight * elements) / torch.sum(weight)
