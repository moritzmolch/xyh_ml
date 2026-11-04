import logging
from time import time

import numpy as np
import torch
from torch.nn import Module
from torch.nn.functional import softmax
from torch.utils.data import TensorDataset

# Set up logging
logger = logging.getLogger(__name__)


def predict(
    model: Module,
    dataset: TensorDataset,
    device: str,
    batch_size: int = 65536,
) -> np.ndarray:
    """
    Predict output of a trained model for a `TensorDataset`.

    The dataset is expected to contain the attributes `x` for the inputs
    variables and `p` for the input parameters. Look at the documentation of
    `xyh_ml.data.torch.create_torch_dataset_for_evaluation` for more
    information.

    The function returns a numpy array with shape `(n_events, n_classes)`,
    where columns contain the softmax-normalized output of the model for each
    class.

    The tensors of the dataset are evaluated in consecutive slices of
    `batch_size` events. A `DataLoader` is not used, as it fetches and
    collates every event individually, which took far longer than the model
    evaluation itself.

    Parameters
    ----------
    model : Module
        Trained model to use for predictions.

    dataset : TensorDataset
        Dataset to use for predictions. The dataset is expected to contain the
        the attributes `x` and `p`.

    device : str
        Device to use for predictions. Can be either `cpu` or `cuda`.

    batch_size : int, optional
        Batch size to use for predictions. (default: `65536`)

    Returns
    -------
    np.ndarray
        Array with shape `(n_events, n_classes)` containing the predicted
        class probabilities for each event in the dataset.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    # Move the model to the evaluation device and turn on evaluation mode
    model = model.eval().to(device)

    # Start time tracking
    start = time()

    # Evaluate the model in slices of the dataset tensors. At least one
    # (possibly empty) slice is evaluated, so that the output has the correct
    # number of columns also for empty datasets.
    x, p = dataset.tensors
    n_events = len(x)
    pred = []
    with torch.no_grad():
        for i in range(0, max(n_events, 1), batch_size):
            # Concatenate inputs and parameters and move them to the device
            xp = torch.cat(
                (x[i : i + batch_size], p[i : i + batch_size]), dim=-1
            ).to(device)

            # Directly calculate normalized softmax scores
            pred.append(softmax(model(xp), dim=-1))

    # Concatenate predictions and convert to a numpy array
    pred = torch.cat(pred, dim=0).cpu().numpy()

    # Stop time tracking
    stop = time()

    # Log number of events and time spent for evaluation
    delta = round(stop - start, 2)
    logger.debug(
        f"Evaluated model predictions for {n_events} events in {delta} s"
    )

    return pred
