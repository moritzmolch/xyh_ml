import logging
from time import time

import torch
from torch.nn import Module
from torch.nn.functional import softmax
from torch.utils.data import DataLoader, TensorDataset

# Set up logging
logger = logging.getLogger(__name__)


def predict(
    model: Module,
    dataset: TensorDataset,
    device: str,
    data_loader_num_workers: int = 1,
    data_loader_pin_memory: bool = False,
    batch_size: int = 65536,
):
    """
    Predict output of a trained model for a `TensorDataset`.

    The dataset is expected to contain the attributes `x` for the inputs
    variables and `p` for the input parameters. Look at the documentation of
    `xyh_ml.data.torch.create_torch_dataset_for_evaluation` for more
    information.

    The function returns a numpy array with shape `(n_events, n_classes)`,
    where columns contain the softmax-normalized output of the model for each
    class.

    Parameters
    ----------
    model : Module
        Trained model to use for predictions.

    dataset : TensorDataset
        Dataset to use for predictions. The dataset is expected to contain the
        the attributes `x` and `p`.

    device : str
        Device to use for predictions. Can be either `cpu` or `cuda`.

    data_loader_num_workers : int, optional
        Number of workers to use for the data loader. (default: `1`)

    data_loader_pin_memory : bool, optional
        Whether to pin memory for the data loader. (default: `False`)

    batch_size : int, optional
        Batch size to use for predictions. (default: `65536`)

    Returns
    -------
    np.ndarray
        Array with shape `(n_events, n_classes)` containing the predicted
        class probabilities for each event in the dataset.
    """

    # Create the data loader for predictions
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_loader_num_workers,
        pin_memory=data_loader_pin_memory,
    )

    # Move the model to the training device and turn on evaluation mode
    model = model.eval().to(device)

    # Start time tracking
    start = time()

    pred = []
    for x, p in data_loader:
        # Move tensors to training device
        x = x.to(device)
        p = p.to(device)

        # Get the network prediction and append it to the list of prediction
        # tensors. Directly calculate normalized softmax scores
        with torch.no_grad():
            pred.append(
                softmax(
                    model(torch.cat((x, p), dim=-1)),
                    dim=-1,
                )
            )

    # Concatenate predictions, targets, and weights and convert to numpy arrays
    pred = torch.cat(pred, dim=0).cpu().numpy()

    # Stop time tracking
    stop = time()

    # Log number of events and time spent for evaluation
    n_events = len(pred)
    delta = round(stop - start, 2)
    logger.debug(
        f"Evaluated model predictions for {n_events} events in {delta} s"
    )

    return pred
