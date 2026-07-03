import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from xyh_ml.config.data import DataConfig


class PNNDataset(Dataset):
    def __init__(
        self,
        inputs: Tensor,
        parameters: Tensor,
        targets: Tensor,
        weights: Tensor,
        resample_parameters: Tensor,
    ):
        self.inputs = inputs
        self.parameters = parameters
        self.targets = targets
        self.weights = weights
        self.resample_parameters = resample_parameters

        if not (len(self.inputs) == len(self.targets) == len(self.weights)):
            raise ValueError(
                "Inputs, targets, and weights must have the same length"
            )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            self.inputs[idx],
            self.parameters[idx],
            self.targets[idx],
            self.weights[idx],
            self.resample_parameters[idx],
        )


def create_pnn_data_loaders(
    data_frames: dict,
    data_config: DataConfig,
    batch_size: int,
    weight_name: str,
    signal_flag_name: str,
    num_workers: int | None = None,
):
    # Container for the output datasets and data loaders
    data_loaders = {}
    datasets = {}

    for key in data_frames:
        df = data_frames[key]

        # Create tensors for inputs, parameters, and targets
        input_names = [v.name for v in data_config.input_variables]
        parameter_names = [p.name for p in data_config.input_parameters]
        target_names = [f"class_{c.name}" for c in data_config.output_classes]
        inputs = torch.from_numpy(df[input_names].values).float()
        parameters = torch.from_numpy(df[parameter_names].values).float()
        targets = torch.from_numpy(df[target_names].values).int()

        # Get weights from the column with name weight_name
        weights = torch.from_numpy(df[weight_name].values).float()

        # Flag to indicate for which events the parameters should be resampled
        resample_parameters = torch.from_numpy(
            ~df[signal_flag_name].values
        ).bool()

        # Create the custom PNN dataset
        dataset = PNNDataset(
            inputs=inputs,
            parameters=parameters,
            targets=targets,
            weights=weights,
            resample_parameters=resample_parameters,
        )

        # Create the data loader for this dataset and add it to the container
        shuffle = key[1] == "train"
        datasets[key] = dataset
        data_loaders[key] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
        )

    return data_loaders, datasets
