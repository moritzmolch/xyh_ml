from pathlib import Path
import pandas as pd
import logging
from time import time
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import pickle
import json
import copy

from xyh_ml.scripts.model import MultiLayerPerceptron


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Seeds
TRAIN_TEST_SPLIT_SEED = 4712


THIS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = THIS_DIR.parent.parent.parent / "test_output"

EVENT_PARITY = 0

INPUT_DIR = Path("/ceph/mmolch/xyh-bbtautau-crown/bbtautau/data/output/TrainingDataFrames/shapes-2026-06-10/2024/mt_base_sr")

CATEGORIES = [
    {
        "name": "xyh",
        "label": "$\\text{X} \\to \\text{H}\\text{Y}$",
        "file_basenames": [
            # "2024__mt__mt_base_sr__xyh_y2b_h2tau__Nominal.h5",
            "2024__mt__mt_base_sr__xyh_y2tau_h2b__Nominal.h5",
        ],
        "is_signal": True,
    },
    {
        "name": "t",
        "label": "Single t",
        "file_basenames": [
            "2024__mt__mt_base_sr__single_t__Nominal.h5",
        ],
        "is_signal": False,
    },
    {
        "name": "tt",
        "label": "$\\text{t}\\bar{\\text{t}}$",
        "file_basenames": [
            "2024__mt__mt_base_sr__tt_rem__Nominal.h5",
            "2024__mt__mt_base_sr__tt_tautau__Nominal.h5",
        ],
        "is_signal": False,
    },
    {
        "name": "z",
        "label": "Z($\\tau\\tau$)",
        "file_basenames": [
            "2024__mt__mt_base_sr__dy_tautau__Nominal.h5",
        ],
        "is_signal": False,
    },
    {
        "name": "jetfakes",
        "label": "$\\text{j} \\to \\tau_{\\text{h}}$",
        "file_basenames":
        [
            "2024__mt__mt_base_sr__jetfakes__tau_antiid_vs_jet.h5",
        ],
        "is_signal": False,
    },
    # {
    #     "name": "single_h",
    #     "label": "Single H",
    #     "file_basenames": [
    #         "2024__mt__mt_base_sr__single_h__Nominal.h5",
    #     ],
    #     "is_signal": False,
    # },
    # {
    #     "name": "diboson",
    #     "label": "Diboson",
    #     "file_basenames": [
    #         "2024__mt__mt_base_sr__vv__Nominal.h5",
    #     ],
    #     "is_signal": False,
    # },
]

INPUT_VARIABLES = [
    {
        "name": "n_jets",
        "transformation": "none",
    },
    {
        "name": "n_bjets",
        "transformation": "none",
    },
    {
        "name": "pt_1",
        "transformation": "standard_scaling",
    },
    {
        "name": "pt_2",
        "transformation": "standard_scaling",
    },
    {
        "name": "eta_1",
        "transformation": "standard_scaling",
    },
    {
        "name": "eta_2",
        "transformation": "standard_scaling",
    },
    {
        "name": "phi_1",
        "transformation": "standard_scaling",
    },
    {
        "name": "phi_2",
        "transformation": "standard_scaling",
    },
    {
        "name": "m_vis",
        "transformation": "standard_scaling",
    },
    {
        "name": "pt_vis",
        "transformation": "standard_scaling",
    },
    {
        "name": "deltaR_ditaupair",
        "transformation": "standard_scaling",
    },
    {
        "name": "bpair_pt_1",
        "transformation": "standard_scaling",
    },
    {
        "name": "bpair_pt_2",
        "transformation": "standard_scaling",
    },
    {
        "name": "bpair_eta_1",
        "transformation": "standard_scaling",
    },
    {
        "name": "bpair_eta_2",
        "transformation": "standard_scaling",
    },
    {
        "name": "bpair_phi_1",
        "transformation": "standard_scaling",
    },
    {
        "name": "bpair_phi_2",
        "transformation": "standard_scaling",
    },
    {
        "name": "bpair_btag_value_1",
        "transformation": "standard_scaling",
    },
    {
        "name": "bpair_btag_value_2",
        "transformation": "standard_scaling",
    },
    {
        "name": "bpair_m_inv",
        "transformation": "standard_scaling",
    },
    {
        "name": "bpair_pt_dijet",
        "transformation": "standard_scaling",
    },
    {
        "name": "bpair_deltaR",
        "transformation": "standard_scaling",
    },
    # {
    #     "name": "jpt_1",
    #     "transformation": "standard_scaling",
    # },
    # {
    #     "name": "jpt_2",
    #     "transformation": "standard_scaling",
    # },
    # {
    #     "name": "jeta_1",
    #     "transformation": "standard_scaling",
    # },
    # {
    #     "name": "jeta_2",
    #     "transformation": "standard_scaling",
    # },
    # {
    #     "name": "jphi_1",
    #     "transformation": "standard_scaling",
    # },
    # {
    #     "name": "jphi_2",
    #     "transformation": "standard_scaling",
    # },
    # {
    #     "name": "jtag_value_1",
    #     "transformation": "standard_scaling",
    # },
    # {
    #     "name": "jtag_value_2",
    #     "transformation": "standard_scaling",
    # },
    # {
    #     "name": "mjj",
    #     "transformation": "standard_scaling",
    # },
    # {
    #     "name": "pt_dijet",
    #     "transformation": "standard_scaling",
    # },
    {
        "name": "met",
        "transformation": "standard_scaling",
    },
    {
        "name": "metphi",
        "transformation": "standard_scaling",
    },
    {
        "name": "mt_1",
        "transformation": "standard_scaling",
    },
    {
        "name": "mt_2",
        "transformation": "standard_scaling",
    },
    {
        "name": "mt_tot",
        "transformation": "standard_scaling",
    },
    # {
    #     "name": "mass_tautaubb",
    #     "transformation": "standard_scaling",
    # },
    # {
    #     "name": "pt_tautaubb",
    #     "transformation": "standard_scaling",
    # },
    # {
    #     "name": "pt_tautau",
    #     "transformation": "standard_scaling",
    # },
    # {
    #     "name": "m_fastmtt",
    #     "transformation": "standard_scaling",
    # },
    # {
    #     "name": "pt_fastmtt",
    #     "transformation": "standard_scaling",
    # },
    # {
    #     "name": "eta_fastmtt",
    #     "transformation": "standard_scaling",
    # },
    # {
    #     "name": "phi_fastmtt",
    #     "transformation": "standard_scaling",
    # },
]

OUTPUT_VARIABLES = [
    {
        "name": "category",
        "transformation": "ordinal_encoding",
    },
]

DATASET_HYPERPARAMETERS = {
    "train_frac": 0.8,
}

NETWORK_HYPERPARAMETERS = {
    "hidden_layer_sizes": [512, 512, 512],
    "activation": "relu",
    "dropout_frac": 0.1,
}

TRAINING_HYPERPARAMETERS = {
    "max_epochs": 50,
    "batch_size": 2**15,
    "optimizer_learning_rate": 5.0e-3,
    "data_loader_num_workers": 4,
    "data_loader_pin_memory": True,
}


def _prepare_output_dir(output_dir: Path):
    # Create the output directory if it does not exist
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        logger.debug(f"Created output directory {output_dir}")


def _prepare_input_files(input_dir: Path, categories: list):
    # Create dictionary of input file lists, where keys correspond to the
    # output categories of the network
    input_files = {}

    for category in categories:
        for basename in category["file_basenames"]:
            # Get the category name
            category_name = category["name"]

            # Construct the absolute input file path
            input_file = input_dir / basename

            # Check if the input file exists
            if not input_file.exists():
                msg = f"Could not find input file {input_file}"
                logger.critical(msg)
                raise FileNotFoundError(msg)

            # Add input file to dictionary
            input_files.setdefault(category_name, []).append(input_file)
            logger.debug(
                f"Added {input_file} to output category {category_name}",
            )

    return input_files


def _load_data_frame(input_files: dict, categories: list):

    # List of all data frames that are loaded. The data frames are concatenated
    # to a single data frame later.
    dfs = []

    # Start time tracking
    start = time()

    for category_name, input_file_list in input_files.items():

        # Load data frame from each file separately and add it to the list of
        # all data frames. The category name is added as column.
        for input_file in input_file_list:
            df = pd.read_hdf(input_file)
            df["category"] = category_name
            dfs.append(df)
            logger.debug(f"Loaded data frame from {input_file}")

    # Concatenate all data frames. Sample with 100% fraction to shuffle the
    # dataset
    df = pd.concat(dfs, axis=0).sample(frac=1.0).reset_index(drop=True)

    # Stop time tracking
    stop = time()

    # Log information about data frame loader
    n_events = len(df)
    mem = round(df.memory_usage().sum() / (1024**2), 2)
    delta = round(stop - start, 2)
    logger.debug(
        f"Loaded data frame with {n_events} events of size {mem} MB in "
        + f"{delta} s"
    )

    return df


def _filter_event_parity(df: pd.DataFrame, event_parity: int):
    # Raise exception if parity is not 0 or 1
    if event_parity not in [0, 1]:
        msg = f"Event parity must be either 0 or 1, got {event_parity}"
        logger.critical(msg)
        raise ValueError(msg)

    # Filter events for given event parity
    n_events_before = len(df)
    df = df.loc[df["event"] % 2 == event_parity]

    # Log reduction of the data frame
    n_events_after = len(df)
    frac = round(n_events_after / n_events_before * 100, 2)
    logger.debug(
        f"Filtered data frame for events with parity {event_parity}, resulting "
        + f"in {n_events_after} selected events (efficiency: {frac} %)"
    )

    return df


def _train_val_split(
    df: pd.DataFrame,
    train_frac: float,
):
    # Raise an exception if training fraction is not between 0 and 1
    if train_frac < 0 or train_frac > 1:
        msg = f"Training fraction must be between 0 and 1, got {train_frac}"
        logger.critical(msg)
        raise ValueError(msg)

    df_train, df_val = train_test_split(
        df,
        train_size=train_frac,
        shuffle=True,
        random_state=TRAIN_TEST_SPLIT_SEED,
    )

    return {
        "train": df_train,
        "val": df_val,
    }


def _reweight_by_group(
    df: pd.DataFrame,
    group: str | list[str],
    input_weight_name: str,
    output_weight_name: str,
    mask: np.ndarray | None = None,
):
    # Log the results
    logger.debug(
        f"Reweight column groups {group} from input weight "
        + f"{input_weight_name} to output weight {output_weight_name}"
    )

    # Create dummy mask that accepts all events if the mask has not been set
    # in the function call
    mask = mask if mask is not None else np.ones(df.shape[0], dtype=bool)

    # Create the grouper
    df_groups = df.loc[mask].groupby(by=group)

    # Calculate sum of weights for each group and get the average value over
    # all individual group values
    sum_of_weights = df_groups.agg({input_weight_name: "sum"})
    mean_sum_of_weights = sum_of_weights.mean()[0]
    logger.debug(
        f"Sum of weights per group before reweighting:\n{sum_of_weights.to_string()}"
    )
    logger.debug(
        f"Average sum of weights: \n{mean_sum_of_weights}"
    )

    # Calculate the scale factor for each group, which scales the sum of weights
    # of all groups to the mean sum of weights
    scale_factor = mean_sum_of_weights / sum_of_weights
    logger.debug(
        f"Scale factors for reweighting:\n{scale_factor.to_string()}"
    )

    # Apply the scale factor
    for group_values, df_group in df_groups:
        df.loc[df_group.index, output_weight_name] = (
            scale_factor.loc[group_values][0]
            * df.loc[df_group.index, input_weight_name]
        )

    # Calculate sum of weights after reweighting (for logging purposes)
    sum_of_weights = df.loc[mask].groupby(by=group).agg({output_weight_name: "sum"})
    logger.debug(
        f"Sum of weights per group after reweighting:\n{sum_of_weights.to_string()}"
    )

    return df


def _setup_transformations(df: pd.DataFrame, variables: list):

    # Set up the transformations with sklearn preprocessing objects
    transformations = {
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
        transformation_dict.update({
            "variables": [
                v["name"]
                for v in variables
                if v["transformation"] == transformation
            ]
        })

    # Fit the transformations
    for transformation_dict in transformations.values():
        # Get the transformation object and the variables to transform
        t = transformation_dict["transformation"]
        v = transformation_dict["variables"]

        # Fit the transformation
        t.fit(df[v])
        logger.debug(
            f"Fitted transformation '{transformation}' for variables {v}"
        )

    return transformations


def _apply_transformations(df: pd.DataFrame, transformations: dict):
    for transformation, transformation_dict in transformations.items():
        # Get the transformation object and the variables to transform
        t = transformation_dict["transformation"]
        v = transformation_dict["variables"]

        # Apply transform to the columns
        df.loc[:, v] = t.transform(df[v])
        logger.debug(
            f"Applied transformation {transformation} for variables {v}"
        )

    return df


def _prepare_dataset(
    df: pd.DataFrame,
    input_variables: list,
    output_variables: list,
    weight_variable: list,
    transformations: dict,
):
    # Get column names
    input_variable_names = [v["name"] for v in input_variables]
    output_variable_names = [v["name"] for v in output_variables]
    weight_variable_name = weight_variable["name"]

    # Transform input and output variables for training dataset creation
    df_trans = _apply_transformations(df, transformations)

    # Create torch tensors from transformed data frame
    x = torch.from_numpy(
        df_trans.loc[:, input_variable_names].astype(np.float32).to_numpy()
    ).float()
    y = torch.from_numpy(
        df_trans.loc[:, output_variable_names].astype(np.int64).to_numpy()
    ).long()
    w = torch.from_numpy(
        df_trans.loc[:, weight_variable_name].astype(np.float32).to_numpy()
    ).float()

    dataset = torch.utils.data.TensorDataset(x, y, w)
    logger.debug("Created dataset with torch tensor objects from data frame")

    return dataset


def _create_multi_layer_perceptron_model(
    input_size: int,
    output_size: int,
    network_hyperparameters: dict
):
    # Create the model
    model = MultiLayerPerceptron(
        input_size,
        output_size,
        network_hyperparameters["hidden_layer_sizes"],
        network_hyperparameters["activation"],
        network_hyperparameters["dropout_frac"],
        activation_kwargs=network_hyperparameters.get("activation_kwargs", {}),
    )

    return model


def cross_entropy_loss(prediction, target, weight):
    # Get the single losses for each event
    elements = torch.nn.functional.cross_entropy(
        prediction,
        target,
        reduction="none",
    )

    # Calculate weighted sum
    loss = torch.sum(weight * elements) / torch.sum(weight)

    return loss


def _train_and_validate(
    model: torch.nn.Module,
    dataset_train: torch.utils.data.TensorDataset,
    dataset_val: torch.utils.data.TensorDataset,
    training_hyperparameters: dict,
    device: str,
    n_steps_logging: int = 25,
):
    # Get training hyperparameters
    max_epochs = training_hyperparameters["max_epochs"]
    batch_size = training_hyperparameters["batch_size"]
    optimizer_learning_rate = training_hyperparameters["optimizer_learning_rate"]
    data_loader_num_workers = training_hyperparameters["data_loader_num_workers"]
    data_loader_pin_memory = training_hyperparameters["data_loader_pin_memory"]

    # Create the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=optimizer_learning_rate,
    )

    # Create the data loaders for training and validation
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=data_loader_num_workers,
        pin_memory=data_loader_pin_memory,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=len(dataset_val),
        shuffle=False,
        num_workers=data_loader_num_workers,
        pin_memory=data_loader_pin_memory,
    )

    # Move the model to the training device
    model = model.to(device)

    # Metrics container
    metrics = {}

    # Checkpoints container
    checkpoints = []

    # Global step counter
    step = 0

    for epoch in range(max_epochs):
        # Log training epoch
        logger.debug(f"Start training epoch {epoch}")

        # Use model in training mode
        model = model.train()

        for batch, (x, y, w) in enumerate(data_loader_train):
            # Clear gradients from previous iteration
            optimizer.zero_grad()

            # Move tensors to training device
            x, y, w = x.to(device), y.to(device), w.to(device)

            # Get the network prediction and calculate the loss
            pred = model(x)
            loss = cross_entropy_loss(pred, y.view(-1), w)

            # Perform backpropagation
            loss.backward()
            optimizer.step()

            # Track training metrics
            metrics.setdefault("train_step", []).append(step)
            metrics.setdefault("train_epoch", []).append(epoch)
            metrics.setdefault("train_batch", []).append(batch)
            metrics.setdefault("train_count", []).append(w.size(0))
            metrics.setdefault("train_weight_sum", []).append(w.sum().item())
            metrics.setdefault("train_loss", []).append(loss.item())

            # Log results
            if batch % n_steps_logging == 0:
                logger.debug(
                    "Training metrics "
                    + "  /  ".join([
                        f"{name}: {metric_list[-1]}"
                        for name, metric_list in metrics.items()
                        if name.startswith("train")
                    ])
                )

            # Increment step counter
            step += 1

        # Use model in evaluation mode
        model = model.eval()

        for batch, (x, y, w) in enumerate(data_loader_val):
            # Move tensors to training device
            x, y, w = x.to(device), y.to(device), w.to(device)

            # Get the network prediction and calculate the loss
            with torch.no_grad():
                pred = model(x)
                loss = cross_entropy_loss(pred, y.view(-1), w)

            # Track training metrics
            metrics.setdefault("val_step", []).append(step)
            metrics.setdefault("val_epoch", []).append(epoch)
            metrics.setdefault("val_count", []).append(w.size(0))
            metrics.setdefault("val_weight_sum", []).append(w.sum().item())
            metrics.setdefault("val_loss", []).append(loss.item())

            # Log results
            logger.debug(
                "Validation metrics "
                + "  /  ".join([
                    f"{name}: {metric_list[-1]}"
                    for name, metric_list in metrics.items()
                    if name.startswith("val")
                ])
            )

        # Add model checkpoint after epoch
        checkpoints.append({
            "epoch": epoch,
            "step": step,
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        })

    return {
        "checkpoints": checkpoints,
        "metrics": metrics,
    }


def _dump_transformations(output_file: Path, transformations: dict):
    # Serialize transformation objects in a pickle file
    with output_file.open(mode="wb") as f:
        pickle.dump(transformations, f)
    logger.debug(
        f"Serialized transformations to {output_file} in pickle format"
    )


def _dump_metrics(output_file: Path, metrics: dict):
    # Serialize tracked metrics in a JSON file
    with output_file.open(mode="w") as f:
        json.dump(metrics, f)
    logger.debug(
        f"Serialized metrics to {output_file} in JSON format"
    )


def _dump_checkpoints(output_file: Path, checkpoints: list):
    # Serialize tracked checkpoints with the torch.save function
    torch.save(checkpoints, output_file)
    logger.debug(
        f"Serialized model checkpoints to {output_file} in torch format"
    )


def main():
    # Prepare the output directory
    _prepare_output_dir(OUTPUT_DIR)

    # Load the input files
    input_files = _prepare_input_files(INPUT_DIR, CATEGORIES)

    # Load the input files into a data frame
    data_frame = _load_data_frame(input_files, CATEGORIES)

    # Filter events according to the event parity
    data_frame = _filter_event_parity(data_frame, EVENT_PARITY)


    # Split into a training and validation data frame
    data_frames = _train_val_split(data_frame, train_frac=DATASET_HYPERPARAMETERS["train_frac"])

    # Reweight events so that all categories have the same sum of weights
    for split in data_frames:
        data_frames[split] = _reweight_by_group(
            data_frames[split],
            "category",
            "weight",
            "train_weight",
        )

    # Fit transformations to preprocess input and output variables on the
    # training dataset
    transformations = _setup_transformations(
        data_frames["train"],
        INPUT_VARIABLES + OUTPUT_VARIABLES,
    )

    # Prepare training dataset by applying transformations and creating torch
    # tensors from the data frame. Create separate datasets for training and
    # validation.
    datasets = {}
    for split, data_frame in data_frames.items():
        datasets[split] = _prepare_dataset(
            data_frame,
            INPUT_VARIABLES,
            OUTPUT_VARIABLES,
            {
                "name": "train_weight",
            },
            transformations,
        )

    # Create the neural network model
    model = _create_multi_layer_perceptron_model(
        len(INPUT_VARIABLES),
        len(CATEGORIES),
        NETWORK_HYPERPARAMETERS,
    )

    # Run the training
    training_results = _train_and_validate(
        model,
        datasets["train"],
        datasets["val"],
        TRAINING_HYPERPARAMETERS,
        "cuda" if torch.cuda.is_available() else "cpu",
    )

    # Serialize transformations, checkpoints, and metrics
    _dump_transformations(OUTPUT_DIR / "transformations.pkl", transformations)
    _dump_metrics(OUTPUT_DIR / "metrics.json", training_results["metrics"])
    _dump_checkpoints(OUTPUT_DIR / "checkpoints.pt", training_results["checkpoints"])


if __name__ == "__main__":
    main()

