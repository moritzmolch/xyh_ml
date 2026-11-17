import logging
from pathlib import Path

import hydra
import pandas as pd
import torch

from xyh_ml import PROJECT_DIR
from xyh_ml.cli.evaluate_test_sample import (
    filter_and_set_evaluation_parameters,
    load_trained_model,
    predict,
)
from xyh_ml.config import Config
from xyh_ml.data.data_frame import (
    get_unique_input_parameter_combinations,
    load_data_frame,
)
from xyh_ml.data.torch import create_torch_dataset
from xyh_ml.io import load_transformations
from xyh_ml.taylor import (
    compute_taylor_coefficients,
    get_taylor_coefficients_file,
    select_benchmark_mass_points,
)

# Set up logging
logger = logging.getLogger(__name__)


@hydra.main(
    config_path=str(PROJECT_DIR / "config"),
    config_name="evaluate_taylor_coefficients",
    version_base="1.3",
)
def main(cfg: Config):
    # Create output directory if it does not exist
    taylor_dir = Path(cfg.artifacts.taylor_dir)
    taylor_dir.mkdir(parents=True, exist_ok=True)

    # Load transformations from the training artifacts
    transformations = load_transformations(
        Path(cfg.artifacts.training_dir) / "transformations.pkl"
    )

    # Create the neural network model and load the best checkpoint
    model = load_trained_model(cfg)

    # Load the held-out fold
    data_frame = load_data_frame(
        Path(cfg.artifacts.data_frame_dir),
        cfg.mode.decay_mode,
        cfg.outputs.classes,
        cfg.mode.n_folds,
        cfg.mode.fold,
        "eval",
        cfg.mode.channel,
    )

    # Restrict to the benchmark mass points
    unique_parameters = get_unique_input_parameter_combinations(
        data_frame,
        cfg.inputs.parameters,
    )
    data_frame, unique_parameters = select_benchmark_mass_points(
        data_frame,
        unique_parameters,
        [list(m) for m in cfg.taylor.mass_points],
    )

    # Sample events and set the parameters of the background events
    data_frame = filter_and_set_evaluation_parameters(
        data_frame,
        unique_parameters,
        cfg.mode.n_folds,
        n_samples_per_category=cfg.taylor.n_samples_per_category,
    )

    dataset = create_torch_dataset(
        data_frame,
        cfg.inputs.variables,
        cfg.inputs.parameters,
        cfg.outputs.variables[0],
        "__total_weight",
        transformations,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Softmax scores and targets
    predictions, targets, _ = predict(
        model, dataset, device, batch_size=cfg.taylor.batch_size
    )
    t = transformations["ordinal_encoding"]["transformation"]
    j = t.feature_names_in_.tolist().index(cfg.outputs.variables[0].name)
    class_names = t.categories_[j].tolist()
    scores = {
        f"score_{c}": predictions[:, i] for i, c in enumerate(class_names)
    }
    scores["target"] = targets.reshape(len(targets), -1)[:, 0]

    # Taylor coefficients with respect to the input variables, and the input
    # parameters (m_x, m_y) only if taylor.include_parameters is set
    variable_names = [v.name for v in cfg.inputs.variables]
    parameter_names = [p.name for p in cfg.inputs.parameters]
    input_names = variable_names + parameter_names
    tc_input_names = (
        input_names if cfg.taylor.include_parameters else variable_names
    )
    tcs = compute_taylor_coefficients(
        model,
        dataset,
        input_names,
        class_names,
        device,
        batch_size=cfg.taylor.batch_size,
        max_order=cfg.taylor.max_order,
        tc_input_names=tc_input_names,
    )

    data_frame = pd.concat(
        [
            data_frame,
            pd.DataFrame(scores, index=data_frame.index),
            pd.DataFrame(tcs, index=data_frame.index),
        ],
        axis=1,
    )
    data_frame.to_hdf(
        get_taylor_coefficients_file(cfg), key="taylor", mode="w"
    )


if __name__ == "__main__":
    main()
