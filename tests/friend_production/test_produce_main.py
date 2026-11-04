"""End-to-end tests of the friend production script with the real
``produce`` configuration and small models stored like training artifacts."""

import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch
import uproot
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from xyh_ml import PROJECT_DIR
from xyh_ml.cli.friends import produce
from xyh_ml.data.transformations import derive_transformations
from xyh_ml.friend_production.dataclasses import Sample
from xyh_ml.io import dump_json, dump_transformations
from xyh_ml.nn.mlp import create_multi_layer_perceptron_model

N_EVENTS = 7
PARAMETER_VALUES = [{"m_x": 400, "m_y": 100}, {"m_x": 1000, "m_y": 300}]


def _snapshot_variables(cfg):
    return [
        v.name
        for v in cfg.inputs.variables
        if v.get("is_part_of_snapshot", True)
    ]


@pytest.fixture
def setup(tmp_path):
    """Compose the produce config pointing to a temporary main file,
    metadata file and training artifacts of two folds."""
    rng = np.random.default_rng(0)
    input_file = str(tmp_path / "ntuples" / "2024" / "DY" / "et" / "f_0.root")
    output_file = tmp_path / "friend" / "f_0.root"

    with initialize_config_dir(
        str(PROJECT_DIR / "config"), version_base="1.3"
    ):
        cfg = compose(
            "produce",
            overrides=[
                f"metadata_file={tmp_path / 'metadata.json'}",
                f"input_file={input_file}",
                f"output_file={output_file}",
                "artifacts.training_dir="
                + str(tmp_path / "training")
                + "/fold${mode.fold}of${mode.n_folds}/${mode.channel}",
                "mode.channel=et",
                "mode.n_folds=2",
                "network.hidden_layers=[4]",
                "network.dropout_frac=0.0",
                "chunk_size=3",
            ],
        )
    cfg.parameter_values = PARAMETER_VALUES

    # Main file with all input variables that are part of the ntuples
    variables = _snapshot_variables(cfg)
    (tmp_path / "ntuples" / "2024" / "DY" / "et").mkdir(parents=True)
    with uproot.recreate(input_file) as f:
        f["ntuple"] = {
            "event": np.arange(100, 100 + N_EVENTS, dtype=np.uint64),
            **{v: rng.normal(size=N_EVENTS) for v in variables},
        }

    # Metadata with the main file of the input sample and another sample
    sample = Sample("2024", "DY", "et", [input_file], {}, [])
    other = dataclasses.replace(
        sample, channel="mt", main_files=[input_file.replace("et", "mt")]
    )
    dump_json(
        tmp_path / "metadata.json",
        [dataclasses.asdict(sample), dataclasses.asdict(other)],
    )

    # Training artifacts of every fold
    n_train = 20
    campaigns = [c.name for c in cfg.campaigns]
    classes = [c.name for c in cfg.outputs.classes]
    train = pd.DataFrame(
        {
            **{v: rng.normal(size=n_train) for v in variables},
            "campaign": [
                campaigns[i % len(campaigns)] for i in range(n_train)
            ],
            "m_x": rng.choice([400.0, 1000.0], n_train),
            "m_y": rng.choice([100.0, 300.0], n_train),
            "class": [classes[i % len(classes)] for i in range(n_train)],
        }
    )
    transformations = derive_transformations(
        train,
        list(cfg.inputs.variables)
        + list(cfg.inputs.parameters)
        + list(cfg.outputs.variables),
    )
    for fold, training_dir in produce._fold_training_dirs(cfg).items():
        training_dir.mkdir(parents=True)
        torch.manual_seed(fold)
        model = create_multi_layer_perceptron_model(
            len(cfg.inputs.variables) + len(cfg.inputs.parameters),
            len(cfg.outputs.classes),
            cfg.network,
        )
        torch.save(
            {
                "state_dict": {
                    f"model.{k}": v for k, v in model.state_dict().items()
                }
            },
            training_dir / "best_model.ckpt",
        )
        dump_transformations(
            training_dir / "transformations.pkl", transformations
        )

    return cfg, output_file


def test_main_writes_friend_for_input_file(setup):
    cfg, output_file = setup

    produce.main(cfg)

    classes = [c.name for c in cfg.outputs.classes]
    with uproot.open(output_file) as f:
        tree = f["ntuple"]
        assert tree.num_entries == N_EVENTS
        assert len(tree.keys()) == len(PARAMETER_VALUES) * (len(classes) + 1)
        arrays = tree.arrays(library="np")

    for p in ["m_x_400_m_y_100", "m_x_1000_m_y_300"]:
        scores = np.stack([arrays[f"score_{c}_{p}"] for c in classes], axis=1)
        np.testing.assert_allclose(scores.sum(axis=1), 1.0, rtol=1e-5)
        np.testing.assert_array_equal(
            arrays[f"max_score_index_{p}"], np.argmax(scores, axis=1)
        )

    # The two parameter value combinations give different scores
    assert not np.allclose(
        arrays[f"score_{classes[0]}_m_x_400_m_y_100"],
        arrays[f"score_{classes[0]}_m_x_1000_m_y_300"],
    )


def test_main_default_output_file(setup, tmp_path):
    cfg, _ = setup
    cfg.output_file = None
    cfg.output_dir = str(tmp_path / "friends")

    produce.main(cfg)

    with uproot.open(
        tmp_path / "friends" / "2024" / "DY" / "et" / "f_0.root"
    ) as f:
        assert f["ntuple"].num_entries == N_EVENTS


def test_main_rejects_other_channel(setup):
    cfg, output_file = setup
    cfg.mode.channel = "mt"
    # Point the models to a missing directory: the channel check has to
    # fail before any model is loaded
    cfg.artifacts.training_dir = "/nonexistent"

    with pytest.raises(ValueError, match="belongs to channel et"):
        produce.main(cfg)

    assert not output_file.exists()


def test_produce_config_defaults():
    with initialize_config_dir(
        str(PROJECT_DIR / "config"), version_base="1.3"
    ):
        cfg = compose("produce", overrides=["input_file=x.root"])

    assert cfg.output_file is None
    assert cfg.chunk_size > 0
    assert cfg.output_dir.endswith(f"CROWNFriend/{cfg.friend_name}")
    assert cfg.friend_name == (
        f"pnn-{cfg.tags.training_tag}-{cfg.mode.decay_mode}"
    )
    assert OmegaConf.is_missing(compose_without_input_file(), "input_file")


def compose_without_input_file():
    with initialize_config_dir(
        str(PROJECT_DIR / "config"), version_base="1.3"
    ):
        return compose("produce")
