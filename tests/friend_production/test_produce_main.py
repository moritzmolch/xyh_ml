"""End-to-end tests of the friend production script with the real
``produce`` configuration and small models stored like training artifacts."""

import json
from pathlib import Path

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
from xyh_ml.friend_production import paths
from xyh_ml.friend_production.dataclasses import Sample
from xyh_ml.friend_production.metadata import write_sample_metadata
from xyh_ml.io import dump_transformations
from xyh_ml.nn.mlp import create_multi_layer_perceptron_model

N_EVENTS = 7
PARAMETER_VALUES = [{"m_x": 400, "m_y": 100}, {"m_x": 1000, "m_y": 300}]


def _snapshot_variables(cfg):
    return [
        v.name
        for v in cfg.inputs.variables
        if v.get("is_part_of_snapshot", True)
    ]


# Main files of the metadata: (era, sample, channel, file name) -> number of
# events. The first files of DY in et and mt hold identical events.
MAIN_FILES = {
    ("2024", "DY", "et", "f_0.root"): N_EVENTS,
    ("2024", "DY", "et", "f_1.root"): 4,
    ("2024", "DY", "mt", "f_0.root"): N_EVENTS,
    ("2024", "TT", "et", "f_0.root"): 3,
}
CHANNEL_SEEDS = {"et": 0, "mt": 100}


def _main_file(tmp_path, era, sample, channel, name):
    return str(tmp_path / "ntuples" / era / sample / channel / name)


@pytest.fixture
def setup(tmp_path):
    """Compose the produce config pointing to temporary main files, metadata
    files and training artifacts of two folds for the channels et and mt. The
    config selects the input file DY/et/f_0.root."""
    output_file = tmp_path / "friends" / "2024" / "DY" / "et" / "f_0.root"

    with initialize_config_dir(
        str(PROJECT_DIR / "config"), version_base="1.3"
    ):
        cfg = compose(
            "produce",
            overrides=[
                f"metadata_dir={tmp_path / 'metadata'}",
                "input_files=["
                + _main_file(tmp_path, "2024", "DY", "et", "f_0.root")
                + "]",
                f"output_dir={tmp_path / 'friends'}",
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

    # Main files with all input variables that are part of the ntuples
    variables = _snapshot_variables(cfg)
    samples = {}
    for (era, sample, channel, name), n in MAIN_FILES.items():
        path = _main_file(tmp_path, era, sample, channel, name)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(n)
        with uproot.recreate(path) as f:
            f["ntuple"] = {
                "event": np.arange(100, 100 + n, dtype=np.uint64),
                **{v: rng.normal(size=n) for v in variables},
            }
        samples.setdefault(
            (era, sample, channel), Sample(era, sample, channel, [], {}, [])
        ).main_files.append(path)
    by_sample = {}
    for record in samples.values():
        by_sample.setdefault((record.era, record.sample), []).append(record)
    for (era, sample), records in by_sample.items():
        write_sample_metadata(tmp_path / "metadata", era, sample, records)

    # Training artifacts of every fold and channel
    rng = np.random.default_rng(0)
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
    for channel, seed in CHANNEL_SEEDS.items():
        channel_cfg = OmegaConf.merge(cfg, {"mode": {"channel": channel}})
        for fold, training_dir in produce._fold_training_dirs(
            channel_cfg
        ).items():
            training_dir.mkdir(parents=True)
            torch.manual_seed(seed + fold)
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


def _select(cfg, tmp_path, *files):
    """Select input files given as (sample, channel, file name)."""
    cfg.input_files = [
        _main_file(tmp_path, "2024", sample, channel, name)
        for sample, channel, name in files
    ]
    return cfg


DY_ET = [("DY", "et", "f_0.root"), ("DY", "et", "f_1.root")]
ALL_DY = [*DY_ET, ("DY", "mt", "f_0.root")]


def _friend_arrays(path):
    with uproot.open(path) as f:
        return f["ntuple"].arrays(library="np")


def test_main_writes_friend_for_input_file(setup):
    cfg, output_file = setup

    produce.main(cfg)

    n_classes = len(cfg.outputs.classes)
    with uproot.open(output_file) as f:
        tree = f["ntuple"]
        assert tree.num_entries == N_EVENTS
        assert tree.keys() == [
            f"{c}_{p}"
            for p in ["m_x_400_m_y_100", "m_x_1000_m_y_300"]
            for c in ["max_score", "max_score_index"]
        ]
        arrays = tree.arrays(library="np")

    for p in ["m_x_400_m_y_100", "m_x_1000_m_y_300"]:
        max_score = arrays[f"max_score_{p}"]
        max_score_index = arrays[f"max_score_index_{p}"]
        assert max_score.dtype == np.float32
        assert max_score_index.dtype == np.int32
        # The maximum of a softmax over n classes is in [1/n, 1]
        assert np.all(max_score >= 1.0 / n_classes - 1e-6)
        assert np.all(max_score <= 1.0)
        assert np.all((max_score_index >= 0) & (max_score_index < n_classes))

    # The two parameter value combinations give different scores
    assert not np.allclose(
        arrays["max_score_m_x_400_m_y_100"],
        arrays["max_score_m_x_1000_m_y_300"],
    )


def test_main_writes_profile(setup, tmp_path):
    cfg, _ = setup
    cfg.profile_file = str(tmp_path / "profile" / "profile.json")

    produce.main(cfg)

    profile = json.loads((tmp_path / "profile" / "profile.json").read_text())
    assert profile["n_events"] == N_EVENTS
    assert profile["n_rows"] == N_EVENTS * len(PARAMETER_VALUES)
    # chunk_size=3
    assert profile["n_chunks"] == 3
    assert profile["stages"]["predict"]["per_row_s"] > 0
    assert profile["memory"]["loaded_data_frame_max_bytes"] > 0
    assert profile["memory"]["rss_peak_bytes"] > 0


@pytest.mark.parametrize("channel", ["mt", "tt"])
def test_main_infers_channel_from_input_file(setup, channel):
    cfg, output_file = setup
    produce.main(cfg)
    expected = _friend_arrays(output_file)

    # The input file belongs to channel et: a different mode.channel must not
    # be used to load the models
    cfg.mode.channel = channel
    produce.main(cfg)

    result = _friend_arrays(output_file)
    for key, value in expected.items():
        np.testing.assert_array_equal(result[key], value)


def test_main_infers_channel_for_output_file(setup, tmp_path):
    cfg, output_file = setup
    cfg.mode.channel = "mt"

    produce.main(cfg)

    assert output_file.exists()
    assert not (tmp_path / "friends" / "2024" / "DY" / "mt").exists()


def test_main_unknown_input_file_fails_before_loading_models(setup, tmp_path):
    """All paths are checked before any work starts, also if an earlier file
    is fine."""
    cfg, output_file = setup
    _select(cfg, tmp_path, ("DY", "et", "f_0.root"), ("DY", "et", "f_9.root"))
    # Point the models to a missing directory: the input file lookup has to
    # fail before any model is loaded
    cfg.artifacts.training_dir = "/nonexistent"

    with pytest.raises(ValueError, match="f_9.root not found"):
        produce.main(cfg)

    assert not output_file.exists()


def test_main_trailing_path_of_input_file(setup):
    cfg, output_file = setup
    cfg.input_files = ["2024/DY/et/f_0.root"]

    produce.main(cfg)

    assert output_file.exists()


def test_main_missing_metadata_file_points_to_scraper(setup, tmp_path):
    cfg, _ = setup
    _select(cfg, tmp_path, ("ZZ", "et", "f_0.root"))
    cfg.artifacts.training_dir = "/nonexistent"

    with pytest.raises(ValueError, match="scrape_metadata"):
        produce.main(cfg)


@pytest.mark.parametrize("input_files", [[], "", ["f_0.root"]])
def test_main_invalid_input_files(setup, input_files):
    cfg, _ = setup
    cfg.input_files = input_files
    cfg.artifacts.training_dir = "/nonexistent"

    with pytest.raises(ValueError, match="input_files is empty|does not have"):
        produce.main(cfg)


def test_produce_config_defaults():
    with initialize_config_dir(
        str(PROJECT_DIR / "config"), version_base="1.3"
    ):
        cfg = compose("produce")

    assert list(cfg.input_files) == []
    assert cfg.skip_existing is False
    assert cfg.status_file is None
    for removed in ["input_file", "sample", "campaign", "channel"]:
        assert removed not in cfg
    assert "output_file" not in cfg
    assert cfg.chunk_size > 0
    assert cfg.output_dir.endswith(f"CROWNFriend/{cfg.friend_name}")
    assert cfg.friend_name == (
        f"pnn-{cfg.tags.training_tag}-{cfg.mode.decay_mode}"
    )


def test_produce_config_input_files_from_command_line():
    """A list of URLs is accepted, as produce_many passes them."""
    urls = [
        "root://server:1094//store/2024/A=1/et/f_0.root",
        "root://server:1094//store/2024/B,2/et/f_1.root",
    ]
    quoted = ",".join(f"'{u}'" for u in urls)
    with initialize_config_dir(
        str(PROJECT_DIR / "config"), version_base="1.3"
    ):
        cfg = compose("produce", overrides=[f"input_files=[{quoted}]"])

    assert list(cfg.input_files) == urls


# ------------------------------------------------------ several input files


def test_main_processes_all_input_files(setup, tmp_path):
    cfg = _select(setup[0], tmp_path, *ALL_DY, ("TT", "et", "f_0.root"))

    produce.main(cfg)

    friends = tmp_path / "friends" / "2024"
    assert sorted(
        str(p.relative_to(friends)) for p in friends.rglob("*.root")
    ) == [
        "DY/et/f_0.root",
        "DY/et/f_1.root",
        "DY/mt/f_0.root",
        "TT/et/f_0.root",
    ]
    for (era, sample, channel, name), n in MAIN_FILES.items():
        arrays = _friend_arrays(friends / sample / channel / name)
        assert len(arrays["max_score_m_x_400_m_y_100"]) == n


def test_main_only_processes_selected_files(setup, tmp_path):
    cfg = _select(setup[0], tmp_path, ("DY", "et", "f_1.root"))

    produce.main(cfg)

    friends = tmp_path / "friends" / "2024"
    assert [str(p.relative_to(friends)) for p in friends.rglob("*.root")] == [
        "DY/et/f_1.root"
    ]


def test_main_uses_models_of_the_file_channel(setup, tmp_path):
    cfg = _select(setup[0], tmp_path, *ALL_DY)

    produce.main(cfg)

    # Identical events in et and mt, evaluated with different models
    friends = tmp_path / "friends" / "2024" / "DY"
    et = _friend_arrays(friends / "et" / "f_0.root")
    mt = _friend_arrays(friends / "mt" / "f_0.root")
    assert not np.allclose(
        et["max_score_m_x_400_m_y_100"], mt["max_score_m_x_400_m_y_100"]
    )

    # The et friend equals the one of a job with this file only
    cfg = _select(cfg, tmp_path, ("DY", "et", "f_0.root"))
    produce.main(cfg)
    single = _friend_arrays(friends / "et" / "f_0.root")
    for key, value in single.items():
        np.testing.assert_array_equal(et[key], value)


def test_main_only_loads_models_of_the_channels_that_occur(
    setup, tmp_path, monkeypatch
):
    cfg = _select(setup[0], tmp_path, *DY_ET)
    load_models = produce.load_models
    loaded = []

    def recording_load_models(cfg, channel):
        loaded.append(channel)
        return load_models(cfg, channel)

    monkeypatch.setattr(produce, "load_models", recording_load_models)

    produce.main(cfg)

    assert loaded == ["et"]


def test_main_profile_sums_over_files(setup, tmp_path):
    cfg = _select(setup[0], tmp_path, *DY_ET)
    cfg.profile_file = str(tmp_path / "profile.json")

    produce.main(cfg)

    profile = json.loads((tmp_path / "profile.json").read_text())
    n_events = N_EVENTS + 4
    assert profile["n_events"] == n_events
    assert profile["n_rows"] == n_events * len(PARAMETER_VALUES)
    # chunk_size=3: 3 chunks for 7 events, 2 chunks for 4 events
    assert profile["n_chunks"] == 5


def test_main_skip_existing(setup, tmp_path):
    cfg = _select(setup[0], tmp_path, *DY_ET)
    produce.main(cfg)
    friends = tmp_path / "friends" / "2024" / "DY" / "et"
    # Mark the existing friend of f_0 and remove the one of f_1
    (friends / "f_0.root").write_bytes(b"existing")
    (friends / "f_1.root").unlink()

    cfg.skip_existing = True
    cfg.status_file = str(tmp_path / "status" / "job.json")
    produce.main(cfg)

    assert (friends / "f_0.root").read_bytes() == b"existing"
    assert (
        len(_friend_arrays(friends / "f_1.root")["max_score_m_x_400_m_y_100"])
        == 4
    )
    status = json.loads((tmp_path / "status" / "job.json").read_text())
    assert status == {
        "produced": [cfg.input_files[1]],
        "skipped": [cfg.input_files[0]],
        "failed": [],
    }


def test_main_only_existing_friends_does_not_load_models(setup, tmp_path):
    cfg = _select(setup[0], tmp_path, *DY_ET)
    produce.main(cfg)

    # Loading models would fail
    cfg.artifacts.training_dir = "/nonexistent"
    cfg.skip_existing = True
    cfg.status_file = str(tmp_path / "status.json")
    produce.main(cfg)

    status = json.loads((tmp_path / "status.json").read_text())
    assert status == {
        "produced": [],
        "skipped": list(cfg.input_files),
        "failed": [],
    }


def test_main_overwrites_without_skip_existing(setup, tmp_path):
    cfg = _select(setup[0], tmp_path, *DY_ET)
    friends = tmp_path / "friends" / "2024" / "DY" / "et"
    friends.mkdir(parents=True)
    (friends / "f_0.root").write_bytes(b"existing")

    produce.main(cfg)

    assert (
        len(_friend_arrays(friends / "f_0.root")["max_score_m_x_400_m_y_100"])
        == N_EVENTS
    )


def test_main_continues_after_failure_and_reports_status(setup, tmp_path):
    cfg = _select(setup[0], tmp_path, *DY_ET)
    cfg.status_file = str(tmp_path / "status.json")
    # Corrupt the first input file
    broken = _main_file(tmp_path, "2024", "DY", "et", "f_0.root")
    Path(broken).write_bytes(b"not a ROOT file")

    with pytest.raises(RuntimeError, match="1 file\\(s\\)") as error:
        produce.main(cfg)

    assert broken in str(error.value)
    friends = tmp_path / "friends" / "2024" / "DY" / "et"
    assert not (friends / "f_0.root").exists()
    assert (
        len(_friend_arrays(friends / "f_1.root")["max_score_m_x_400_m_y_100"])
        == 4
    )
    status = json.loads((tmp_path / "status.json").read_text())
    assert status == {
        "produced": [cfg.input_files[1]],
        "skipped": [],
        "failed": [broken],
    }


# ------------------------------------------------------- upload to XRootD

REMOTE = "root://server.example:1094"


@pytest.fixture
def fake_remote(tmp_path, monkeypatch):
    """Store "uploaded" files in a local directory instead of the XRootD
    server. Returns that directory and the list of uploads."""
    remote_dir = tmp_path / "remote"
    uploads = []

    def local_path(url):
        assert url.startswith(REMOTE + "//")
        return remote_dir / url[len(REMOTE + "//") :]

    def upload(local_file, url):
        # The staged file is complete when it is uploaded
        assert local_file.exists() and local_file.parent.name.startswith(
            "produce_"
        )
        uploads.append((local_file, url))
        target = local_path(url)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(local_file.read_bytes())

    monkeypatch.setattr(produce, "xrd_upload", upload)
    monkeypatch.setattr(
        paths, "xrd_exists", lambda url: local_path(url).exists()
    )
    return remote_dir, uploads


def _remote_mode(cfg, tmp_path, *files):
    cfg = _select(cfg, tmp_path, *files)
    cfg.nn_friend_tag = "pnn-test"
    cfg.ntuple.xrootd_server = REMOTE
    cfg.ntuple.base_dir = "/store/user/me/ntuples"
    cfg.staging_dir = str(tmp_path / "staging")
    return cfg


def _remote_friends(remote_dir, cfg):
    return (
        remote_dir
        / "store/user/me/ntuples"
        / cfg.ntuple.tag
        / "CROWNFriend"
        / f"pnn-test-{cfg.mode.decay_mode}"
    )


def test_main_uploads_friends_with_nn_friend_tag(setup, tmp_path, fake_remote):
    remote_dir, uploads = fake_remote
    cfg = _remote_mode(setup[0], tmp_path, *ALL_DY)

    produce.main(cfg)

    friends = _remote_friends(remote_dir, cfg) / "2024" / "DY"
    assert sorted(
        str(p.relative_to(friends)) for p in friends.rglob("*.root")
    ) == ["et/f_0.root", "et/f_1.root", "mt/f_0.root"]
    assert len(uploads) == 3
    # Nothing is written to the local output directory, and the staging
    # directory is cleaned up
    assert not (tmp_path / "friends").exists()
    assert list((tmp_path / "staging").iterdir()) == []

    # The uploaded friend equals the locally written one
    cfg.nn_friend_tag = None
    produce.main(cfg)
    local = _friend_arrays(tmp_path / "friends/2024/DY/et/f_0.root")
    remote = _friend_arrays(friends / "et" / "f_0.root")
    for key, value in local.items():
        np.testing.assert_array_equal(remote[key], value)


def test_main_upload_skip_existing_checks_remote(setup, tmp_path, fake_remote):
    remote_dir, uploads = fake_remote
    cfg = _remote_mode(setup[0], tmp_path, *DY_ET)
    existing = _remote_friends(remote_dir, cfg) / "2024/DY/et/f_0.root"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"old")
    cfg.skip_existing = True

    produce.main(cfg)

    assert [url.rsplit("/", 2)[-2:] for _, url in uploads] == [
        ["et", "f_1.root"]
    ]
    assert existing.read_bytes() == b"old"


def test_main_upload_failure_continues_and_cleans_up(
    setup, tmp_path, fake_remote, monkeypatch
):
    _, uploads = fake_remote
    upload = produce.xrd_upload

    def failing_upload(local_file, url):
        if url.endswith("et/f_0.root"):
            raise RuntimeError("upload failed")
        upload(local_file, url)

    monkeypatch.setattr(produce, "xrd_upload", failing_upload)
    cfg = _remote_mode(setup[0], tmp_path, *ALL_DY)

    with pytest.raises(RuntimeError, match="1 file"):
        produce.main(cfg)

    assert len(uploads) == 2
    assert list((tmp_path / "staging").iterdir()) == []


def test_produce_config_nn_friend_tag_from_command_line():
    with initialize_config_dir(
        str(PROJECT_DIR / "config"), version_base="1.3"
    ):
        cfg = compose(
            "produce",
            overrides=[
                "nn_friend_tag=pnn-2026-11-04",
                "inputs=y2tau_h2b",
                "mode.decay_mode=y2tau_h2b",
            ],
        )

    assert produce.resolve_output_dir(cfg) == (
        f"{cfg.ntuple.xrootd_server}//{cfg.ntuple.base_dir.strip('/')}/"
        + f"{cfg.ntuple.tag}/CROWNFriend/pnn-2026-11-04-y2tau_h2b"
    )
