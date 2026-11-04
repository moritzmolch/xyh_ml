"""Tests for the parallel friend production wrapper."""

import json
import logging
import sys
import threading
import time
from pathlib import Path

import pytest

from xyh_ml.cli.friends import produce_many as pm
from xyh_ml.config.dataclasses import Campaign
from xyh_ml.friend_production.dataclasses import Sample
from xyh_ml.io import dump_json

CAMPAIGNS = [
    Campaign(name="2024_nano_v15", era="2024"),
    Campaign(name="2025_nano_v15", era="2025"),
]


def _sample(era, sample, channel, n_files):
    main_dir = f"root://server//store/ntuples/{era}/{sample}/{channel}"
    return Sample(
        era=era,
        sample=sample,
        channel=channel,
        main_files=[f"{main_dir}/{sample}_{i}.root" for i in range(n_files)],
        friend_files={},
        shifts=[],
    )


def _samples():
    return [
        _sample("2024", "DY", "et", 2),
        _sample("2024", "DY", "mt", 1),
        _sample("2025", "DY", "et", 1),
        _sample("2024", "TT", "et", 4),
        _sample("2025", "TT", "tt", 1),
        _sample("2023", "TT", "et", 1),  # era without campaign
        _sample("2024", "WJets", "et", 1),
    ]


# ------------------------------------------------------------ job selection


def test_read_samples_file(tmp_path):
    path = tmp_path / "samples.txt"
    path.write_text(
        "DY_2024-v1\n"
        "\n"
        "# comment\n"
        "  TT_2024-v2  \n"
        "/GluGluH-Hto2B_Par-M-125/RunIII2024Summer24-v2/NANOAODSIM\n"
    )

    assert pm.read_samples_file(path) == [
        "DY_2024-v1",
        "TT_2024-v2",
        "GluGluH-Hto2B_Par-M-125_RunIII2024Summer24-v2",
    ]


def test_select_sample_names():
    available = ["DY", "DYto2L", "TT", "WJets"]

    selected, missing = pm.select_sample_names(
        available, ["DY.*"], ["WJets", "ZZ", "TT"]
    )

    assert selected == ["DY", "DYto2L", "TT", "WJets"]
    assert missing == ["ZZ"]


def test_select_sample_names_matches_whole_name():
    selected, _ = pm.select_sample_names(["DY", "DYto2L", "xDY"], ["DY"], [])

    assert selected == ["DY"]


def test_build_jobs_one_job_per_sample_and_campaign(tmp_path):
    jobs = pm.build_jobs(_samples(), ["DY", "TT"], CAMPAIGNS, tmp_path)

    # Largest jobs first; the TT sample of 2023 has no campaign
    assert [(j.sample, j.campaign, j.channels, j.n_files) for j in jobs] == [
        ("TT", "2024_nano_v15", ["et"], 4),
        ("DY", "2024_nano_v15", ["et", "mt"], 3),
        ("DY", "2025_nano_v15", ["et"], 1),
        ("TT", "2025_nano_v15", ["tt"], 1),
    ]
    assert jobs[1].name == "2024_nano_v15__DY"
    assert jobs[1].output_files == [
        str(tmp_path / "2024/DY/et/DY_0.root"),
        str(tmp_path / "2024/DY/et/DY_1.root"),
        str(tmp_path / "2024/DY/mt/DY_0.root"),
    ]


@pytest.mark.parametrize(
    "campaigns, channels, expected",
    [
        (["2025_nano_v15"], None, [("DY", "2025"), ("TT", "2025")]),
        (None, ["mt", "tt"], [("DY", "2024"), ("TT", "2025")]),
        (["2024_nano_v15"], ["et"], [("DY", "2024"), ("TT", "2024")]),
    ],
)
def test_build_jobs_restricted(tmp_path, campaigns, channels, expected):
    jobs = pm.build_jobs(
        _samples(),
        ["DY", "TT"],
        CAMPAIGNS,
        tmp_path,
        campaign_names=campaigns,
        channels=channels,
    )

    assert sorted((j.sample, j.era) for j in jobs) == expected
    for job in jobs:
        assert channels is None or set(job.channels) <= set(channels)


def test_build_jobs_channel_restriction_reduces_files(tmp_path):
    (job,) = pm.build_jobs(
        _samples(),
        ["DY"],
        CAMPAIGNS,
        tmp_path,
        campaign_names=["2024_nano_v15"],
        channels=["mt"],
    )

    assert (job.channels, job.n_files) == (["mt"], 1)


def test_build_jobs_unknown_campaign(tmp_path):
    with pytest.raises(ValueError, match="Unknown campaigns \\['2024'\\]"):
        pm.build_jobs(
            _samples(), ["DY"], CAMPAIGNS, tmp_path, campaign_names=["2024"]
        )


def test_build_jobs_nothing_selected(tmp_path):
    assert pm.build_jobs(_samples(), [], CAMPAIGNS, tmp_path) == []


def _job(name="DY", channels=("et", "mt")):
    return pm.Job(
        sample=name,
        era="2024",
        campaign="2024_nano_v15",
        channels=list(channels),
        n_files=2,
        output_files=[],
    )


def test_job_command(tmp_path):
    command = pm.job_command(
        _job("NMSSM-X_Par-MX-1000"),
        ["mode.decay_mode=y2b_h2tau", "inputs=y2b_h2tau"],
        tmp_path,
        skip_existing=True,
        profile=False,
        python="python",
    )

    assert command == [
        "python",
        "-m",
        "xyh_ml.cli.friends.produce",
        "mode.decay_mode=y2b_h2tau",
        "inputs=y2b_h2tau",
        "sample='NMSSM-X_Par-MX-1000'",
        "campaign=2024_nano_v15",
        "channel=[et,mt]",
        "skip_existing=true",
        f"hydra.run.dir={tmp_path}/hydra/2024_nano_v15__NMSSM-X_Par-MX-1000",
    ]


def test_job_command_with_profile(tmp_path):
    command = pm.job_command(
        _job(), [], tmp_path, skip_existing=False, profile=True
    )

    assert "skip_existing=false" in command
    assert command[-1] == (
        f"profile_file={tmp_path}/profiles/2024_nano_v15__DY.json"
    )


def test_job_command_is_parsed_by_hydra(tmp_path):
    """The overrides of a job are accepted by the produce config."""
    command = pm.job_command(
        _job("NMSSM-X_Par-MX-1000"), [], tmp_path, True, True
    )
    overrides = [o for o in command[3:] if not o.startswith("hydra.run.dir")]

    cfg = pm._compose(overrides)

    assert cfg.sample == "NMSSM-X_Par-MX-1000"
    assert cfg.campaign == "2024_nano_v15"
    assert list(cfg.channel) == ["et", "mt"]
    assert cfg.skip_existing is True
    assert cfg.profile_file == str(
        tmp_path / "profiles" / "2024_nano_v15__NMSSM-X_Par-MX-1000.json"
    )


@pytest.mark.parametrize(
    "override",
    [
        "sample=DY",
        "campaign=2024_nano_v15",
        "channel=[et]",
        "input_file=x.root",
        "+output_file=x.root",
        "skip_existing=true",
        "hydra.run.dir=x",
    ],
)
def test_check_overrides_rejects_per_job_overrides(override):
    with pytest.raises(ValueError, match="set per job"):
        pm.check_overrides(["inputs=y2b_h2tau", override])


def test_check_overrides_accepts_others():
    pm.check_overrides(
        ["mode.decay_mode=y2b_h2tau", "chunk_size=5000", "output_dir=x"]
    )


# ---------------------------------------------------------------- execution


def _python(code):
    return [sys.executable, "-c", code]


def test_runner_statuses_and_logs(tmp_path):
    runner = pm.Runner(2, None, None, tmp_path)
    jobs = [
        (_job("A"), _python("print('hello')")),
        (_job("B"), _python("import sys; print('bad'); sys.exit(3)")),
    ]

    results = {r.name: r for r in runner.run(jobs)}

    a, b = results["2024_nano_v15__A"], results["2024_nano_v15__B"]
    assert (a.status, a.returncode) == ("ok", 0)
    assert (b.status, b.returncode) == ("failed", 3)
    assert "hello" in Path(a.log_file).read_text()
    assert Path(b.log_file) == tmp_path / "logs" / "2024_nano_v15__B.log"
    # The log starts with the command
    assert Path(b.log_file).read_text().startswith(sys.executable)


def test_runner_environment_gpus_and_threads(tmp_path):
    runner = pm.Runner(2, ["2", "3"], 1, tmp_path)
    code = (
        "import os; print('GPU', os.environ['CUDA_VISIBLE_DEVICES'], "
        "os.environ['OMP_NUM_THREADS']); import time; time.sleep(0.5)"
    )
    jobs = [(_job(f"S{i}"), _python(code)) for i in range(4)]

    results = runner.run(jobs)

    for result in results:
        assert f"GPU {result.gpu} 1" in Path(result.log_file).read_text()
    # Two jobs at a time, each on its own GPU
    assert sorted(r.gpu for r in results) == ["2", "2", "3", "3"]


def test_runner_without_gpus_keeps_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    runner = pm.Runner(1, None, None, tmp_path)
    code = "import os; print('GPU', os.environ['CUDA_VISIBLE_DEVICES'])"

    (result,) = runner.run([(_job(), _python(code))])

    assert result.gpu is None
    assert "GPU 7" in Path(result.log_file).read_text()


def test_runner_limits_parallel_jobs(tmp_path):
    runner = pm.Runner(2, None, None, tmp_path)
    running, peak = [0], [0]
    lock = threading.Lock()
    run_job = runner.run_job

    def counting_run_job(job, command):
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        try:
            return run_job(job, command)
        finally:
            with lock:
                running[0] -= 1

    runner.run_job = counting_run_job
    sleep = _python("import time; time.sleep(0.3)")

    results = runner.run([(_job(f"S{i}"), sleep) for i in range(5)])

    assert len(results) == 5
    assert peak[0] == 2


def test_runner_stop_terminates_running_jobs(tmp_path):
    runner = pm.Runner(1, None, None, tmp_path)
    result = []
    thread = threading.Thread(
        target=lambda: result.append(
            runner.run_job(_job(), _python("import time; time.sleep(60)"))
        )
    )
    thread.start()
    while not runner.processes:
        time.sleep(0.01)

    runner.stop()
    thread.join(timeout=10)

    assert result[0].status == "failed"
    # A job that has not started yet is skipped after stop
    skipped = runner.run_job(_job("B"), _python("pass"))
    assert (skipped.status, skipped.log_file) == ("skipped", None)


def test_runner_invalid_number_of_jobs(tmp_path):
    with pytest.raises(ValueError, match="n_parallel"):
        pm.Runner(0, None, None, tmp_path)


# ---------------------------------------------------------------------- main


@pytest.fixture
def metadata(tmp_path):
    path = tmp_path / "metadata.json.gz"
    dump_json(path, [s.__dict__ | {"shifts": []} for s in _samples()])
    return path


def _overrides(metadata, tmp_path):
    return [
        f"metadata_file={metadata}",
        f"output_dir={tmp_path / 'friends'}",
    ]


def test_main_requires_sample_selection(metadata, tmp_path):
    with pytest.raises(SystemExit):
        pm.main(["--dry-run", *_overrides(metadata, tmp_path)])


def test_main_rejects_per_job_override(metadata, tmp_path, caplog):
    args = ["--sample", "DY", "sample=TT", *_overrides(metadata, tmp_path)]

    assert pm.main(args) == 2
    assert "set per job" in caplog.text


def test_main_dry_run(metadata, tmp_path, capsys):
    samples_file = tmp_path / "samples.txt"
    samples_file.write_text("TT\nZZ\n")

    code = pm.main(
        [
            "--sample",
            "D.*",
            "--samples-file",
            str(samples_file),
            "--campaign",
            "2024_nano_v15",
            "--dry-run",
            *_overrides(metadata, tmp_path),
        ]
    )

    out = capsys.readouterr().out
    assert code == 0
    assert out.splitlines()[0] == (
        "2024_nano_v15__TT: 4 files, channels ['et']"
    )
    assert "2024_nano_v15__DY: 3 files, channels ['et', 'mt']" in out
    assert "sample='DY' campaign=2024_nano_v15 channel=[et,mt]" in out
    assert "WJets" not in out
    assert not (tmp_path / "logs").exists()


def _fake_job_command(job, overrides, log_dir, skip_existing, profile):
    """Write the friend files of the job, or fail for sample TT."""
    code = (
        "import sys, pathlib\n"
        f"if {job.sample!r} == 'TT': sys.exit(1)\n"
        f"for f in {job.output_files!r}:\n"
        "    p = pathlib.Path(f); p.parent.mkdir(parents=True, "
        "exist_ok=True); p.touch()\n"
    )
    return _python(code)


def test_main_runs_jobs_and_writes_summary(
    metadata, tmp_path, monkeypatch, caplog
):
    monkeypatch.setattr(pm, "job_command", _fake_job_command)
    log_dir = tmp_path / "run"
    args = [
        "--sample",
        "DY|TT",
        "--gpus",
        "0,1",
        "-j",
        "2",
        "--log-dir",
        str(log_dir),
        *_overrides(metadata, tmp_path),
    ]

    with caplog.at_level(logging.INFO):
        code = pm.main(args)

    summary = json.loads((log_dir / "summary.json").read_text())
    statuses = {j["name"]: j["status"] for j in summary["jobs"]}
    assert code == 1
    assert statuses == {
        "2024_nano_v15__DY": "ok",
        "2025_nano_v15__DY": "ok",
        "2024_nano_v15__TT": "failed",
        "2025_nano_v15__TT": "failed",
    }
    assert (summary["n_jobs"], summary["n_failed"]) == (4, 2)
    assert (tmp_path / "friends/2024/DY/mt/DY_0.root").exists()
    assert "2 jobs failed" in caplog.text

    # Repeating with --skip-existing only reruns the failed jobs
    caplog.clear()
    log_dir_2 = tmp_path / "run2"
    args[args.index(str(log_dir))] = str(log_dir_2)
    with caplog.at_level(logging.INFO):
        assert pm.main(["--skip-existing", *args]) == 1

    summary = json.loads((log_dir_2 / "summary.json").read_text())
    assert sorted(j["name"] for j in summary["jobs"]) == [
        "2024_nano_v15__TT",
        "2025_nano_v15__TT",
    ]
    assert "Skipping 2 jobs whose friends all exist" in caplog.text


def test_main_nothing_to_do(metadata, tmp_path, caplog):
    assert pm.main(["--sample", "ZZ", *_overrides(metadata, tmp_path)]) == 0
    assert "No jobs to run" in caplog.text


# ------------------------------------------------------- upload to XRootD


def test_build_jobs_remote_output_dir():
    output_dir = "root://server:1094//store/CROWNFriend/pnn-test-y2b_h2tau"

    (job,) = pm.build_jobs(
        _samples(),
        ["DY"],
        CAMPAIGNS,
        output_dir,
        campaign_names=["2025_nano_v15"],
    )

    assert job.output_files == [f"{output_dir}/2025/DY/et/DY_0.root"]


def test_main_nn_friend_tag(metadata, tmp_path, monkeypatch, capsys, caplog):
    """With nn_friend_tag, jobs write to the XRootD server, which is also
    checked for existing friends."""
    checked = []

    def exists(url):
        checked.append(url)
        return "/2024/" in url

    monkeypatch.setattr(pm, "friend_exists", exists)
    args = [
        "--sample",
        "DY",
        "--skip-existing",
        "--dry-run",
        *_overrides(metadata, tmp_path),
        "nn_friend_tag=pnn-test",
        "ntuple.xrootd_server=root://server:1094",
        "ntuple.base_dir=/store/me",
    ]

    with caplog.at_level(logging.INFO):
        assert pm.main(args) == 0

    out = capsys.readouterr().out
    remote = "root://server:1094//store/me/ntuples-2026-10-08/CROWNFriend"
    assert all(
        url.startswith(f"{remote}/pnn-test-y2b_h2tau/") for url in checked
    )
    # The 2024 job is complete, the 2025 job is run with the tag
    assert "Skipping 1 jobs" in caplog.text
    assert out.splitlines()[0].startswith("2025_nano_v15__DY:")
    assert "nn_friend_tag=pnn-test" in out
    assert f"friends in {remote}/pnn-test-y2b_h2tau" in caplog.text


def test_main_invalid_nn_friend_tag(metadata, tmp_path, caplog):
    args = [
        "--sample",
        "DY",
        *_overrides(metadata, tmp_path),
        "nn_friend_tag=a/b",
    ]

    assert pm.main(args) == 2
    assert "Invalid nn_friend_tag" in caplog.text
