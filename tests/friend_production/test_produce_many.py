"""Tests for the parallel friend production wrapper."""

import json
import logging
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from xyh_ml import PROJECT_DIR
from xyh_ml.cli.friends import produce_many as pm
from xyh_ml.config.dataclasses import Campaign
from xyh_ml.friend_production import paths
from xyh_ml.friend_production.dataclasses import Sample
from xyh_ml.friend_production.metadata import (
    read_samples_file,
    write_sample_metadata,
)

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

    assert read_samples_file(path) == [
        "DY_2024-v1",
        "TT_2024-v2",
        "GluGluH-Hto2B_Par-M-125_RunIII2024Summer24-v2",
    ]


def _write_metadata(metadata_dir, samples=None):
    by_sample = {}
    for s in samples if samples is not None else _samples():
        by_sample.setdefault((s.era, s.sample), []).append(s)
    for (era, sample), records in by_sample.items():
        write_sample_metadata(metadata_dir, era, sample, records)


def test_select_samples_by_pattern_and_name(tmp_path):
    _write_metadata(tmp_path)

    selected, missing = pm.select_samples(
        tmp_path, ["2024", "2025"], ["W.*"], ["DY", "ZZ"]
    )

    assert selected == [("2024", "DY"), ("2024", "WJets"), ("2025", "DY")]
    assert missing == ["ZZ"]


def test_select_samples_only_selected_eras(tmp_path):
    _write_metadata(tmp_path)

    selected, missing = pm.select_samples(
        tmp_path, ["2025"], [".*"], ["WJets"]
    )

    assert selected == [("2025", "DY"), ("2025", "TT")]
    assert missing == ["WJets"]


def test_select_samples_pattern_matches_whole_name(tmp_path):
    _write_metadata(tmp_path, [_sample("2024", "DYto2L", "et", 1)])

    assert pm.select_samples(tmp_path, ["2024"], ["DY"], []) == ([], [])


def test_select_samples_names_do_not_list_directories(tmp_path, monkeypatch):
    _write_metadata(tmp_path)

    def no_listing(*args):
        raise AssertionError("The metadata directories must not be listed")

    monkeypatch.setattr(pm, "available_samples", no_listing)

    selected, _ = pm.select_samples(tmp_path, ["2024"], [], ["TT"])

    assert selected == [("2024", "TT")]


def _input_files(tmp_path, samples=None, **kwargs):
    return pm.list_input_files(
        _samples() if samples is None else samples,
        CAMPAIGNS,
        tmp_path,
        **kwargs,
    )


def _selected_samples():
    return [s for s in _samples() if s.era != "2023"]


def test_list_input_files(tmp_path):
    files = _input_files(tmp_path, [_sample("2024", "DY", "et", 2)])

    assert files == [
        pm.InputFile(
            era="2024",
            campaign="2024_nano_v15",
            sample="DY",
            channel="et",
            input_file=(
                "root://server//store/ntuples/2024/DY/et/" + f"DY_{i}.root"
            ),
            output_file=str(tmp_path / f"2024/DY/et/DY_{i}.root"),
        )
        for i in range(2)
    ]


def test_list_input_files_restricted_channels(tmp_path):
    files = _input_files(tmp_path, _selected_samples(), channels=["mt", "tt"])

    assert sorted((f.era, f.sample, f.channel) for f in files) == [
        ("2024", "DY", "mt"),
        ("2025", "TT", "tt"),
    ]


def test_list_input_files_remote_output_dir():
    output_dir = "root://server:1094//store/CROWNFriend/pnn-test-y2b_h2tau"

    (file,) = pm.list_input_files(
        [_sample("2025", "DY", "et", 1)], CAMPAIGNS, output_dir
    )

    assert file.output_file == f"{output_dir}/2025/DY/et/DY_0.root"


def test_drop_existing_local(tmp_path):
    files = _input_files(tmp_path, [_sample("2024", "DY", "et", 3)])
    Path(files[1].output_file).parent.mkdir(parents=True)
    Path(files[1].output_file).touch()

    assert pm.drop_existing(files) == [files[0], files[2]]


def test_drop_existing_missing_directory(tmp_path):
    files = _input_files(tmp_path, [_sample("2024", "DY", "et", 2)])

    assert pm.drop_existing(files) == files


def test_drop_existing_lists_each_directory_once(tmp_path, monkeypatch):
    files = _input_files(
        tmp_path,
        [_sample("2024", "DY", "et", 3), _sample("2024", "DY", "mt", 2)],
    )
    listed = []

    def listing(directory):
        listed.append(directory)
        return {"DY_0.root"}

    monkeypatch.setattr(pm, "list_directory", listing)

    remaining = pm.drop_existing(files)

    assert listed == [
        str(tmp_path / "2024/DY/et"),
        str(tmp_path / "2024/DY/mt"),
    ]
    assert len(remaining) == 3


def test_build_jobs_block_sizes(tmp_path):
    files = _input_files(tmp_path, [_sample("2024", "DY", "et", 5)])

    jobs = pm.build_jobs(files, n_files=2)

    assert [len(j.input_files) for j in jobs] == [2, 2, 1]
    assert [j.name for j in jobs] == [
        "2024_nano_v15__et__00000",
        "2024_nano_v15__et__00001",
        "2024_nano_v15__et__00002",
    ]
    assert jobs[2].output_files == [str(tmp_path / "2024/DY/et/DY_4.root")]


def test_build_jobs_do_not_mix_eras_and_channels(tmp_path):
    files = _input_files(tmp_path, _selected_samples())

    jobs = pm.build_jobs(files, n_files=100)

    assert [(j.campaign, j.channel) for j in jobs] == [
        ("2024_nano_v15", "et"),
        ("2024_nano_v15", "mt"),
        ("2025_nano_v15", "et"),
        ("2025_nano_v15", "tt"),
    ]
    for job in jobs:
        for input_file in job.input_files:
            assert f"/{job.era}/" in input_file
            assert f"/{job.channel}/" in input_file


def test_build_jobs_span_samples(tmp_path):
    files = _input_files(
        tmp_path,
        [
            _sample("2024", "TT", "et", 2),
            _sample("2024", "DY", "et", 3),
        ],
    )

    jobs = pm.build_jobs(files, n_files=4)

    assert [j.samples for j in jobs] == [["DY", "TT"], ["TT"]]
    # Files of a sample stay together
    assert [f.rsplit("/", 1)[-1] for f in jobs[0].input_files] == [
        "DY_0.root",
        "DY_1.root",
        "DY_2.root",
        "TT_0.root",
    ]


@pytest.mark.parametrize("n_files", [1, 2, 3, 100])
def test_build_jobs_cover_all_files_once(tmp_path, n_files):
    files = _input_files(tmp_path, _selected_samples())

    jobs = pm.build_jobs(files, n_files)

    assert sorted(f for j in jobs for f in j.input_files) == sorted(
        f.input_file for f in files
    )
    assert all(1 <= len(j.input_files) <= n_files for j in jobs)
    assert len({j.name for j in jobs}) == len(jobs)


def test_build_jobs_nothing_selected():
    assert pm.build_jobs([], n_files=2) == []


@pytest.mark.parametrize("n_files", [0, -1])
def test_build_jobs_invalid_n_files(n_files):
    with pytest.raises(ValueError, match="n_files"):
        pm.build_jobs([], n_files)


INPUT_FILES = [
    "root://server:1094//store/2024/NMSSM-X_Par-MX-1000/et/f_0.root",
    "root://server:1094//store/2024/NMSSM-X_Par-MX-1000/et/f_1.root",
]


def _job(index=0, input_files=INPUT_FILES):
    return pm.Job(
        index=index,
        era="2024",
        campaign="2024_nano_v15",
        channel="et",
        input_files=list(input_files),
        output_files=[],
        samples=["NMSSM-X_Par-MX-1000"],
    )


def test_job_command(tmp_path):
    command = pm.job_command(
        _job(3),
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
        f"input_files=['{INPUT_FILES[0]}','{INPUT_FILES[1]}']",
        f"status_file={tmp_path}/status/2024_nano_v15__et__00003.json",
        "skip_existing=true",
        f"hydra.run.dir={tmp_path}/hydra",
        "hydra.output_subdir=null",
        "hydra/job_logging=stdout",
    ]


def test_job_command_with_profile(tmp_path):
    command = pm.job_command(
        _job(), [], tmp_path, skip_existing=False, profile=True
    )

    assert "skip_existing=false" in command
    assert command[-1] == (
        f"profile_file={tmp_path}/profiles/2024_nano_v15__et__00000.json"
    )


def test_job_command_is_parsed_by_hydra(tmp_path):
    """The overrides of a job are accepted by the produce config."""
    command = pm.job_command(_job(), [], tmp_path, True, True)
    overrides = [o for o in command[3:] if not o.startswith("hydra")]

    cfg = pm._compose(overrides)

    assert list(cfg.input_files) == INPUT_FILES
    assert cfg.skip_existing is True
    assert cfg.status_file == str(
        tmp_path / "status" / "2024_nano_v15__et__00000.json"
    )
    assert cfg.profile_file == str(
        tmp_path / "profiles" / "2024_nano_v15__et__00000.json"
    )


def test_job_command_logging_override_is_accepted_by_hydra():
    with pm.initialize_config_dir(
        str(PROJECT_DIR / "config"), version_base="1.3"
    ):
        cfg = pm.compose(
            "produce",
            overrides=["hydra/job_logging=stdout", "hydra.output_subdir=null"],
            return_hydra_config=True,
        )

    assert cfg.hydra.output_subdir is None
    assert list(cfg.hydra.job_logging.root.handlers) == ["console"]


def test_check_command_length(tmp_path):
    pm.check_command_length(["python", "x" * 1000])
    with pytest.raises(ValueError, match="--n-files"):
        pm.check_command_length(["python", "x" * pm.MAX_ARGUMENT_LENGTH])


@pytest.mark.parametrize(
    "override",
    [
        "input_files=[x.root]",
        "+input_files=[x.root]",
        "status_file=x",
        "skip_existing=true",
        "hydra.run.dir=x",
        "hydra/job_logging=none",
    ],
)
def test_check_overrides_rejects_per_job_overrides(override):
    with pytest.raises(ValueError, match="set per job"):
        pm.check_overrides(["inputs=y2b_h2tau", override])


def test_check_overrides_accepts_others():
    pm.check_overrides(
        ["mode.decay_mode=y2b_h2tau", "chunk_size=5000", "output_dir=x"]
    )


def _result(status="failed", returncode=1):
    return pm.JobResult("job", status, returncode, 1.0, None, None, [])


def test_failed_files_of_successful_job(tmp_path):
    assert pm.failed_files(_job(), _result("ok", 0), tmp_path) == []


def test_failed_files_from_status_file(tmp_path):
    job = _job()
    path = pm.status_file(tmp_path, job)
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "produced": [INPUT_FILES[0]],
                "skipped": [],
                "failed": [INPUT_FILES[1]],
            }
        )
    )

    assert pm.failed_files(job, _result(), tmp_path) == [INPUT_FILES[1]]


@pytest.mark.parametrize("content", [None, "not json", '{"produced": []}'])
def test_failed_files_without_usable_status_file(tmp_path, content):
    """After a crash, nothing is known about the files of the job."""
    job = _job()
    if content is not None:
        path = pm.status_file(tmp_path, job)
        path.parent.mkdir()
        path.write_text(content)

    assert pm.failed_files(job, _result(), tmp_path) == INPUT_FILES


# ---------------------------------------------------------------- execution


def _python(code):
    return [sys.executable, "-c", code]


def test_runner_statuses_and_logs(tmp_path):
    runner = pm.Runner(2, None, None, tmp_path)
    jobs = [
        (_job(0), _python("print('hello')")),
        (_job(1), _python("import sys; print('bad'); sys.exit(3)")),
    ]

    results = {r.name: r for r in runner.run(jobs)}

    a, b = (
        results["2024_nano_v15__et__00000"],
        results["2024_nano_v15__et__00001"],
    )
    assert (a.status, a.returncode) == ("ok", 0)
    assert (b.status, b.returncode) == ("failed", 3)
    assert "hello" in Path(a.log_file).read_text()
    assert (
        Path(b.log_file) == tmp_path / "logs" / "2024_nano_v15__et__00001.log"
    )
    # The log starts with the command
    assert Path(b.log_file).read_text().startswith(sys.executable)


def test_runner_environment_gpus_and_threads(tmp_path):
    runner = pm.Runner(2, ["2", "3"], 1, tmp_path)
    code = (
        "import os; print('GPU', os.environ['CUDA_VISIBLE_DEVICES'], "
        "os.environ['OMP_NUM_THREADS']); import time; time.sleep(0.5)"
    )
    jobs = [(_job(i), _python(code)) for i in range(4)]

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

    results = runner.run([(_job(i), sleep) for i in range(5)])

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
    skipped = runner.run_job(_job(1), _python("pass"))
    assert (skipped.status, skipped.log_file) == ("skipped", None)


def test_runner_invalid_number_of_jobs(tmp_path):
    with pytest.raises(ValueError, match="n_parallel"):
        pm.Runner(0, None, None, tmp_path)


# ---------------------------------------------------------------------- main


@pytest.fixture
def metadata(tmp_path):
    metadata_dir = tmp_path / "metadata"
    _write_metadata(metadata_dir, _selected_samples())
    return metadata_dir


def _overrides(metadata, tmp_path):
    return [
        f"metadata_dir={metadata}",
        f"output_dir={tmp_path / 'friends'}",
    ]


def test_main_requires_sample_selection(metadata, tmp_path):
    with pytest.raises(SystemExit):
        pm.main(["--dry-run", *_overrides(metadata, tmp_path)])


def test_main_invalid_n_files(metadata, tmp_path):
    with pytest.raises(SystemExit):
        pm.main(
            ["--sample", "DY", "--n-files", "0"]
            + _overrides(metadata, tmp_path)
        )


def test_main_rejects_per_job_override(metadata, tmp_path, caplog):
    args = [
        "--sample",
        "DY",
        "input_files=[x]",
        *_overrides(metadata, tmp_path),
    ]

    assert pm.main(args) == 2
    assert "set per job" in caplog.text


def test_main_unknown_campaign(metadata, tmp_path, caplog):
    args = ["--sample", "DY", "--campaign", "2024"]

    assert pm.main([*args, *_overrides(metadata, tmp_path)]) == 2
    assert "Unknown campaigns" in caplog.text


def test_main_dry_run(metadata, tmp_path, capsys, caplog):
    samples_file = tmp_path / "samples.txt"
    samples_file.write_text("TT\nZZ\n")

    with caplog.at_level(logging.INFO):
        code = pm.main(
            [
                "--sample",
                "D.*",
                "--samples-file",
                str(samples_file),
                "--campaign",
                "2024_nano_v15",
                "--n-files",
                "2",
                "--dry-run",
                *_overrides(metadata, tmp_path),
            ]
        )

    out = capsys.readouterr().out
    assert code == 0
    assert out.splitlines()[:2] == [
        "2024_nano_v15 et: 3 jobs, 6 files",
        "2024_nano_v15 mt: 1 jobs, 1 files",
    ]
    assert "input_files=['root://server//store/ntuples/2024/DY/et" in out
    assert "WJets" not in out
    assert "1 samples of the sample files have no metadata" in caplog.text
    assert not (tmp_path / "logs").exists()


def test_main_too_long_command(metadata, tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(pm, "MAX_ARGUMENT_LENGTH", 100)

    args = ["--sample", "TT", "--dry-run", *_overrides(metadata, tmp_path)]

    assert pm.main(args) == 2
    assert "--n-files" in caplog.text


def _fake_job_command(job, overrides, log_dir, skip_existing, profile):
    """Write the friend files of the job, except for the files of sample TT.
    Report the result like produce, or crash for 2025 and channel tt."""
    status = pm.status_file(log_dir, job)
    code = (
        "import json, pathlib, sys\n"
        f"if {job.channel!r} == 'tt': sys.exit(137)\n"
        "produced, failed = [], []\n"
        f"for i, o in zip({job.input_files!r}, {job.output_files!r}):\n"
        "    if '/TT/' in i:\n"
        "        failed.append(i)\n"
        "        continue\n"
        "    p = pathlib.Path(o); p.parent.mkdir(parents=True, "
        "exist_ok=True); p.touch()\n"
        "    produced.append(i)\n"
        f"s = pathlib.Path({str(status)!r}); "
        "s.parent.mkdir(parents=True, exist_ok=True)\n"
        "s.write_text(json.dumps({'produced': produced, 'skipped': [], "
        "'failed': failed}))\n"
        "sys.exit(1 if failed else 0)\n"
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
        "--n-files",
        "3",
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
        # 2024 et: DY_0, DY_1, TT_0 and TT_1, TT_2, TT_3
        "2024_nano_v15__et__00000": "failed",
        "2024_nano_v15__et__00001": "failed",
        "2024_nano_v15__mt__00000": "ok",
        "2025_nano_v15__et__00000": "ok",
        "2025_nano_v15__tt__00000": "failed",
    }
    assert (summary["n_jobs"], summary["n_failed"]) == (5, 3)
    assert (summary["n_files"], summary["n_failed_files"]) == (9, 5)
    # Files of the first job that were produced are not failed
    assert (tmp_path / "friends/2024/DY/et/DY_0.root").exists()
    assert (tmp_path / "friends/2024/DY/mt/DY_0.root").exists()
    failed_list = (log_dir / "failed_files.txt").read_text().splitlines()
    assert sorted(f.rsplit("/", 1)[-1] for f in failed_list) == [
        "TT_0.root",
        "TT_0.root",  # crash of the 2025 tt job
        "TT_1.root",
        "TT_2.root",
        "TT_3.root",
    ]
    job = next(
        j for j in summary["jobs"] if j["name"] == "2024_nano_v15__et__00000"
    )
    assert len(job["input_files"]) == 3
    assert [f.rsplit("/", 1)[-1] for f in job["failed_files"]] == ["TT_0.root"]
    assert "3 jobs failed, 5 files without friend" in caplog.text

    # Repeating with --skip-existing forms new jobs of the remaining files
    caplog.clear()
    log_dir_2 = tmp_path / "run2"
    args[args.index(str(log_dir))] = str(log_dir_2)
    with caplog.at_level(logging.INFO):
        assert pm.main(["--skip-existing", *args]) == 1

    summary = json.loads((log_dir_2 / "summary.json").read_text())
    assert summary["n_files"] == 5
    assert sorted(f for j in summary["jobs"] for f in j["input_files"]) == (
        sorted(failed_list)
    )
    assert "Skipping 4 files whose friends exist" in caplog.text


def test_main_all_jobs_succeed(metadata, tmp_path, monkeypatch):
    monkeypatch.setattr(pm, "job_command", _fake_job_command)
    log_dir = tmp_path / "run"
    args = [
        "--sample",
        "DY",
        "--gpus",
        "none",
        "--log-dir",
        str(log_dir),
        *_overrides(metadata, tmp_path),
    ]

    assert pm.main(args) == 0
    assert not (log_dir / "failed_files.txt").exists()

    # Everything exists now: no job is started
    assert pm.main(["--skip-existing", *args]) == 0


def test_main_nothing_to_do(metadata, tmp_path, caplog):
    assert pm.main(["--sample", "ZZ", *_overrides(metadata, tmp_path)]) == 0
    assert "No jobs to run" in caplog.text


# ------------------------------------------------------- upload to XRootD


def test_main_nn_friend_tag(metadata, tmp_path, monkeypatch, capsys, caplog):
    """With nn_friend_tag, jobs write to the XRootD server, which is also
    checked for existing friends: only the selected directories."""
    listed = []

    def listing(directory):
        listed.append(directory)
        return {"DY_0.root", "DY_1.root"} if "/2024/" in directory else set()

    monkeypatch.setattr(pm, "list_directory", listing)
    args = [
        "--sample",
        "DY",
        "--channel",
        "et",
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
    friend = f"{remote}/pnn-test-y2b_h2tau"
    assert listed == [f"{friend}/2024/DY/et", f"{friend}/2025/DY/et"]
    # The 2024 files are complete, the 2025 file is run with the tag
    assert "Skipping 2 files" in caplog.text
    assert out.splitlines()[0] == "2025_nano_v15 et: 1 jobs, 1 files"
    assert "nn_friend_tag=pnn-test" in out
    assert f"friends in {friend}" in caplog.text


def test_main_invalid_nn_friend_tag(metadata, tmp_path, caplog):
    args = [
        "--sample",
        "DY",
        *_overrides(metadata, tmp_path),
        "nn_friend_tag=a/b",
    ]

    assert pm.main(args) == 2
    assert "Invalid nn_friend_tag" in caplog.text


def test_list_directory_remote(monkeypatch, fake_fs_factory):
    fs = fake_fs_factory({"/store/friend/2024/DY/et": ["a.root", "b.root"]})
    monkeypatch.setattr(paths, "FileSystem", lambda server: fs)

    assert paths.list_directory(
        "root://server:1094//store/friend/2024/DY/et"
    ) == {"a.root", "b.root"}
    assert paths.list_directory("root://server:1094//store/missing") == set()


def test_list_directory_local(tmp_path):
    (tmp_path / "a.root").touch()

    assert paths.list_directory(tmp_path) == {"a.root"}
    assert paths.list_directory(tmp_path / "missing") == set()


def test_produce_many_does_not_need_torch():
    """Planning jobs does not load torch and the model code."""
    code = (
        "import sys; import xyh_ml.cli.friends.produce_many; "
        "sys.exit('torch' in sys.modules)"
    )
    process = subprocess.run([sys.executable, "-c", code], check=False)
    assert process.returncode == 0
