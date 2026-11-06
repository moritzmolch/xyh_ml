"""Tests for the HTCondor submission, without a real HTCondor."""

import io
import json
import logging
import os
import stat
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

from xyh_ml.cli.friends import produce_many as pm
from xyh_ml.friend_production import condor
from xyh_ml.friend_production.dataclasses import Sample
from xyh_ml.friend_production.metadata import write_sample_metadata

URL = "root://server:1094//store/2024/DY=1/et/f_{}.root"
OVERRIDE = "artifacts.training_dir=/x/fold${mode.fold}of${mode.n_folds}"
THREAD_VARIABLES = ["OMP_NUM_THREADS", "MKL_NUM_THREADS"]


def _write(tmp_path, commands=None, options=None):
    commands = commands or {"job_a": ["python", "-m", "x"]}
    return condor.write_submission(
        tmp_path,
        tmp_path / "repo",
        commands,
        tmp_path / "proxy",
        THREAD_VARIABLES,
        options or condor.CondorOptions(),
    )


# ------------------------------------------------------------------ submit file


def test_submit_description(tmp_path):
    options = condor.CondorOptions(
        memory_mb=3000,
        disk_gb=7,
        walltime_hours=1.5,
        accounting_group="cms.jet",
        max_materialize=50,
        threads_per_job=3,
    )

    text = _write(tmp_path, options=options).read_text()

    for line in [
        f"container_image = {condor.CONTAINER_IMAGE}",
        f"executable = {tmp_path}/job.sh",
        "requirements = (TARGET.ProvidesETPResources =?= True)",
        "accounting_group = cms.jet",
        "request_cpus = 3",
        "request_memory = 3000 + 2000 * NumJobStarts",
        "request_disk = 7G",
        "+RequestWalltime = 5400",
        f"x509userproxy = {tmp_path}/proxy",
        'transfer_output_files = ""',
        "max_retries = 2",
        "periodic_release = (HoldReasonCode == 34) && (NumJobStarts < 3)",
        "max_materialize = 50",
        f"output = {tmp_path}/logs/$(job).out",
        f"queue job from {tmp_path}/jobs.txt",
    ]:
        assert line in text.splitlines()
    # ETP-local on CPU slots: the requirement excludes remote sites
    assert "RemoteJob" not in text
    assert "request_gpus" not in text


def test_write_submission_files(tmp_path):
    command = ["python", "-m", "x", OVERRIDE, f"input_files=['{URL}']"]

    _write(tmp_path, {"job_a": command, "job_b": ["b"]})

    assert (tmp_path / "jobs.txt").read_text() == "job_a\njob_b\n"
    assert (tmp_path / "commands" / "job_a.args").read_text().splitlines() == (
        command
    )
    assert (tmp_path / "logs").is_dir()
    assert (tmp_path / "job.sh").stat().st_mode & stat.S_IXUSR


def test_write_submission_rejects_line_break(tmp_path):
    with pytest.raises(condor.CondorError, match="line break"):
        _write(tmp_path, {"job_a": ["python", "a\nb"]})


# -------------------------------------------------------------------- job script


def test_job_script_runs_command_unchanged(tmp_path):
    """Arguments with special characters arrive unchanged, in an environment
    with the repository and without the home directory of the user."""
    repo = tmp_path / "repo"
    repo.mkdir()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    fake_python = tmp_path / "fake_python"
    fake_python.write_text(
        "#!/bin/bash\n"
        'for a in "$@"; do echo "arg:$a"; done\n'
        'echo "cwd:$PWD"\n'
        'echo "home:$HOME"\n'
        'echo "path:$PYTHONPATH"\n'
        'echo "omp:$OMP_NUM_THREADS mkl:$MKL_NUM_THREADS"\n'
    )
    fake_python.chmod(0o755)
    url = URL.format(0)
    arguments = [OVERRIDE, f"input_files=['{url}','{url}']", "a b"]
    _write(
        tmp_path,
        {"job_a": [str(fake_python), *arguments]},
        condor.CondorOptions(threads_per_job=3),
    )

    result = subprocess.run(
        ["bash", str(tmp_path / "job.sh"), "job_a"],
        capture_output=True,
        text=True,
        check=True,
        env={
            "PATH": os.environ["PATH"],
            "_CONDOR_SCRATCH_DIR": str(scratch),
        },
    )

    lines = result.stdout.splitlines()
    assert [ln[4:] for ln in lines if ln.startswith("arg:")] == [
        *arguments,
        f"staging_dir={scratch}/staging",
    ]
    assert f"cwd:{repo}" in lines
    assert f"home:{scratch}" in lines
    assert f"path:{repo}/src" in lines
    assert "omp:3 mkl:3" in lines
    assert "job:     job_a" in lines


def test_job_script_passes_exit_code(tmp_path):
    (tmp_path / "repo").mkdir()
    _write(tmp_path, {"job_a": ["bash", "-c", "exit 7"]})

    result = subprocess.run(
        ["bash", str(tmp_path / "job.sh"), "job_a"],
        capture_output=True,
        check=False,
        env={"PATH": os.environ["PATH"], "_CONDOR_SCRATCH_DIR": str(tmp_path)},
    )

    assert result.returncode == 7


# ---------------------------------------------------------------- remote jobs

ENV_URL = "root://server//store/sandboxes/env-cpu-abc.sh"
BUNDLE_URL = "root://server//store/sandboxes/bundle-def.tar.gz"
SANDBOX = condor.RemoteSandbox(ENV_URL, BUNDLE_URL)


def _write_remote(tmp_path, options=None, commands=None):
    log_dir = tmp_path / "run"
    log_dir.mkdir()
    return condor.write_submission(
        log_dir,
        tmp_path / "not_used",
        commands or {"job_a": ["env/bin/python", "-m", "x", OVERRIDE, "a b"]},
        tmp_path / "proxy",
        THREAD_VARIABLES,
        options or condor.CondorOptions(threads_per_job=3),
        sandbox=SANDBOX,
    )


def test_remote_submit_description(tmp_path):
    lines = _write_remote(tmp_path).read_text().splitlines()
    log_dir = tmp_path / "run"

    for line in [
        "+RemoteJob = True",
        f"transfer_input_files = {log_dir}/commands/$(job).args",
        "transfer_output_files = status.json",
        f'transfer_output_remaps = "status.json = {log_dir}/status/$(job).json"',
        f"x509userproxy = {tmp_path}/proxy",
        f"executable = {log_dir}/job.sh",
        "max_retries = 2",
    ]:
        assert line in lines
    # Never together with +RemoteJob, as such a job would never match
    assert not any("ProvidesETPResources" in line for line in lines)
    assert not any(line.startswith("requirements") for line in lines)
    assert (log_dir / "status").is_dir()


def test_remote_extra_requirements(tmp_path):
    options = condor.CondorOptions(requirements='TARGET.CloudSite =?= "x"')

    lines = _write_remote(tmp_path, options).read_text().splitlines()

    assert 'requirements = (TARGET.CloudSite =?= "x")' in lines
    assert "+RemoteJob = True" in lines


def test_local_extra_requirements(tmp_path):
    options = condor.CondorOptions(requirements="TARGET.Cpus > 2")

    lines = _write(tmp_path, options=options).read_text().splitlines()

    assert (
        "requirements = (TARGET.ProvidesETPResources =?= True) && "
        "(TARGET.Cpus > 2)"
    ) in lines


def test_local_submit_description_unchanged(tmp_path):
    """Local jobs keep the exact settings that were tested with real jobs."""
    text = _write(tmp_path).read_text()

    assert "RemoteJob" not in text
    assert "transfer_input_files" not in text
    assert "transfer_output_remaps" not in text
    assert 'transfer_output_files = ""' in text.splitlines()
    assert (
        "requirements = (TARGET.ProvidesETPResources =?= True)"
    ) in text.splitlines()
    assert not (tmp_path / "status").exists()


class FakeWorker:
    """A scratch directory of a worker node with fake tools.

    ``xrdcp`` copies from a directory of stored files, the pack creates the
    environment with a fake Python, and the bundle holds a marker file.
    """

    def __init__(self, tmp_path, log_dir):
        self.log_dir = log_dir
        self.scratch = tmp_path / "scratch"
        self.scratch.mkdir()
        self.tools = tmp_path / "tools"
        self.tools.mkdir()
        self.store = tmp_path / "store"
        self.store.mkdir()
        self._tool(
            "xrdcp",
            f"""#!/bin/bash
echo "xrdcp $*" >> {tmp_path}/xrdcp.log
name="${{@: -2:1}}"
cp "{self.store}/${{name##*/}}" "${{@: -1}}"
""",
        )
        self._tool("sleep", "#!/bin/bash\n")
        (self.store / "env-cpu-abc.sh").write_text(
            """#!/bin/bash
mkdir -p env/bin
cat > env/bin/python <<'EOF'
#!/bin/bash
for a in "$@"; do echo "arg:$a"; done
echo "cwd:$PWD"
echo "home:$HOME"
echo "tmp:$TMPDIR"
echo "path:$PYTHONPATH"
echo "activated:$ACTIVATED"
echo "omp:$OMP_NUM_THREADS"
ls src/marker
echo '{"produced": ["x"], "skipped": [], "failed": []}' > status.json
exit ${EXIT_CODE:-0}
EOF
chmod +x env/bin/python
echo 'export ACTIVATED=yes' > activate.sh
"""
        )
        source = tmp_path / "bundle_source"
        (source / "src").mkdir(parents=True)
        (source / "src" / "marker").write_text("")
        subprocess.run(
            [
                "tar",
                "-czf",
                str(self.store / "bundle-def.tar.gz"),
                "-C",
                str(source),
                "src",
            ],
            check=True,
        )
        # HTCondor transfers the executable and the command file
        for name in ["job.sh", "commands/job_a.args"]:
            source_file = log_dir / name
            (self.scratch / source_file.name).write_text(
                source_file.read_text()
            )

    def _tool(self, name, content):
        path = self.tools / name
        path.write_text(content)
        path.chmod(0o755)

    def run(self, **env):
        return subprocess.run(
            ["bash", str(self.scratch / "job.sh"), "job_a"],
            capture_output=True,
            text=True,
            check=False,
            cwd=self.scratch,
            env={
                "PATH": f"{self.tools}:{os.environ['PATH']}",
                "_CONDOR_SCRATCH_DIR": str(self.scratch),
                **env,
            },
        )


def test_remote_job_script_runs_command(tmp_path):
    _write_remote(tmp_path)
    worker = FakeWorker(tmp_path, tmp_path / "run")

    result = worker.run()

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    scratch = worker.scratch
    assert [ln[4:] for ln in lines if ln.startswith("arg:")] == [
        "-m",
        "x",
        OVERRIDE,
        "a b",
        f"staging_dir={scratch}/staging",
    ]
    assert f"cwd:{scratch}" in lines
    assert f"home:{scratch}" in lines
    assert f"tmp:{scratch}/tmp" in lines
    assert f"path:{scratch}/src" in lines
    assert "activated:yes" in lines
    assert "omp:3" in lines
    # The fetched files are removed, to save disk
    assert not (scratch / "environment.sh").exists()
    assert not (scratch / "bundle.tar.gz").exists()
    assert json.loads((scratch / "status.json").read_text())["produced"] == [
        "x"
    ]


def test_remote_job_script_passes_exit_code(tmp_path):
    _write_remote(tmp_path)
    worker = FakeWorker(tmp_path, tmp_path / "run")

    assert worker.run(EXIT_CODE="7").returncode == 7


@pytest.mark.parametrize("name", ["env-cpu-abc.sh", "bundle-def.tar.gz"])
def test_remote_job_script_failed_fetch_leaves_status_file(tmp_path, name):
    """The status file exists, so that the job fails without going on hold."""
    _write_remote(tmp_path)
    worker = FakeWorker(tmp_path, tmp_path / "run")
    (worker.store / name).unlink()

    result = worker.run()

    assert result.returncode != 0
    assert json.loads((worker.scratch / "status.json").read_text()) == {}
    # Every attempt is made
    log = (tmp_path / "xrdcp.log").read_text()
    assert log.count(name) == condor.FETCH_ATTEMPTS


def test_remote_job_script_retries_fetch(tmp_path):
    _write_remote(tmp_path)
    worker = FakeWorker(tmp_path, tmp_path / "run")
    (worker.tools / "xrdcp").write_text(
        f"""#!/bin/bash
echo "xrdcp $*" >> {tmp_path}/xrdcp.log
if [ ! -e {tmp_path}/failed_once ]; then
    touch {tmp_path}/failed_once
    exit 1
fi
name="${{@: -2:1}}"
cp "{worker.store}/${{name##*/}}" "${{@: -1}}"
"""
    )

    result = worker.run()

    assert result.returncode == 0, result.stderr
    assert "Attempt 1 to fetch" in result.stderr


def test_remote_job_script_failed_unpack(tmp_path):
    _write_remote(tmp_path)
    worker = FakeWorker(tmp_path, tmp_path / "run")
    (worker.store / "env-cpu-abc.sh").write_text("exit 3\n")

    result = worker.run()

    assert result.returncode == 3
    assert json.loads((worker.scratch / "status.json").read_text()) == {}


# ------------------------------------------------------------- condor commands


def _fake_run(monkeypatch, stdout="", returncode=0, stderr=""):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        output = stdout(command) if callable(stdout) else stdout
        return SimpleNamespace(
            stdout=output, stderr=stderr, returncode=returncode
        )

    monkeypatch.setattr(condor.subprocess, "run", run)
    return calls


def test_submit_returns_cluster(monkeypatch, tmp_path):
    calls = _fake_run(
        monkeypatch,
        "Submitting job(s)...\n3 job(s) submitted to cluster 1234.\n",
    )

    assert condor.submit(tmp_path / "submit.jdl") == 1234
    assert calls == [["condor_submit", str(tmp_path / "submit.jdl")]]


def test_submit_failure(monkeypatch, tmp_path):
    _fake_run(monkeypatch, returncode=1, stderr="no schedd")

    with pytest.raises(condor.CondorError, match="no schedd"):
        condor.submit(tmp_path / "submit.jdl")


def test_submit_without_condor(monkeypatch, tmp_path):
    def run(command, **kwargs):
        raise FileNotFoundError("condor_submit")

    monkeypatch.setattr(condor.subprocess, "run", run)

    with pytest.raises(condor.CondorError, match="condor_submit failed"):
        condor.submit(tmp_path / "submit.jdl")


def test_job_states(monkeypatch):
    _fake_run(
        monkeypatch,
        lambda c: "" if "-factory" in c else "0 2\n3 5\n",
    )

    assert condor.job_states(12) == ({0: 2, 3: 5}, True)


def test_job_states_only_factory_left(monkeypatch):
    _fake_run(monkeypatch, lambda c: "12\n" if "-factory" in c else "")

    assert condor.job_states(12) == ({}, True)


def test_job_states_finished(monkeypatch):
    _fake_run(monkeypatch)

    assert condor.job_states(12) == ({}, False)


@pytest.mark.parametrize(
    "stdout, returncode, match",
    [
        ("3600\n", 0, "valid for only 1.0 h"),
        ("0\n", 0, "valid for only 0.0 h"),
        ("", 1, "No valid proxy"),
        ("not a number\n", 0, "No valid proxy"),
    ],
)
def test_check_proxy_invalid(monkeypatch, stdout, returncode, match):
    _fake_run(monkeypatch, stdout, returncode)

    with pytest.raises(condor.CondorError, match=match) as error:
        condor.check_proxy()

    assert "voms-proxy-init" in str(error.value)


def test_check_proxy_valid(monkeypatch, tmp_path):
    monkeypatch.setenv("X509_USER_PROXY", str(tmp_path / "proxy"))
    calls = _fake_run(monkeypatch, "100000\n")

    assert condor.check_proxy() == tmp_path / "proxy"
    assert str(tmp_path / "proxy") in calls[0]


def test_proxy_path_default(monkeypatch):
    monkeypatch.delenv("X509_USER_PROXY", raising=False)

    assert str(condor.proxy_path()) == f"/tmp/x509up_u{os.getuid()}"


def test_check_shared_paths():
    condor.check_shared_paths({"a": "/work/x", "b": "/ceph/y", "c": "/home/z"})

    with pytest.raises(condor.CondorError, match=r"metadata \(/tmp/x\)"):
        condor.check_shared_paths({"a": "/work/x", "metadata": "/tmp/x"})
    # Only whole directory names count
    with pytest.raises(condor.CondorError, match="/workspace"):
        condor.check_shared_paths({"a": "/workspace/x"})


# ------------------------------------------------------------------------- main


@pytest.fixture
def metadata(tmp_path):
    metadata_dir = tmp_path / "metadata"
    records = [
        Sample(
            era="2024",
            sample="DY",
            channel="et",
            main_files=[
                f"root://server//store/2024/DY/et/f_{i}.root" for i in range(3)
            ],
            friend_files={},
            shifts=[],
        ),
    ]
    write_sample_metadata(metadata_dir, "2024", "DY", records)
    return metadata_dir


@pytest.fixture
def fake_condor(monkeypatch):
    """No checks of the file systems and the proxy, and record submissions."""
    submitted = []

    def fake_submit(submit_file):
        submitted.append(submit_file)
        return 77

    monkeypatch.setattr(pm, "check_shared_paths", lambda paths: None)
    monkeypatch.setattr(pm, "check_proxy", lambda: "/tmp/proxy")
    monkeypatch.setattr(pm, "submit", fake_submit)
    monkeypatch.setattr(pm.socket, "gethostname", lambda: "portal")
    return submitted


def _args(metadata, tmp_path, *options):
    return [
        "--sample",
        "DY",
        "--n-files",
        "2",
        "--log-dir",
        str(tmp_path / "run"),
        *options,
        f"metadata_dir={metadata}",
        f"output_dir={tmp_path / 'friends'}",
    ]


def test_main_condor_submits(metadata, tmp_path, fake_condor, caplog):
    log_dir = tmp_path / "run"

    with caplog.at_level(logging.INFO):
        code = pm.main(_args(metadata, tmp_path, "--condor"))

    assert code == 0
    assert fake_condor == [log_dir / "submit.jdl"]
    names = (log_dir / "jobs.txt").read_text().splitlines()
    assert names == ["2024_nano_v15__et__00000", "2024_nano_v15__et__00001"]
    assert "max_materialize = 100" in (log_dir / "submit.jdl").read_text()
    manifest = json.loads((log_dir / "jobs.json").read_text())
    assert manifest["cluster"] == 77
    assert manifest["submit_host"] == "portal"
    assert len(manifest["jobs"][0]["input_files"]) == 2
    arguments = (log_dir / "commands" / f"{names[1]}.args").read_text()
    assert "status_file=" in arguments
    assert "f_2.root" in arguments
    assert "--collect" in caplog.text
    # Nothing has run
    assert not (tmp_path / "friends").exists()


def test_main_condor_options_reach_the_submit_file(
    metadata, tmp_path, fake_condor
):
    options = [
        "--condor",
        "-j",
        "7",
        "--threads-per-job",
        "1",
        "--condor-memory",
        "2500",
        "--condor-disk",
        "9",
        "--condor-walltime",
        "2",
        "--accounting-group",
        "cms.production",
    ]

    assert pm.main(_args(metadata, tmp_path, *options)) == 0

    lines = (tmp_path / "run" / "submit.jdl").read_text().splitlines()
    for line in [
        "max_materialize = 7",
        "request_cpus = 1",
        "request_memory = 2500 + 2000 * NumJobStarts",
        "request_disk = 9G",
        "+RequestWalltime = 7200",
        "accounting_group = cms.production",
    ]:
        assert line in lines


def test_main_condor_dry_run_writes_nothing(
    metadata, tmp_path, fake_condor, capsys
):
    args = _args(metadata, tmp_path, "--condor", "--dry-run")

    assert pm.main(args) == 0

    assert fake_condor == []
    assert not (tmp_path / "run").exists()
    assert "2024_nano_v15 et: 2 jobs, 3 files" in capsys.readouterr().out


def test_main_condor_rejects_gpus(metadata, tmp_path):
    with pytest.raises(SystemExit):
        pm.main(_args(metadata, tmp_path, "--condor", "--gpus", "0"))


def test_main_condor_rejects_staging_dir_override(metadata, tmp_path, caplog):
    args = _args(metadata, tmp_path, "--condor")

    assert pm.main([*args, "staging_dir=x"]) == 2
    assert "set per job" in caplog.text


def test_main_local_accepts_staging_dir_override():
    pm.check_overrides(["staging_dir=x"])


def test_main_condor_proxy_problem(
    metadata, tmp_path, fake_condor, monkeypatch, caplog
):
    def no_proxy():
        raise condor.CondorError("No valid proxy")

    monkeypatch.setattr(pm, "check_proxy", no_proxy)

    assert pm.main(_args(metadata, tmp_path, "--condor")) == 2
    assert fake_condor == []
    assert "No valid proxy" in caplog.text


def test_main_condor_checks_paths(metadata, tmp_path, monkeypatch, caplog):
    """Paths of pytest are not on a shared file system or passed to it."""
    checked = {}

    def check(paths):
        checked.update(paths)
        raise condor.CondorError("not shared")

    monkeypatch.setattr(pm, "check_shared_paths", check)

    assert pm.main(_args(metadata, tmp_path, "--condor")) == 2
    assert set(checked) == {
        "Python interpreter",
        "repository",
        "log directory",
        "metadata",
        "training artifacts",
        "output directory",
    }


def test_main_local_does_not_use_condor(
    metadata, tmp_path, fake_condor, monkeypatch
):
    monkeypatch.setattr(
        pm,
        "job_command",
        lambda *args: ["true"],
    )

    assert pm.main(_args(metadata, tmp_path, "--gpus", "none")) == 0

    assert fake_condor == []
    assert not (tmp_path / "run" / "submit.jdl").exists()
    assert not (tmp_path / "run" / "jobs.json").exists()


# ---------------------------------------------------------------------- collect


def _submit(metadata, tmp_path):
    assert pm.main(_args(metadata, tmp_path, "--condor")) == 0
    log_dir = tmp_path / "run"
    manifest = json.loads((log_dir / "jobs.json").read_text())
    return log_dir, manifest["jobs"]


def _report(log_dir, job, produced=(), skipped=(), failed=()):
    path = log_dir / "status" / f"{pm.Job(**job).name}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "produced": list(produced),
                "skipped": list(skipped),
                "failed": list(failed),
            }
        )
    )


def _states(monkeypatch, states=None, active=False):
    monkeypatch.setattr(pm, "job_states", lambda c: (states or {}, active))


def _summary(log_dir):
    return json.loads((log_dir / "summary.json").read_text())


def test_collect_all_ok(metadata, tmp_path, fake_condor, monkeypatch, caplog):
    log_dir, jobs = _submit(metadata, tmp_path)
    _report(log_dir, jobs[0], produced=jobs[0]["input_files"])
    _report(log_dir, jobs[1], skipped=jobs[1]["input_files"])
    _states(monkeypatch)

    with caplog.at_level(logging.INFO):
        assert pm.main(["--collect", str(log_dir)]) == 0

    summary = _summary(log_dir)
    assert (summary["n_jobs"], summary["n_failed"]) == (2, 0)
    assert summary["cluster"] == 77
    assert not (log_dir / "failed_files.txt").exists()
    assert "2 jobs ok, 0 failed, 0 unfinished" in caplog.text


def test_collect_partly_failed(metadata, tmp_path, fake_condor, monkeypatch):
    log_dir, jobs = _submit(metadata, tmp_path)
    first, second = jobs[0]["input_files"]
    _report(log_dir, jobs[0], produced=[first], failed=[second])
    _report(log_dir, jobs[1], produced=jobs[1]["input_files"])
    _states(monkeypatch)

    assert pm.main(["--collect", str(log_dir)]) == 1

    assert (log_dir / "failed_files.txt").read_text() == f"{second}\n"
    summary = _summary(log_dir)
    assert summary["n_failed_files"] == 1
    assert summary["jobs"][0]["failed_files"] == [second]


def test_collect_crashed_job_lists_all_files(
    metadata, tmp_path, fake_condor, monkeypatch
):
    """A job without status file after the end of the cluster has crashed."""
    log_dir, jobs = _submit(metadata, tmp_path)
    _report(log_dir, jobs[1], produced=jobs[1]["input_files"])
    _states(monkeypatch)

    assert pm.main(["--collect", str(log_dir)]) == 1

    assert (log_dir / "failed_files.txt").read_text().splitlines() == (
        jobs[0]["input_files"]
    )


def test_collect_status_without_all_files_is_failed(
    metadata, tmp_path, fake_condor, monkeypatch
):
    log_dir, jobs = _submit(metadata, tmp_path)
    _report(log_dir, jobs[0], produced=jobs[0]["input_files"][:1])
    _report(log_dir, jobs[1], produced=jobs[1]["input_files"])
    _states(monkeypatch)

    assert pm.main(["--collect", str(log_dir)]) == 1
    assert _summary(log_dir)["n_failed"] == 1


def test_collect_queued_and_not_materialized_are_unfinished(
    metadata, tmp_path, fake_condor, monkeypatch, caplog
):
    """A queued job can have the status of an earlier attempt, and a job that
    is not materialized yet is not listed by condor_q."""
    log_dir, jobs = _submit(metadata, tmp_path)
    _report(log_dir, jobs[0], failed=jobs[0]["input_files"])
    _states(monkeypatch, {0: 2}, active=True)

    with caplog.at_level(logging.INFO):
        assert pm.main(["--collect", str(log_dir)]) == 1

    summary = _summary(log_dir)
    assert [j["status"] for j in summary["jobs"]] == [
        "unfinished",
        "unfinished",
    ]
    assert summary["n_failed"] == 0
    assert not (log_dir / "failed_files.txt").exists()
    assert "0 failed, 2 unfinished" in caplog.text
    assert "preliminary" in caplog.text


def test_collect_reports_held_jobs(
    metadata, tmp_path, fake_condor, monkeypatch, caplog
):
    log_dir, jobs = _submit(metadata, tmp_path)
    _report(log_dir, jobs[1], produced=jobs[1]["input_files"])
    _states(monkeypatch, {0: condor.JOB_STATUS_HELD}, active=True)

    assert pm.main(["--collect", str(log_dir)]) == 1

    assert "1 jobs are held" in caplog.text
    assert "2024_nano_v15__et__00000" in caplog.text


def test_collect_later_success_replaces_failed_list(
    metadata, tmp_path, fake_condor, monkeypatch
):
    log_dir, jobs = _submit(metadata, tmp_path)
    _states(monkeypatch)
    assert pm.main(["--collect", str(log_dir)]) == 1
    assert (log_dir / "failed_files.txt").exists()

    _report(log_dir, jobs[0], produced=jobs[0]["input_files"])
    _report(log_dir, jobs[1], produced=jobs[1]["input_files"])

    assert pm.main(["--collect", str(log_dir)]) == 0
    assert not (log_dir / "failed_files.txt").exists()


def test_collect_on_other_machine(
    metadata, tmp_path, fake_condor, monkeypatch, caplog
):
    log_dir, _ = _submit(metadata, tmp_path)
    monkeypatch.setattr(pm.socket, "gethostname", lambda: "other")

    assert pm.main(["--collect", str(log_dir)]) == 2
    assert "submitted on portal" in caplog.text


def test_collect_without_submission(tmp_path, caplog):
    assert pm.main(["--collect", str(tmp_path)]) == 2
    assert "jobs.json" in caplog.text


def test_collect_takes_no_other_options(tmp_path):
    with pytest.raises(SystemExit):
        pm.main(["--collect", str(tmp_path), "--sample", "DY"])


# ------------------------------------------------------------- --condor-remote


ENV_PACK_URL = "root://server//store/sandboxes/env-cpu-abc.sh"
TRAINING_DIR = "fold${mode.fold}of${mode.n_folds}/${mode.channel}"


@pytest.fixture
def training_dir(tmp_path):
    root = tmp_path / "training"
    for fold in range(5):
        for channel in ["et", "mt", "tt"]:
            directory = root / f"fold{fold}of5" / channel
            directory.mkdir(parents=True)
            for name in ["transformations.pkl", "best_model.ckpt"]:
                (directory / name).write_text(f"{fold} {channel} {name}")
    return root


@pytest.fixture
def storage(monkeypatch):
    """Grid storage for the bundle, and a fake environment pack."""
    files = {}
    sandbox = sys.modules["xyh_ml.friend_production.sandbox"]
    monkeypatch.setattr(sandbox, "xrd_exists", lambda url: url in files)
    monkeypatch.setattr(
        sandbox,
        "xrd_upload",
        lambda local_file, url: files.update({url: local_file.read_bytes()}),
    )
    monkeypatch.setattr(
        pm, "environment_pack", lambda url, project: ENV_PACK_URL
    )
    return files


def _remote_args(metadata, tmp_path, training_dir, *options):
    return [
        "--sample",
        "DY",
        "--n-files",
        "2",
        "--log-dir",
        str(tmp_path / "run"),
        "--condor-remote",
        *options,
        f"metadata_dir={metadata}",
        f"artifacts.training_dir={training_dir}/{TRAINING_DIR}",
        "nn_friend_tag=tag",
    ]


def test_remote_job_command_is_parsed_by_hydra():
    job = pm.Job(
        index=0,
        era="2024",
        campaign="2024_nano_v15",
        channel="tt",
        input_files=["root://server//store/f.root"],
        output_files=["x"],
        samples=["DY"],
    )

    command = pm.remote_job_command(
        job,
        [
            "metadata_dir=/work/x",
            "artifacts.training_dir=/work/y",
            "nn_friend_tag=tag",
        ],
        True,
    )

    assert command[0] == "env/bin/python"
    assert "status_file=status.json" in command
    assert "/work" not in " ".join(command)
    cfg = pm._compose(command[3:] + ["mode.channel=tt", "mode.fold=2"])
    assert cfg.metadata_dir == "metadata"
    assert cfg.artifacts.training_dir == "models/fold2of5/tt"
    assert cfg.nn_friend_tag == "tag"
    assert cfg.status_file == "status.json"


def test_main_remote_submits(
    metadata, tmp_path, training_dir, fake_condor, storage, monkeypatch
):
    checked = []
    monkeypatch.setattr(pm, "check_shared_paths", checked.append)
    log_dir = tmp_path / "run"

    code = pm.main(_remote_args(metadata, tmp_path, training_dir))

    assert code == 0
    assert fake_condor == [log_dir / "submit.jdl"]
    # Nothing has to be on a shared file system
    assert checked == []
    lines = (log_dir / "submit.jdl").read_text().splitlines()
    assert "+RemoteJob = True" in lines
    assert not any("ProvidesETPResources" in line for line in lines)

    (bundle_url,) = storage
    assert bundle_url.startswith(
        "root://cmsdcache-kit-disk.gridka.de:1094//store/user/mmolch/xyh_ml/"
        "sandboxes/bundle-"
    )
    script = (log_dir / "job.sh").read_text()
    assert ENV_PACK_URL in script
    assert bundle_url in script

    manifest = json.loads((log_dir / "jobs.json").read_text())
    assert manifest["sandbox"] == {
        "environment_url": ENV_PACK_URL,
        "bundle_url": bundle_url,
    }
    assert manifest["cluster"] == 77
    names = (log_dir / "jobs.txt").read_text().splitlines()
    arguments = (
        (log_dir / "commands" / f"{names[0]}.args").read_text().splitlines()
    )
    assert arguments[0] == "env/bin/python"
    assert "metadata_dir=metadata" in arguments
    assert f"metadata_dir={metadata}" not in arguments
    assert f"artifacts.training_dir=models/{TRAINING_DIR}" in arguments
    assert "nn_friend_tag=tag" in arguments


def test_main_remote_bundle_content(
    metadata, tmp_path, training_dir, fake_condor, storage
):
    """Only the channels of the run, and the files the jobs need."""
    args = _remote_args(metadata, tmp_path, training_dir)

    assert pm.main(args) == 0

    (content,) = storage.values()
    with tarfile.open(fileobj=io.BytesIO(content)) as archive:
        names = set(archive.getnames())
    assert "src/xyh_ml/__init__.py" in names
    assert "config/produce.yaml" in names
    assert "metadata/2024/DY.json.gz" in names
    assert "models/fold4of5/et/best_model.ckpt" in names
    assert "models/fold0of5/et/transformations.pkl" in names
    assert not any("/mt/" in name or "/tt/" in name for name in names)


def test_main_remote_missing_model_fails_at_submission(
    metadata, tmp_path, training_dir, fake_condor, storage, caplog
):
    (training_dir / "fold3of5" / "et" / "best_model.ckpt").unlink()

    code = pm.main(_remote_args(metadata, tmp_path, training_dir))

    assert code == 2
    assert fake_condor == []
    assert "best_model.ckpt" in caplog.text
    assert storage == {}


def test_main_remote_sandbox_dir_option(
    metadata, tmp_path, training_dir, fake_condor, storage
):
    args = _remote_args(
        metadata,
        tmp_path,
        training_dir,
        "--sandbox-dir",
        "root://other//store/mine",
    )

    assert pm.main(args) == 0

    (bundle_url,) = storage
    assert bundle_url.startswith("root://other//store/mine/bundle-")


def test_main_remote_requirements_option(
    metadata, tmp_path, training_dir, fake_condor, storage
):
    args = _remote_args(
        metadata,
        tmp_path,
        training_dir,
        "--condor-requirements",
        'TARGET.CloudSite =?= "BwForCluster-Nemo2"',
    )

    assert pm.main(args) == 0

    lines = (tmp_path / "run" / "submit.jdl").read_text().splitlines()
    assert (
        'requirements = (TARGET.CloudSite =?= "BwForCluster-Nemo2")' in lines
    )


def test_main_remote_needs_upload(
    metadata, tmp_path, training_dir, fake_condor, storage, caplog
):
    args = _remote_args(metadata, tmp_path, training_dir)
    args.remove("nn_friend_tag=tag")

    assert pm.main(args) == 2
    assert "nn_friend_tag" in caplog.text
    assert fake_condor == []
    assert storage == {}


def test_main_remote_dry_run_writes_and_uploads_nothing(
    metadata, tmp_path, training_dir, fake_condor, storage, capsys
):
    args = _remote_args(metadata, tmp_path, training_dir, "--dry-run")

    assert pm.main(args) == 0

    assert fake_condor == []
    assert storage == {}
    assert not (tmp_path / "run").exists()
    assert "env/bin/python" in capsys.readouterr().out


def test_main_remote_proxy_problem_uploads_nothing(
    metadata, tmp_path, training_dir, fake_condor, storage, monkeypatch, caplog
):
    def no_proxy():
        raise condor.CondorError("No valid proxy")

    monkeypatch.setattr(pm, "check_proxy", no_proxy)

    assert pm.main(_remote_args(metadata, tmp_path, training_dir)) == 2
    assert storage == {}
    assert "No valid proxy" in caplog.text


@pytest.mark.parametrize(
    "options",
    [
        ["--gpus", "0"],
        ["--profile"],
        ["--sandbox-dir", "/work/local"],
    ],
)
def test_main_remote_rejected_options(
    metadata, tmp_path, training_dir, options
):
    with pytest.raises(SystemExit):
        pm.main(_remote_args(metadata, tmp_path, training_dir, *options))


def test_sandbox_dir_needs_remote(metadata, tmp_path):
    with pytest.raises(SystemExit):
        pm.main(
            _args(
                metadata,
                tmp_path,
                "--condor",
                "--sandbox-dir",
                "root://other//store/mine",
            )
        )


def test_requirements_need_condor(metadata, tmp_path):
    with pytest.raises(SystemExit):
        pm.main(_args(metadata, tmp_path, "--condor-requirements", "x"))


def test_collect_takes_no_remote_option(tmp_path):
    with pytest.raises(SystemExit):
        pm.main(["--collect", str(tmp_path), "--condor-remote"])


def test_main_condor_does_not_use_sandbox(
    metadata, tmp_path, fake_condor, monkeypatch
):
    def fail(*args):
        raise AssertionError("no sandbox for local jobs")

    monkeypatch.setattr(pm, "environment_pack", fail)
    monkeypatch.setattr(pm, "job_bundle", fail)

    assert pm.main(_args(metadata, tmp_path, "--condor")) == 0

    manifest = json.loads((tmp_path / "run" / "jobs.json").read_text())
    assert "sandbox" not in manifest


def test_collect_of_remote_run(
    metadata, tmp_path, training_dir, fake_condor, storage, monkeypatch
):
    """HTCondor copies the status files to the same place as in local runs."""
    assert pm.main(_remote_args(metadata, tmp_path, training_dir)) == 0
    log_dir = tmp_path / "run"
    jobs = json.loads((log_dir / "jobs.json").read_text())["jobs"]
    # First job complete, second job crashed with the placeholder status file
    _report(log_dir, jobs[0], produced=jobs[0]["input_files"])
    placeholder = log_dir / "status" / f"{pm.Job(**jobs[1]).name}.json"
    placeholder.write_text("{}\n")
    _states(monkeypatch)

    assert pm.main(["--collect", str(log_dir)]) == 1

    summary = _summary(log_dir)
    assert (summary["n_jobs"], summary["n_failed"]) == (2, 1)
    assert (log_dir / "failed_files.txt").read_text().splitlines() == (
        jobs[1]["input_files"]
    )
