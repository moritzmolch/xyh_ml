"""Submission of friend production jobs to the ETP HTCondor pool.

By default, jobs run at the ETP-local sites, where the shared file systems
with the software, models and metadata are mounted. With a
:class:`RemoteSandbox`, jobs fetch all of this from grid storage, so that they
can also run at remote sites. The arguments of a job are stored in a file, as
the list of input files is too long for the ``arguments`` line of a submit
file.

This module does not import torch, so that planning jobs stays cheap.
"""

import logging
import os
import re
import shlex
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

CONTAINER_IMAGE = (
    "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/"
    "cverstege/alma9-gridjob:latest"
)

# File systems mounted at the ETP-local sites
SHARED_FILE_SYSTEMS = ("/work", "/ceph", "/home")

# Attempts of a job after a non-zero exit code, e.g. after transient failures
# of the storage
MAX_RETRIES = 2

# Additional memory per start of a job, for jobs held for exceeding the limit
MEMORY_INCREASE_PER_START_MB = 2000

# Hold reason code for exceeding the requested memory, and the number of job
# starts up to which such a job is released
HOLD_CODE_MEMORY = 34
MAX_STARTS_AFTER_MEMORY_HOLD = 3

# Name of the status file of a job at a remote site, in its scratch directory
REMOTE_STATUS_FILE = "status.json"

# Attempts to fetch a file at a remote site, and the time between them
FETCH_ATTEMPTS = 3
FETCH_RETRY_DELAY_S = 60

# Minimum lifetime of the grid proxy at submission
MIN_PROXY_LIFETIME_HOURS = 24

# HTCondor job status code
JOB_STATUS_HELD = 5

COMMAND_TIMEOUT_S = 300


class CondorError(RuntimeError):
    """The submission can not be done or its state can not be determined."""


@dataclass
class CondorOptions:
    """Resources and scheduling of the jobs."""

    memory_mb: int = 4000
    disk_gb: int = 5
    walltime_hours: float = 6.0
    accounting_group: str = "cms.higgs"
    max_materialize: int = 100
    threads_per_job: int = 2
    requirements: str | None = None


@dataclass
class RemoteSandbox:
    """Files that jobs fetch from grid storage, as they do not rely on the
    shared file systems."""

    environment_url: str
    bundle_url: str


def proxy_path() -> Path:
    """Path of the grid proxy of the user."""
    path = os.environ.get("X509_USER_PROXY")
    return Path(path) if path else Path(f"/tmp/x509up_u{os.getuid()}")


def check_proxy(
    min_lifetime_hours: float = MIN_PROXY_LIFETIME_HOURS,
) -> Path:
    """Return the path of the grid proxy, which has to be valid for at least
    ``min_lifetime_hours``."""
    path = proxy_path()
    renew = (
        "Renew it with: voms-proxy-init -rfc -bits 4096 "
        + "--voms cms:/cms/country/de --valid 192:00"
    )
    try:
        result = subprocess.run(
            ["voms-proxy-info", "-file", str(path), "--timeleft"],
            capture_output=True,
            text=True,
            check=False,
            timeout=COMMAND_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise CondorError(f"Can not check the proxy {path}: {e}") from e

    lifetime = result.stdout.strip()
    if result.returncode != 0 or not lifetime.isdigit():
        raise CondorError(f"No valid proxy found at {path}. {renew}")

    hours = int(lifetime) / 3600
    if hours < min_lifetime_hours:
        raise CondorError(
            f"The proxy {path} is valid for only {hours:.1f} h, at least "
            + f"{min_lifetime_hours} h are required. {renew}"
        )
    return path


def check_shared_paths(paths: dict[str, str | Path]):
    """Reject paths that do not exist on the worker nodes, given as a
    description per path."""
    outside = [
        f"{description} ({path})"
        for description, path in paths.items()
        if not str(Path(path).absolute()).startswith(
            tuple(f"{fs}/" for fs in SHARED_FILE_SYSTEMS)
        )
    ]
    if outside:
        raise CondorError(
            "Not on a file system of the worker nodes "
            + f"{SHARED_FILE_SYSTEMS}: {', '.join(outside)}"
        )


def job_script(
    project_dir: Path,
    log_dir: Path,
    thread_variables: list[str],
    threads: int,
) -> str:
    """Executable of a job, which gets the name of the job as argument."""
    project = shlex.quote(str(project_dir))
    commands = shlex.quote(str(log_dir / "commands"))
    threads_setup = "\n".join(
        f"export {variable}={threads}" for variable in thread_variables
    )
    return f"""#!/bin/bash
set -euo pipefail

echo "job:     $1"
echo "host:    $(hostname)"
echo "scratch: $_CONDOR_SCRATCH_DIR"

export HOME="$_CONDOR_SCRATCH_DIR"
export PYTHONPATH={project}/src
{threads_setup}

cd {project}
mapfile -t command < {commands}/"$1".args
exec "${{command[@]}}" "staging_dir=$_CONDOR_SCRATCH_DIR/staging"
"""


def remote_job_script(
    sandbox: RemoteSandbox,
    thread_variables: list[str],
    threads: int,
) -> str:
    """Executable of a job that does not rely on the shared file systems,
    which gets the name of the job as argument and runs in scratch."""
    threads_setup = "\n".join(
        f"export {variable}={threads}" for variable in thread_variables
    )
    environment_url = shlex.quote(sandbox.environment_url)
    bundle_url = shlex.quote(sandbox.bundle_url)
    return f"""#!/bin/bash
set -euo pipefail
cd "$_CONDOR_SCRATCH_DIR"

# A missing output file puts a job on hold, so this comes before anything can
# fail. It does not list any file, so the job counts as failed until replaced.
echo '{{}}' > {REMOTE_STATUS_FILE}

echo "job:     $1"
echo "host:    $(hostname)"
echo "scratch: $_CONDOR_SCRATCH_DIR"

export HOME="$_CONDOR_SCRATCH_DIR"
export TMPDIR="$_CONDOR_SCRATCH_DIR/tmp"
mkdir -p "$TMPDIR"
{threads_setup}

fetch() {{
    for attempt in $(seq {FETCH_ATTEMPTS}); do
        if xrdcp -f --nopbar "$1" "$2"; then
            return 0
        fi
        echo "Attempt $attempt to fetch $1 failed" >&2
        if [ "$attempt" -lt {FETCH_ATTEMPTS} ]; then
            sleep {FETCH_RETRY_DELAY_S}
        fi
    done
    return 1
}}

fetch {environment_url} environment.sh
bash environment.sh -q -o "$PWD"
rm -f environment.sh

fetch {bundle_url} bundle.tar.gz
tar -xzf bundle.tar.gz
rm -f bundle.tar.gz

# Only now, as the libraries of the environment could break xrdcp of the image
source activate.sh
export PYTHONPATH="$PWD/src"
export PYTHONNOUSERSITE=1

mapfile -t command < "$1".args
exec "${{command[@]}}" "staging_dir=$_CONDOR_SCRATCH_DIR/staging"
"""


def submit_description(
    log_dir: Path,
    proxy: Path,
    options: CondorOptions,
    remote: bool = False,
) -> str:
    """Content of the submit file, with one job per line of ``jobs.txt``."""
    memory = (
        f"{options.memory_mb} + "
        + f"{MEMORY_INCREASE_PER_START_MB} * NumJobStarts"
    )
    release = (
        f"(HoldReasonCode == {HOLD_CODE_MEMORY}) && "
        + f"(NumJobStarts < {MAX_STARTS_AFTER_MEMORY_HOLD})"
    )
    requirements = [] if remote else ["(TARGET.ProvidesETPResources =?= True)"]
    if options.requirements:
        requirements.append(f"({options.requirements})")
    placement = "+RemoteJob = True\n" if remote else ""
    if requirements:
        placement += f"requirements = {' && '.join(requirements)}\n"
    transfers = (
        f"""transfer_input_files = {log_dir}/commands/$(job).args
transfer_output_files = {REMOTE_STATUS_FILE}
transfer_output_remaps = "{REMOTE_STATUS_FILE} = {log_dir}/status/$(job).json\""""
        if remote
        else 'transfer_output_files = ""'
    )
    return f"""universe = container
container_image = {CONTAINER_IMAGE}

executable = {log_dir}/job.sh
arguments = $(job)
batch_name = produce_many_{log_dir.name}

{placement}accounting_group = {options.accounting_group}
request_cpus = {options.threads_per_job}
request_memory = {memory}
request_disk = {options.disk_gb}G
+RequestWalltime = {int(options.walltime_hours * 3600)}

x509userproxy = {proxy}

should_transfer_files = YES
when_to_transfer_output = ON_EXIT
{transfers}

max_retries = {MAX_RETRIES}
periodic_release = {release}
max_materialize = {options.max_materialize}

log = {log_dir}/condor.log
output = {log_dir}/logs/$(job).out
error = {log_dir}/logs/$(job).err

queue job from {log_dir}/jobs.txt
"""


def write_submission(
    log_dir: Path,
    project_dir: Path,
    commands: dict[str, list[str]],
    proxy: Path,
    thread_variables: list[str],
    options: CondorOptions,
    sandbox: RemoteSandbox | None = None,
) -> Path:
    """Write all files of a submission to ``log_dir``, with the commands per
    job name, and return the path of the submit file.

    With ``sandbox``, the jobs do not rely on the shared file systems, and
    ``project_dir`` is not used.
    """
    for name, command in commands.items():
        if any("\n" in argument for argument in command):
            raise CondorError(
                f"An argument of job {name} contains a line break, which is "
                + "not possible in a command file."
            )

    (log_dir / "logs").mkdir(parents=True, exist_ok=True)
    (log_dir / "commands").mkdir(exist_ok=True)
    for name, command in commands.items():
        (log_dir / "commands" / f"{name}.args").write_text(
            "\n".join(command) + "\n"
        )
    (log_dir / "jobs.txt").write_text("\n".join(commands) + "\n")

    # HTCondor does not create the directory of the remapped status files
    if sandbox is not None:
        (log_dir / "status").mkdir(exist_ok=True)

    script = log_dir / "job.sh"
    script.write_text(
        job_script(
            project_dir, log_dir, thread_variables, options.threads_per_job
        )
        if sandbox is None
        else remote_job_script(
            sandbox, thread_variables, options.threads_per_job
        )
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)

    submit_file = log_dir / "submit.jdl"
    submit_file.write_text(
        submit_description(log_dir, proxy, options, remote=sandbox is not None)
    )
    return submit_file


def _run_condor(command: list[str]) -> str:
    """Output of a HTCondor command."""
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=COMMAND_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise CondorError(f"{command[0]} failed: {e}") from e
    if result.returncode != 0:
        raise CondorError(
            f"{' '.join(command)} failed:\n{result.stderr or result.stdout}"
        )
    return result.stdout


def submit(submit_file: Path) -> int:
    """Submit ``submit_file`` and return the cluster ID."""
    output = _run_condor(["condor_submit", str(submit_file)])
    m = re.search(r"submitted to cluster (\d+)", output)
    if not m:
        raise CondorError(f"No cluster ID in the output:\n{output}")
    return int(m[1])


def job_states(cluster: int) -> tuple[dict[int, int], bool]:
    """Status code per process ID of the jobs of ``cluster`` in the queue, and
    whether the cluster is still active.

    Jobs that are not materialized yet (``max_materialize``) are not listed,
    but the cluster stays active until they have finished, too.
    """
    output = _run_condor(
        ["condor_q", str(cluster), "-af", "ProcId", "JobStatus"]
    )
    states = {}
    for line in output.splitlines():
        proc_id, status = line.split()
        states[int(proc_id)] = int(status)

    factory = _run_condor(
        ["condor_q", str(cluster), "-factory", "-af", "ClusterId"]
    )
    return states, bool(states) or bool(factory.strip())
