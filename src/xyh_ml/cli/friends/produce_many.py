"""Produce ntuple friends for many samples in parallel.

The input files of the selected samples are split into jobs of ``--n-files``
files. Each job is one subprocess of ``xyh_ml.cli.friends.produce``, several
jobs run at the same time and are distributed over the available GPUs. A job
holds files of one era and channel, so it loads one set of models, but it can
span several samples, so that small samples share the startup cost.

The metadata of the samples has to exist, see
``xyh_ml.cli.friends.scrape_metadata``.

Examples (from the repo root; arguments after the options are Hydra overrides
passed on to every ``produce`` job)::

    # All 2024 samples of a KingMaker sample list, 8 jobs on all GPUs
    pixi run -e deepthought python -m xyh_ml.cli.friends.produce_many \\
        --samples-file ../KingMaker/sample_lists/samples_2024_signal.txt \\
        --campaign 2024_nano_v15 -j 8 \\
        mode.decay_mode=y2b_h2tau inputs=y2b_h2tau

    # Samples matching a regular expression, only the et channel; list the
    # jobs without running them
    pixi run -e deepthought python -m xyh_ml.cli.friends.produce_many \\
        --sample 'NMSSM.*MX-2500.*' --channel et --dry-run \\
        mode.decay_mode=y2b_h2tau inputs=y2b_h2tau

Files whose friend file exists are skipped with ``--skip-existing`` before the
jobs are formed, so an interrupted or partly failed run can be continued by
repeating the command. Every job writes its output to
``<log-dir>/logs/<job>.log``; the summary of all jobs is written to
``<log-dir>/summary.json`` and the input files without friend to
``<log-dir>/failed_files.txt``.

With ``--condor``, the jobs are submitted to the ETP-local sites of the HTCondor
pool on CPU slots instead of running on this machine. This needs a valid grid
proxy, and everything has to be on a shared file system::

    python -m xyh_ml.cli.friends.produce_many \\
        --samples-file ../KingMaker/sample_lists/samples_2024_signal.txt \\
        --campaign 2024_nano_v15 --n-files 30 --skip-existing --condor \\
        mode.decay_mode=y2b_h2tau inputs=y2b_h2tau nn_friend_tag=<tag>

    # Afterwards, on the same machine: write the summary and list the failures
    python -m xyh_ml.cli.friends.produce_many \\
        --collect output/produce_many/<timestamp>

A job that fails is retried. A retry repeats all files of the job, unless
``--skip-existing`` is used, so this is recommended. Repeat the command with
``--skip-existing`` to submit what is still missing.

With ``--condor-remote``, the jobs do not need the shared file systems, so they
can also run at remote sites such as NEMO2. They fetch the environment and a
bundle with code, models and metadata from grid storage, and the friends have to
be uploaded with ``nn_friend_tag``. Both files are stored in ``--sandbox-dir`` and
built only if needed; the environment pack is made with ``pixi-pack`` from the
lock file of this repository, so run this from the ``cpu`` environment or another
one with ``pixi-pack``. ``--collect`` works as for ``--condor``::

    python -m xyh_ml.cli.friends.produce_many \\
        --samples-file ../KingMaker/sample_lists/samples_2024_signal.txt \\
        --campaign 2024_nano_v15 --n-files 30 --skip-existing --condor-remote \\
        mode.decay_mode=y2b_h2tau inputs=y2b_h2tau nn_friend_tag=<tag>
"""

import argparse
import json
import logging
import os
import queue
import re
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from xyh_ml import PROJECT_DIR
from xyh_ml.friend_production.condor import (
    JOB_STATUS_HELD,
    REMOTE_STATUS_FILE,
    CondorError,
    CondorOptions,
    RemoteSandbox,
    check_proxy,
    check_shared_paths,
    job_states,
    submit,
    write_submission,
)
from xyh_ml.friend_production.dataclasses import Sample
from xyh_ml.friend_production.metadata import (
    available_samples,
    load_sample_metadata,
    metadata_path,
    read_samples_file,
    single_file_sample,
)
from xyh_ml.friend_production.paths import (
    campaign_name,
    default_output_file,
    fold_training_dirs,
    list_directory,
    project_path,
    resolve_output_dir,
    select_campaigns,
)
from xyh_ml.friend_production.sandbox import (
    METADATA_DIR,
    MODELS_DIR,
    environment_pack,
    job_bundle,
)
from xyh_ml.friend_production.xrootd import is_xrootd_url, xrootd_url

logger = logging.getLogger(__name__)

PRODUCE_MODULE = "xyh_ml.cli.friends.produce"

# Overrides that are set per job by this script
RESERVED_OVERRIDES = [
    "input_files",
    "status_file",
    "skip_existing",
    "profile_file",
    "hydra.run.dir",
    "hydra.output_subdir",
    "hydra/job_logging",
]

# Overrides that are additionally set per job by the job script with --condor
RESERVED_OVERRIDES_CONDOR = ["staging_dir"]

# Overrides with paths on this machine, which are used to find the files for
# the bundle of remote jobs, and are replaced in the commands of these jobs
REMOTE_REPLACED_OVERRIDES = ["metadata_dir", "artifacts.training_dir"]

# Paths in remote jobs, relative to the scratch directory where they run
REMOTE_PYTHON = "env/bin/python"
REMOTE_HYDRA_DIR = "hydra"
REMOTE_MODEL_DIR = "fold${mode.fold}of${mode.n_folds}/${mode.channel}"

# Result of a job that may still run, and the list of submitted jobs
STATUS_UNFINISHED = "unfinished"
MANIFEST_NAME = "jobs.json"

# Number of jobs at the same time, by default
DEFAULT_JOBS_LOCAL = 4
DEFAULT_JOBS_CONDOR = 100

# Limit of the operating system for the length of one command line argument
# (MAX_ARG_STRLEN on Linux)
MAX_ARGUMENT_LENGTH = 128 * 1024

# Environment variables limiting the CPU threads of a job
THREAD_VARIABLES = [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]


@dataclass
class InputFile:
    """One main file of a sample and the path of its friend."""

    era: str
    campaign: str
    sample: str
    channel: str
    input_file: str
    output_file: str


@dataclass
class Job:
    """Input files of one era and channel, processed by one ``produce`` run."""

    index: int
    era: str
    campaign: str
    channel: str
    input_files: list[str] = field(repr=False)
    output_files: list[str] = field(repr=False)
    samples: list[str] = field(repr=False)

    @property
    def name(self) -> str:
        return f"{self.campaign}__{self.channel}__{self.index:05d}"


@dataclass
class JobResult:
    name: str
    status: str  # "ok", "failed" or "skipped"
    returncode: int | None
    duration_s: float
    log_file: str | None
    gpu: str | None
    command: list[str]


# ------------------------------------------------------------ job selection


def select_samples(
    metadata_dir: Path,
    eras: list[str],
    patterns: list[str],
    names: list[str],
) -> tuple[list[tuple[str, str]], list[str]]:
    """Pairs of era and sample whose metadata file exists and whose name is
    in ``names`` or matches one of the regular expressions ``patterns`` (whole
    name).

    ``names`` are looked up directly; the metadata directories are only
    listed for ``patterns``. Returns the pairs and the entries of ``names``
    without metadata file in any of the ``eras``.
    """
    compiled = [re.compile(p) for p in patterns]
    available = available_samples(metadata_dir, eras) if compiled else {}

    selected = []
    for era in eras:
        matches = {
            s
            for s in available.get(era, [])
            if any(p.fullmatch(s) for p in compiled)
        }
        matches.update(
            n for n in names if metadata_path(metadata_dir, era, n).exists()
        )
        selected.extend((era, s) for s in sorted(matches))

    missing = sorted(set(names).difference(s for _, s in selected))
    return selected, missing


def list_input_files(
    samples: list[Sample],
    campaigns: list,
    output_dir: Path | str,
    channels: list[str] | None = None,
) -> list[InputFile]:
    """All main files of ``samples`` in the selected ``channels`` (default:
    all) with the paths of their friends."""
    files = []
    for s in samples:
        if channels is not None and s.channel not in channels:
            continue
        for i, main_file in enumerate(s.main_files):
            files.append(
                InputFile(
                    era=s.era,
                    campaign=campaign_name(campaigns, s.era),
                    sample=s.sample,
                    channel=s.channel,
                    input_file=main_file,
                    output_file=str(
                        default_output_file(
                            output_dir, single_file_sample(s, i)
                        )
                    ),
                )
            )
    return files


def drop_existing(files: list[InputFile]) -> list[InputFile]:
    """``files`` without those whose friend file exists.

    Only the output directories of ``files`` are listed, each once.
    """
    listings: dict[str, set[str]] = {}
    remaining = []
    for file in files:
        directory, name = file.output_file.rsplit("/", 1)
        if directory not in listings:
            listings[directory] = list_directory(directory)
        if name not in listings[directory]:
            remaining.append(file)
    return remaining


def build_jobs(files: list[InputFile], n_files: int) -> list[Job]:
    """Split ``files`` into jobs of up to ``n_files`` files.

    Files are grouped by era and channel, so that a job loads one set of
    models, and sorted by sample and file name within a group. A job can span
    several samples, so that small samples do not get a job of their own.
    """
    if n_files < 1:
        raise ValueError(f"n_files must be >= 1, got {n_files}.")

    groups: dict[tuple[str, str], list[InputFile]] = {}
    for file in files:
        groups.setdefault((file.era, file.channel), []).append(file)

    jobs = []
    for (era, channel), group in sorted(groups.items()):
        group = sorted(group, key=lambda f: (f.sample, f.input_file))
        for index, start in enumerate(range(0, len(group), n_files)):
            batch = group[start : start + n_files]
            jobs.append(
                Job(
                    index=index,
                    era=era,
                    campaign=batch[0].campaign,
                    channel=channel,
                    input_files=[f.input_file for f in batch],
                    output_files=[f.output_file for f in batch],
                    samples=sorted({f.sample for f in batch}),
                )
            )
    return jobs


def status_file(log_dir: Path, job: Job) -> Path:
    """File in which ``produce`` reports the result per input file."""
    return log_dir / "status" / f"{job.name}.json"


@dataclass
class JobPaths:
    """Paths in the command of a job, as seen by the job."""

    python: str | Path
    status_file: str | Path
    hydra_dir: str | Path
    profile_file: str | Path | None = None


def command_with_paths(
    job: Job,
    overrides: list[str],
    skip_existing: bool,
    paths: JobPaths,
) -> list[str]:
    """Command running ``produce`` for ``job`` with the given ``paths``.

    The output of a job goes to its log file, so Hydra does not write its own
    log and config files per job.
    """
    input_files = ",".join(f"'{f}'" for f in job.input_files)
    command = [
        str(paths.python),
        "-m",
        PRODUCE_MODULE,
        *overrides,
        f"input_files=[{input_files}]",
        f"status_file={paths.status_file}",
        f"skip_existing={str(skip_existing).lower()}",
        f"hydra.run.dir={paths.hydra_dir}",
        "hydra.output_subdir=null",
        "hydra/job_logging=stdout",
    ]
    if paths.profile_file is not None:
        command.append(f"profile_file={paths.profile_file}")
    return command


def job_command(
    job: Job,
    overrides: list[str],
    log_dir: Path,
    skip_existing: bool,
    profile: bool,
    python: str = sys.executable,
) -> list[str]:
    """Command running ``produce`` for ``job`` on a shared file system."""
    return command_with_paths(
        job,
        overrides,
        skip_existing,
        JobPaths(
            python=python,
            status_file=status_file(log_dir, job),
            hydra_dir=log_dir / "hydra",
            profile_file=(
                log_dir / "profiles" / f"{job.name}.json" if profile else None
            ),
        ),
    )


def remote_job_command(
    job: Job,
    overrides: list[str],
    skip_existing: bool,
) -> list[str]:
    """Command running ``produce`` for ``job`` with the files of the bundle."""
    kept = [
        override
        for override in overrides
        if override_key(override) not in REMOTE_REPLACED_OVERRIDES
    ]
    return command_with_paths(
        job,
        [
            *kept,
            f"metadata_dir={METADATA_DIR}",
            f"artifacts.training_dir={MODELS_DIR}/{REMOTE_MODEL_DIR}",
        ],
        skip_existing,
        JobPaths(
            python=REMOTE_PYTHON,
            status_file=REMOTE_STATUS_FILE,
            hydra_dir=REMOTE_HYDRA_DIR,
        ),
    )


def check_command_length(command: list[str]):
    """Reject a command that the operating system would refuse to run."""
    longest = max(len(argument.encode()) for argument in command)
    if longest >= MAX_ARGUMENT_LENGTH:
        msg = (
            f"A command line argument has {longest} bytes, the limit is "
            + f"{MAX_ARGUMENT_LENGTH}. Use a smaller value of --n-files."
        )
        raise ValueError(msg)


def override_key(override: str) -> str:
    """Name of the config value set by a Hydra override."""
    return override.split("=", 1)[0].lstrip("+~")


def check_overrides(overrides: list[str], condor: bool = False):
    """Reject overrides that this script sets per job."""
    reserved = RESERVED_OVERRIDES + (
        RESERVED_OVERRIDES_CONDOR if condor else []
    )
    for override in overrides:
        if override_key(override) in reserved:
            msg = (
                f"Override {override!r} is set per job by this script; use "
                + "its options instead."
            )
            raise ValueError(msg)


def failed_files(job: Job, result: JobResult, log_dir: Path) -> list[str]:
    """Input files of ``job`` that did not get a friend.

    A job that succeeded has none. Otherwise, these are the files that
    ``produce`` reports as failed and those without a known result, as after
    a crash no status file is left.
    """
    if result.status == "ok":
        return []
    try:
        status = json.loads(status_file(log_dir, job).read_text())
        done = set(status["produced"]) | set(status["skipped"])
    except (OSError, ValueError, KeyError):
        return list(job.input_files)
    return [f for f in job.input_files if f not in done]


# ---------------------------------------------------------------- execution


class Runner:
    """Run jobs as subprocesses, at most ``n_parallel`` at the same time.

    Every running job holds one slot; the slots are distributed evenly over
    ``gpus`` (``None``: ``CUDA_VISIBLE_DEVICES`` is not changed).
    """

    def __init__(
        self,
        n_parallel: int,
        gpus: list[str] | None,
        threads_per_job: int | None,
        log_dir: Path,
    ):
        if n_parallel < 1:
            raise ValueError(f"n_parallel must be >= 1, got {n_parallel}.")
        self.n_parallel = n_parallel
        self.slots = queue.Queue()
        for i in range(n_parallel):
            self.slots.put(gpus[i % len(gpus)] if gpus else None)
        self.threads_per_job = threads_per_job
        self.log_dir = log_dir
        self.processes: set[subprocess.Popen] = set()
        self.lock = threading.Lock()
        self.stopping = threading.Event()

    def _environment(self, gpu: str | None) -> dict[str, str]:
        env = dict(os.environ)
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = gpu
        if self.threads_per_job is not None:
            for variable in THREAD_VARIABLES:
                env[variable] = str(self.threads_per_job)
        return env

    def run_job(self, job: Job, command: list[str]) -> JobResult:
        gpu = self.slots.get()
        log_file = self.log_dir / "logs" / f"{job.name}.log"
        start = time.perf_counter()
        try:
            if self.stopping.is_set():
                return JobResult(
                    job.name, "skipped", None, 0.0, None, gpu, command
                )
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "w") as log:
                log.write(" ".join(command) + "\n\n")
                log.flush()
                process = subprocess.Popen(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=self._environment(gpu),
                    cwd=PROJECT_DIR,
                )
                with self.lock:
                    self.processes.add(process)
                try:
                    returncode = process.wait()
                finally:
                    with self.lock:
                        self.processes.discard(process)
        finally:
            self.slots.put(gpu)

        return JobResult(
            name=job.name,
            status="ok" if returncode == 0 else "failed",
            returncode=returncode,
            duration_s=time.perf_counter() - start,
            log_file=str(log_file),
            gpu=gpu,
            command=command,
        )

    def stop(self):
        """Do not start further jobs and terminate the running ones."""
        self.stopping.set()
        with self.lock:
            processes = list(self.processes)
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()

    def run(self, jobs: list[tuple[Job, list[str]]]) -> list[JobResult]:
        results = []
        with ThreadPoolExecutor(max_workers=self.n_parallel) as executor:
            futures = {
                executor.submit(self.run_job, job, command): job
                for job, command in jobs
            }
            try:
                for i, future in enumerate(as_completed(futures), start=1):
                    result = future.result()
                    results.append(result)
                    logger.log(
                        logging.INFO
                        if result.status == "ok"
                        else logging.WARNING,
                        f"[{i}/{len(jobs)}] {result.status.upper()} "
                        + f"{result.name} ({result.duration_s:.0f} s, "
                        + f"GPU {result.gpu}): {result.log_file}",
                    )
            except KeyboardInterrupt:
                logger.warning("Interrupted, terminating the running jobs")
                self.stop()
                raise
        return results


# ---------------------------------------------------------------------- CLI


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="Further arguments are Hydra overrides for every produce "
        + "job, e.g. mode.decay_mode=y2b_h2tau inputs=y2b_h2tau.",
    )
    parser.add_argument(
        "--sample",
        action="append",
        default=[],
        metavar="REGEX",
        help="Regular expression matching whole sample names (repeatable). "
        + "Use '.*' for all samples.",
    )
    parser.add_argument(
        "--samples-file",
        action="append",
        default=[],
        type=Path,
        help="Text file with one sample name or DAS dataset per line "
        + "(repeatable), e.g. a KingMaker sample list.",
    )
    parser.add_argument(
        "--campaign",
        action="append",
        default=None,
        help="Only this campaign (repeatable; default: all).",
    )
    parser.add_argument(
        "--channel",
        action="append",
        default=None,
        help="Only this channel (repeatable; default: all).",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        help="Number of jobs running at the same time (default: "
        + f"{DEFAULT_JOBS_LOCAL}), or in the queue with --condor (default: "
        + f"{DEFAULT_JOBS_CONDOR}).",
    )
    parser.add_argument(
        "--n-files",
        type=int,
        default=20,
        help="Number of input files per job (default: 20).",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        help="Comma-separated GPU indices to distribute the jobs over "
        + "(default: all GPUs found; 'none' leaves CUDA_VISIBLE_DEVICES "
        + "unchanged).",
    )
    parser.add_argument(
        "--threads-per-job",
        type=int,
        default=2,
        help="CPU threads per job (OMP_NUM_THREADS etc., default: 2).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files whose friend file exists.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Write a profile per job to <log-dir>/profiles.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Directory for logs and summary (default: "
        + "output/produce_many/<timestamp>).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list the jobs and their commands.",
    )
    parser.add_argument(
        "--condor",
        action="store_true",
        help="Submit the jobs to HTCondor instead of running them here.",
    )
    parser.add_argument(
        "--condor-remote",
        action="store_true",
        help="Submit jobs to HTCondor that do not need the shared file "
        + "systems, so that they can also run at remote sites. Needs "
        + "nn_friend_tag.",
    )
    parser.add_argument(
        "--sandbox-dir",
        default=None,
        metavar="URL",
        help="XRootD directory for the environment and the bundle of jobs "
        + "with --condor-remote (default: xyh_ml/sandboxes in the user "
        + "directory of the ntuple server).",
    )
    parser.add_argument(
        "--condor-requirements",
        default=None,
        metavar="EXPRESSION",
        help="Additional HTCondor requirements, e.g. to select a site: "
        + "'TARGET.CloudSite =?= \"BwForCluster-Nemo2\"'.",
    )
    parser.add_argument(
        "--collect",
        type=Path,
        default=None,
        metavar="LOG_DIR",
        help="Evaluate a run submitted with --condor: write the summary and "
        + "the failed files. Takes no further options.",
    )
    parser.add_argument(
        "--condor-memory",
        type=int,
        default=CondorOptions.memory_mb,
        metavar="MB",
        help="Memory per job (default: %(default)s), increased at restarts.",
    )
    parser.add_argument(
        "--condor-disk",
        type=int,
        default=CondorOptions.disk_gb,
        metavar="GB",
        help="Scratch disk per job (default: %(default)s).",
    )
    parser.add_argument(
        "--condor-walltime",
        type=float,
        default=CondorOptions.walltime_hours,
        metavar="HOURS",
        help="Runtime per job (default: %(default)s).",
    )
    parser.add_argument(
        "--accounting-group",
        default=CondorOptions.accounting_group,
        help="HTCondor accounting group (default: %(default)s).",
    )
    parser.add_argument("overrides", nargs="*", help="Hydra overrides.")
    args = parser.parse_args(argv)
    if args.collect is not None:
        if (
            args.condor
            or args.condor_remote
            or args.sample
            or args.samples_file
            or args.overrides
        ):
            parser.error("--collect takes no further options.")
        return args
    if args.condor_remote:
        args.condor = True
        if args.profile:
            parser.error(
                "--profile can not be used with --condor-remote, as profiles "
                + "are not sent back."
            )
    if args.sandbox_dir is not None:
        if not args.condor_remote:
            parser.error("--sandbox-dir needs --condor-remote.")
        if not is_xrootd_url(args.sandbox_dir):
            parser.error("--sandbox-dir has to be an XRootD URL.")
    if args.condor_requirements is not None and not args.condor:
        parser.error(
            "--condor-requirements needs --condor or --condor-remote."
        )
    if args.condor and args.gpus is not None:
        parser.error("--gpus can not be used with --condor.")
    if args.jobs is None:
        args.jobs = DEFAULT_JOBS_CONDOR if args.condor else DEFAULT_JOBS_LOCAL
    if not args.sample and not args.samples_file:
        parser.error("Select samples with --sample and/or --samples-file.")
    if args.n_files < 1:
        parser.error("--n-files must be at least 1.")
    return args


def _compose(overrides: list[str]) -> DictConfig:
    with initialize_config_dir(
        str(PROJECT_DIR / "config"), version_base="1.3"
    ):
        return compose("produce", overrides=overrides)


def _default_gpus() -> list[str] | None:
    import torch

    n = torch.cuda.device_count()
    return [str(i) for i in range(n)] if n else None


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
    )
    args = _parse_args(argv)
    if args.collect is not None:
        try:
            return collect(args.collect)
        except CondorError as e:
            logger.error(e)
            return 2

    try:
        check_overrides(args.overrides, condor=args.condor)
    except ValueError as e:
        logger.error(e)
        return 2

    cfg = _compose(args.overrides)
    metadata_dir = project_path(cfg.metadata_dir)

    # Local directory or XRootD URL (with nn_friend_tag)
    try:
        campaigns = select_campaigns(cfg.campaigns, args.campaign)
        output_dir = resolve_output_dir(cfg)
    except ValueError as e:
        logger.error(e)
        return 2
    if not isinstance(output_dir, str):
        output_dir = project_path(output_dir)
    if args.condor_remote and not is_xrootd_url(output_dir):
        logger.error(
            "Jobs at remote sites can only upload their friends, so "
            + "--condor-remote needs nn_friend_tag."
        )
        return 2

    names = [n for f in args.samples_file for n in read_samples_file(f)]
    selected, missing = select_samples(
        metadata_dir, [c.era for c in campaigns], args.sample, names
    )
    if missing:
        logger.warning(
            f"{len(missing)} samples of the sample files have no metadata "
            + f"file in {metadata_dir}:\n  "
            + "\n  ".join(missing)
        )

    samples = [
        record
        for era, sample in selected
        for record in load_sample_metadata(metadata_dir, era, sample)
    ]
    files = list_input_files(samples, campaigns, output_dir, args.channel)

    if args.skip_existing:
        n_files = len(files)
        files = drop_existing(files)
        if n_files != len(files):
            logger.info(
                f"Skipping {n_files - len(files)} files whose friends exist"
            )

    jobs = build_jobs(files, args.n_files)
    if not jobs:
        logger.warning("No jobs to run.")
        return 0

    log_dir = args.log_dir or (
        project_path("output/produce_many")
        / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    log_dir = log_dir.absolute()
    commands = [
        (
            job,
            remote_job_command(job, args.overrides, args.skip_existing)
            if args.condor_remote
            else job_command(
                job, args.overrides, log_dir, args.skip_existing, args.profile
            ),
        )
        for job in jobs
    ]
    try:
        for _, command in commands:
            check_command_length(command)
    except ValueError as e:
        logger.error(e)
        return 2
    logger.info(
        f"{len(jobs)} jobs ({len(files)} files) of {len(selected)} samples, "
        + f"friends in {output_dir}"
    )

    if args.dry_run:
        counts: dict[tuple[str, str], list[int]] = {}
        for job in jobs:
            count = counts.setdefault((job.campaign, job.channel), [0, 0])
            count[0] += 1
            count[1] += len(job.input_files)
        for (campaign, channel), (n_jobs, n_files) in counts.items():
            print(f"{campaign} {channel}: {n_jobs} jobs, {n_files} files")
        print("Example command:")
        print("  " + " ".join(commands[0][1]))
        return 0

    info = {
        "overrides": args.overrides,
        "metadata_dir": str(metadata_dir),
        "output_dir": str(output_dir),
        "n_files_per_job": args.n_files,
    }
    if args.condor:
        try:
            cluster = _submit_jobs(args, cfg, commands, log_dir, info)
        except CondorError as e:
            logger.error(e)
            return 2
        logger.info(
            f"Submitted {len(jobs)} jobs to cluster {cluster}, logs in "
            + f"{log_dir}. Monitor with: condor_q {cluster}. When finished, "
            + "run this script with --collect "
            + f"{log_dir}"
        )
        return 0

    if args.gpus is None:
        gpus = _default_gpus()
    elif args.gpus.lower() == "none":
        gpus = None
    else:
        gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    logger.info(
        f"Running {min(args.jobs, len(jobs))} jobs at a time on GPUs {gpus}, "
        + f"logs in {log_dir}"
    )

    log_dir.mkdir(parents=True, exist_ok=True)
    runner = Runner(args.jobs, gpus, args.threads_per_job, log_dir)
    start = time.perf_counter()
    results = runner.run(commands)
    duration = time.perf_counter() - start

    failed, _ = _write_results(
        results, jobs, log_dir, info | {"duration_s": duration}
    )
    logger.info(
        f"{len(results) - len(failed)}/{len(results)} jobs succeeded in "
        + f"{duration / 60:.1f} min; summary: {log_dir / 'summary.json'}"
    )
    return 1 if failed else 0


def _write_results(
    results: list[JobResult],
    jobs: list[Job],
    log_dir: Path,
    info: dict,
) -> tuple[list[JobResult], list[str]]:
    """Write ``summary.json`` and ``failed_files.txt`` and report failures.

    Returns the jobs that failed and the input files without friend. Unfinished
    jobs (``--collect``) do not count as failed.
    """
    jobs_by_name = {job.name: job for job in jobs}
    finished = [r for r in results if r.status != STATUS_UNFINISHED]
    failed = [r for r in finished if r.status != "ok"]
    summary_jobs = []
    missing_friends = []
    for result in results:
        job = jobs_by_name[result.name]
        job_failed_files = (
            failed_files(job, result, log_dir) if result in finished else []
        )
        missing_friends.extend(job_failed_files)
        summary_jobs.append(
            asdict(result)
            | {
                "input_files": job.input_files,
                "failed_files": job_failed_files,
            }
        )
    summary = info | {
        "n_jobs": len(results),
        "n_failed": len(failed),
        "n_files": sum(len(job.input_files) for job in jobs),
        "n_failed_files": len(missing_friends),
        "jobs": summary_jobs,
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    failed_list = log_dir / "failed_files.txt"
    failed_list.unlink(missing_ok=True)
    if failed:
        failed_list.write_text("".join(f"{f}\n" for f in missing_friends))
        logger.error(
            f"{len(failed)} jobs failed, {len(missing_friends)} files without "
            + f"friend ({failed_list}):\n"
            + "\n".join(f"  {r.name}: {r.log_file}" for r in failed)
        )
    return failed, missing_friends


# ------------------------------------------------------------------- HTCondor


def default_sandbox_dir(cfg: DictConfig) -> str:
    """``xyh_ml/sandboxes`` in the user directory of the ntuples on their
    server, with a ``base_dir`` of the form ``/store/user/<name>/...``."""
    user_dir = "/".join(cfg.ntuple.base_dir.split("/")[:4])
    return xrootd_url(cfg.ntuple.xrootd_server, f"{user_dir}/xyh_ml/sandboxes")


def _build_sandbox(
    args: argparse.Namespace,
    cfg: DictConfig,
    jobs: list[Job],
    metadata_dir: Path,
) -> RemoteSandbox:
    """Provide the files that remote jobs fetch from grid storage."""
    sandbox_dir = args.sandbox_dir or default_sandbox_dir(cfg)
    model_dirs = {
        f"fold{fold}of{cfg.mode.n_folds}/{channel}": project_path(directory)
        for channel in sorted({job.channel for job in jobs})
        for fold, directory in fold_training_dirs(cfg, channel).items()
    }
    metadata_files = {}
    for job in jobs:
        for sample in job.samples:
            path = metadata_path(metadata_dir, job.era, sample)
            metadata_files[path.relative_to(metadata_dir).as_posix()] = path

    # The bundle comes first, as it is quick and finds missing files
    bundle_url = job_bundle(
        sandbox_dir, PROJECT_DIR, model_dirs, metadata_files
    )
    return RemoteSandbox(
        environment_url=environment_pack(sandbox_dir, PROJECT_DIR),
        bundle_url=bundle_url,
    )


def _submit_jobs(
    args: argparse.Namespace,
    cfg: DictConfig,
    commands: list[tuple[Job, list[str]]],
    log_dir: Path,
    info: dict,
) -> int:
    """Check the prerequisites, write the files of the submission and submit
    it. Returns the cluster ID."""
    jobs = [job for job, _ in commands]
    sandbox = None
    if args.condor_remote:
        proxy = check_proxy()
        sandbox = _build_sandbox(args, cfg, jobs, Path(info["metadata_dir"]))
        info = info | {"sandbox": asdict(sandbox)}
    else:
        paths = {
            "Python interpreter": sys.executable,
            "repository": PROJECT_DIR,
            "log directory": log_dir,
            "metadata": info["metadata_dir"],
            "training artifacts": project_path(cfg.artifacts.training_dir),
        }
        if not is_xrootd_url(info["output_dir"]):
            paths["output directory"] = info["output_dir"]
        check_shared_paths(paths)
        proxy = check_proxy()

    log_dir.mkdir(parents=True, exist_ok=True)
    submit_file = write_submission(
        log_dir,
        PROJECT_DIR,
        {job.name: command for job, command in commands},
        proxy,
        THREAD_VARIABLES,
        CondorOptions(
            memory_mb=args.condor_memory,
            disk_gb=args.condor_disk,
            walltime_hours=args.condor_walltime,
            accounting_group=args.accounting_group,
            max_materialize=args.jobs,
            threads_per_job=args.threads_per_job,
            requirements=args.condor_requirements,
        ),
        sandbox=sandbox,
    )
    cluster = submit(submit_file)
    manifest = info | {
        "cluster": cluster,
        "submit_host": socket.gethostname(),
        "jobs": [asdict(job) for job in jobs],
    }
    (log_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    return cluster


def _status_is_ok(job: Job, log_dir: Path) -> bool:
    """Whether the status file of ``job`` reports a friend for all files."""
    try:
        status = json.loads(status_file(log_dir, job).read_text())
        done = set(status["produced"]) | set(status["skipped"])
    except (OSError, ValueError, KeyError):
        return False
    return done.issuperset(job.input_files)


def collect(log_dir: Path) -> int:
    """Evaluate a run submitted with ``--condor``.

    Writes the same summary as a local run. Jobs without complete result count
    as unfinished as long as the cluster is active, and as failed afterwards.
    """
    log_dir = log_dir.absolute()
    manifest_file = log_dir / MANIFEST_NAME
    try:
        manifest = json.loads(manifest_file.read_text())
    except (OSError, ValueError) as e:
        raise CondorError(f"Can not read {manifest_file}: {e}") from e

    # Cluster IDs are only valid on the machine of the submission
    host = socket.gethostname()
    if manifest["submit_host"] != host:
        raise CondorError(
            f"The jobs were submitted on {manifest['submit_host']}, run this "
            + f"on that machine instead of {host}."
        )

    jobs = [Job(**job) for job in manifest["jobs"]]
    states, active = job_states(manifest["cluster"])

    results = []
    held = []
    for proc_id, job in enumerate(jobs):
        if states.get(proc_id) == JOB_STATUS_HELD:
            held.append(job.name)
        if proc_id in states or (
            active and not status_file(log_dir, job).exists()
        ):
            status = STATUS_UNFINISHED
        else:
            status = "ok" if _status_is_ok(job, log_dir) else "failed"
        results.append(
            JobResult(
                name=job.name,
                status=status,
                returncode=None,
                duration_s=0.0,
                log_file=str(log_dir / "logs" / f"{job.name}.err"),
                gpu=None,
                command=[],
            )
        )

    info = {
        key: manifest[key]
        for key in [
            "overrides",
            "metadata_dir",
            "output_dir",
            "n_files_per_job",
            "cluster",
        ]
    }
    failed, _ = _write_results(results, jobs, log_dir, info)

    unfinished = [r for r in results if r.status == STATUS_UNFINISHED]
    n_ok = len(results) - len(failed) - len(unfinished)
    logger.info(
        f"Cluster {manifest['cluster']}: {n_ok} jobs ok, {len(failed)} "
        + f"failed, {len(unfinished)} unfinished; summary: "
        + f"{log_dir / 'summary.json'}"
    )
    if held:
        logger.warning(
            f"{len(held)} jobs are held, see condor_q "
            + f"{manifest['cluster']} -hold: {held}"
        )
    if unfinished:
        logger.warning(
            "The cluster is not finished, so this result is preliminary."
        )
    return 1 if failed or unfinished else 0


if __name__ == "__main__":
    sys.exit(main())
