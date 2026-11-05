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
"""

import argparse
import json
import logging
import os
import queue
import re
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
    list_directory,
    project_path,
    resolve_output_dir,
    select_campaigns,
)

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


def job_command(
    job: Job,
    overrides: list[str],
    log_dir: Path,
    skip_existing: bool,
    profile: bool,
    python: str = sys.executable,
) -> list[str]:
    """Command running ``produce`` for ``job``.

    The output of a job goes to its log file, so Hydra does not write its own
    log and config files per job.
    """
    input_files = ",".join(f"'{f}'" for f in job.input_files)
    command = [
        python,
        "-m",
        PRODUCE_MODULE,
        *overrides,
        f"input_files=[{input_files}]",
        f"status_file={status_file(log_dir, job)}",
        f"skip_existing={str(skip_existing).lower()}",
        f"hydra.run.dir={log_dir / 'hydra'}",
        "hydra.output_subdir=null",
        "hydra/job_logging=stdout",
    ]
    if profile:
        command.append(
            f"profile_file={log_dir / 'profiles' / f'{job.name}.json'}"
        )
    return command


def check_command_length(command: list[str]):
    """Reject a command that the operating system would refuse to run."""
    longest = max(len(argument.encode()) for argument in command)
    if longest >= MAX_ARGUMENT_LENGTH:
        msg = (
            f"A command line argument has {longest} bytes, the limit is "
            + f"{MAX_ARGUMENT_LENGTH}. Use a smaller value of --n-files."
        )
        raise ValueError(msg)


def check_overrides(overrides: list[str]):
    """Reject overrides that this script sets per job."""
    for override in overrides:
        key = override.split("=", 1)[0].lstrip("+~")
        if key in RESERVED_OVERRIDES:
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
        default=4,
        help="Number of jobs running at the same time (default: 4).",
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
    parser.add_argument("overrides", nargs="*", help="Hydra overrides.")
    args = parser.parse_args(argv)
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
    try:
        check_overrides(args.overrides)
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
            job_command(
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

    jobs_by_name = {job.name: job for job in jobs}
    failed = [r for r in results if r.status != "ok"]
    summary_jobs = []
    missing_friends = []
    for result in results:
        job = jobs_by_name[result.name]
        job_failed_files = failed_files(job, result, log_dir)
        missing_friends.extend(job_failed_files)
        summary_jobs.append(
            asdict(result)
            | {
                "input_files": job.input_files,
                "failed_files": job_failed_files,
            }
        )
    summary = {
        "overrides": args.overrides,
        "metadata_dir": str(metadata_dir),
        "output_dir": str(output_dir),
        "duration_s": duration,
        "n_files_per_job": args.n_files,
        "n_jobs": len(results),
        "n_failed": len(failed),
        "n_files": len(files),
        "n_failed_files": len(missing_friends),
        "jobs": summary_jobs,
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    logger.info(
        f"{len(results) - len(failed)}/{len(results)} jobs succeeded in "
        + f"{duration / 60:.1f} min; summary: {log_dir / 'summary.json'}"
    )
    if failed:
        failed_list = log_dir / "failed_files.txt"
        failed_list.write_text("".join(f"{f}\n" for f in missing_friends))
        logger.error(
            f"{len(failed)} jobs failed, {len(missing_friends)} files without "
            + f"friend ({failed_list}):\n"
            + "\n".join(f"  {r.name}: {r.log_file}" for r in failed)
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
