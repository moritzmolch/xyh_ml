"""Produce ntuple friends for many samples in parallel.

Runs ``xyh_ml.cli.friends.produce`` as one subprocess per job, where a job is
one sample of one campaign (all its files and selected channels). Several
jobs run at the same time and are distributed over the available GPUs.

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

Jobs whose friend files all exist are skipped with ``--skip-existing``, so an
interrupted or partly failed run can be continued by repeating the command.
Every job writes its output to ``<log-dir>/logs/<job>.log``; the summary of
all jobs is written to ``<log-dir>/summary.json``.
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
from xyh_ml.cli.friends.produce import (
    campaign_name,
    default_output_file,
    friend_exists,
    resolve_output_dir,
    select_sample_files,
)
from xyh_ml.friend_production.dataclasses import Sample
from xyh_ml.io import load_json

logger = logging.getLogger(__name__)

PRODUCE_MODULE = "xyh_ml.cli.friends.produce"

# Overrides that are set per job by this script
RESERVED_OVERRIDES = [
    "input_file",
    "output_file",
    "sample",
    "campaign",
    "channel",
    "skip_existing",
    "profile_file",
    "hydra.run.dir",
]

# Environment variables limiting the CPU threads of a job
THREAD_VARIABLES = [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]


@dataclass
class Job:
    """One sample of one campaign, processed by one ``produce`` run."""

    sample: str
    era: str
    campaign: str
    channels: list[str]
    n_files: int
    output_files: list[str] = field(repr=False)

    @property
    def name(self) -> str:
        return f"{self.campaign}__{self.sample}"


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


def read_samples_file(path: Path) -> list[str]:
    """Sample names from a text file with one sample per line.

    Empty lines and lines starting with ``#`` are ignored. DAS dataset paths
    ``/<primary>/<processed>/<tier>`` are converted to the sample name
    ``<primary>_<processed>`` used for the ntuple directories.
    """
    names = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if m := re.fullmatch(r"/([^/]+)/([^/]+)/[^/]+", line):
            line = f"{m[1]}_{m[2]}"
        names.append(line)
    return names


def select_sample_names(
    available: list[str],
    patterns: list[str],
    names: list[str],
) -> tuple[list[str], list[str]]:
    """Sample names out of ``available`` that match one of the regular
    expressions ``patterns`` (whole name) or are in ``names``.

    Returns the selected names (in the order of ``available``) and the
    entries of ``names`` that are not available.
    """
    compiled = [re.compile(p) for p in patterns]
    wanted = set(names)
    selected = [
        s
        for s in available
        if s in wanted or any(p.fullmatch(s) for p in compiled)
    ]
    missing = sorted(wanted.difference(available))
    return selected, missing


def build_jobs(
    samples: list[Sample],
    sample_names: list[str],
    campaigns: list,
    output_dir: Path | str,
    campaign_names: list[str] | None = None,
    channels: list[str] | None = None,
) -> list[Job]:
    """One job per selected sample and campaign, largest jobs first.

    ``campaigns`` are the campaign definitions (name, era) of the config;
    ``campaign_names`` and ``channels`` restrict the jobs (default: all).
    """
    era_to_campaign = {c.era: c.name for c in campaigns}
    if campaign_names is not None:
        unknown = sorted(
            set(campaign_names).difference(era_to_campaign.values())
        )
        if unknown:
            msg = (
                f"Unknown campaigns {unknown}. Defined campaigns: "
                + f"{list(era_to_campaign.values())}"
            )
            raise ValueError(msg)

    wanted = set(sample_names)
    groups: dict[tuple[str, str], list[Sample]] = {}
    for s in samples:
        if s.sample not in wanted or s.era not in era_to_campaign:
            continue
        campaign = campaign_name(campaigns, s.era)
        if campaign_names is not None and campaign not in campaign_names:
            continue
        if channels is not None and s.channel not in channels:
            continue
        groups.setdefault((s.sample, s.era), []).append(s)

    jobs = []
    for (sample, era), group in groups.items():
        file_samples = select_sample_files(
            group,
            sample,
            eras=[era],
            channels=sorted({s.channel for s in group}),
        )
        jobs.append(
            Job(
                sample=sample,
                era=era,
                campaign=era_to_campaign[era],
                channels=sorted({s.channel for s in group}),
                n_files=len(file_samples),
                output_files=[
                    str(default_output_file(output_dir, f))
                    for f in file_samples
                ],
            )
        )

    # Start the largest jobs first, so that they do not delay the end
    return sorted(jobs, key=lambda j: (-j.n_files, j.name))


def job_command(
    job: Job,
    overrides: list[str],
    log_dir: Path,
    skip_existing: bool,
    profile: bool,
    python: str = sys.executable,
) -> list[str]:
    """Command running ``produce`` for ``job``."""
    command = [
        python,
        "-m",
        PRODUCE_MODULE,
        *overrides,
        f"sample='{job.sample}'",
        f"campaign={job.campaign}",
        f"channel=[{','.join(job.channels)}]",
        f"skip_existing={str(skip_existing).lower()}",
        f"hydra.run.dir={log_dir / 'hydra' / job.name}",
    ]
    if profile:
        command.append(
            f"profile_file={log_dir / 'profiles' / f'{job.name}.json'}"
        )
    return command


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
        help="Skip jobs whose friend files all exist, and files whose "
        + "friend file exists.",
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
    return args


def _compose(overrides: list[str]) -> DictConfig:
    with initialize_config_dir(
        str(PROJECT_DIR / "config"), version_base="1.3"
    ):
        return compose("produce", overrides=overrides)


def _project_path(path: str | Path) -> Path:
    """``path`` relative to the repo root, where the jobs run."""
    path = Path(path)
    return path if path.is_absolute() else PROJECT_DIR / path


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
    samples = [
        Sample.from_dict(s)
        for s in load_json(_project_path(cfg.metadata_file))
    ]

    names = [n for f in args.samples_file for n in read_samples_file(f)]
    available = sorted({s.sample for s in samples})
    sample_names, missing = select_sample_names(available, args.sample, names)
    if missing:
        logger.warning(
            f"{len(missing)} samples of the sample files are not in the "
            + f"metadata {cfg.metadata_file}:\n  "
            + "\n  ".join(missing)
        )

    # Local directory or XRootD URL (with nn_friend_tag)
    try:
        output_dir = resolve_output_dir(cfg)
    except ValueError as e:
        logger.error(e)
        return 2
    if not isinstance(output_dir, str):
        output_dir = _project_path(output_dir)

    jobs = build_jobs(
        samples,
        sample_names,
        cfg.campaigns,
        output_dir,
        campaign_names=args.campaign,
        channels=args.channel,
    )

    if args.skip_existing:
        done = [j for j in jobs if all(map(friend_exists, j.output_files))]
        if done:
            logger.info(f"Skipping {len(done)} jobs whose friends all exist")
        jobs = [j for j in jobs if j not in done]

    if not jobs:
        logger.warning("No jobs to run.")
        return 0

    log_dir = args.log_dir or (
        _project_path("output/produce_many")
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
    n_files = sum(j.n_files for j in jobs)
    logger.info(
        f"{len(jobs)} jobs ({n_files} files) of {len(sample_names)} samples, "
        + f"friends in {output_dir}"
    )

    if args.dry_run:
        for job, command in commands:
            print(f"{job.name}: {job.n_files} files, channels {job.channels}")
            print("  " + " ".join(command))
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

    failed = [r for r in results if r.status != "ok"]
    summary = {
        "overrides": args.overrides,
        "metadata_file": str(cfg.metadata_file),
        "output_dir": str(output_dir),
        "duration_s": duration,
        "n_jobs": len(results),
        "n_failed": len(failed),
        "jobs": [asdict(r) for r in results],
    }
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    logger.info(
        f"{len(results) - len(failed)}/{len(results)} jobs succeeded in "
        + f"{duration / 60:.1f} min; summary: {log_dir / 'summary.json'}"
    )
    if failed:
        logger.error(
            f"{len(failed)} jobs failed:\n"
            + "\n".join(f"  {r.name}: {r.log_file}" for r in failed)
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
