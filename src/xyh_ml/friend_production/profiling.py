"""Time and memory profiling of the friend production.

The :class:`Profiler` accumulates the wall-clock time of the processing
stages and normalizes it to the number of events and to the number of rows
(events times parameter value combinations) evaluated. It also records the
size of the data frames and the memory usage of the process (resident set
size, RSS) and, if CUDA is used, of the GPU.
"""

import json
import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import torch

# Set up logging
logger = logging.getLogger(__name__)

_PROC_STATUS = Path("/proc/self/status")
_PROC_CLEAR_REFS = Path("/proc/self/clear_refs")


def _read_proc_status_bytes(key: str) -> int | None:
    """Value of ``key`` (e.g. ``VmRSS``, ``VmHWM``) in ``/proc/self/status``
    in bytes, or ``None`` if not available (non-Linux systems)."""
    try:
        for line in _PROC_STATUS.read_text().splitlines():
            if line.startswith(f"{key}:"):
                return int(line.split()[1]) * 1024  # value in kB
    except OSError:
        pass
    return None


def current_rss() -> int | None:
    """Current resident set size of the process in bytes."""
    return _read_proc_status_bytes("VmRSS")


def peak_rss() -> int | None:
    """Peak resident set size (high water mark) of the process in bytes."""
    return _read_proc_status_bytes("VmHWM")


def reset_peak_rss() -> bool:
    """Reset the peak resident set size of the process to its current value.

    Returns ``False`` if the kernel does not support it, in which case the
    peak refers to the whole lifetime of the process.
    """
    try:
        _PROC_CLEAR_REFS.write_text("5")
        return True
    except OSError:
        return False


def format_bytes(n: float) -> str:
    """Human-readable size, e.g. ``12.3 MiB``."""
    for unit in ["B", "KiB", "MiB", "GiB"]:
        if abs(n) < 1024 or unit == "GiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    raise AssertionError  # pragma: no cover


def data_frame_bytes(df: pd.DataFrame) -> int:
    """Memory used by a data frame including its index, in bytes."""
    return int(df.memory_usage(index=True, deep=True).sum())


class Profiler:
    """Accumulates time per processing stage and memory usage.

    Stages are timed with :meth:`measure`; the numbers of events and rows
    are added with :meth:`add_chunk`. Times of nested stages (e.g. ``predict``
    within ``evaluate``) are included in the outer stage.
    """

    def __init__(self) -> None:
        self.times: dict[str, float] = defaultdict(float)
        self.n_chunks = 0
        self.n_events = 0
        self.n_rows = 0
        self.data_frame_bytes: dict[str, list[int]] = defaultdict(list)
        self.rss_increase: dict[str, list[int]] = defaultdict(list)
        self._rss_start: int | None = None
        self._peak_rss_reset = False
        self._cuda = False

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        """Add the wall-clock time spent in the ``with`` block to
        ``stage``."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.times[stage] += time.perf_counter() - start

    @contextmanager
    def measure_memory(self, stage: str) -> Iterator[None]:
        """Like :meth:`measure`, and record the increase of the resident set
        size of the process during the ``with`` block."""
        rss_before = current_rss()
        with self.measure(stage):
            yield
        rss_after = current_rss()
        if rss_before is not None and rss_after is not None:
            self.rss_increase[stage].append(rss_after - rss_before)

    def add_chunk(self, n_events: int, n_rows: int) -> None:
        """Count a processed chunk with ``n_events`` events, evaluated in
        ``n_rows`` rows (events times parameter value combinations)."""
        self.n_chunks += 1
        self.n_events += n_events
        self.n_rows += n_rows

    def add_data_frame(self, name: str, df: pd.DataFrame) -> None:
        """Record the memory used by a data frame of one chunk."""
        self.data_frame_bytes[name].append(data_frame_bytes(df))

    def start_memory_tracking(self) -> None:
        """Reset the peak memory counters (process RSS and CUDA) so that the
        peaks reported by :meth:`summary` refer to the time after this
        call, e.g. after loading the models."""
        self._rss_start = current_rss()
        self._peak_rss_reset = reset_peak_rss()
        self._cuda = torch.cuda.is_available()
        if self._cuda:
            torch.cuda.reset_peak_memory_stats()

    def summary(self) -> dict[str, Any]:
        """Profile as dictionary (times in seconds, memory in bytes)."""
        stages = {
            stage: {
                "total_s": total,
                "per_event_s": total / self.n_events
                if self.n_events
                else None,
                "per_row_s": total / self.n_rows if self.n_rows else None,
            }
            for stage, total in self.times.items()
        }

        memory: dict[str, Any] = {
            f"{name}_data_frame_max_bytes": max(sizes)
            for name, sizes in self.data_frame_bytes.items()
        }
        memory.update(
            {
                f"{name}_data_frame_per_event_bytes": (
                    sum(sizes) / self.n_events if self.n_events else None
                )
                for name, sizes in self.data_frame_bytes.items()
            }
        )
        memory.update(
            {
                f"{stage}_rss_increase_max_bytes": max(increases)
                for stage, increases in self.rss_increase.items()
            }
        )
        memory["rss_start_bytes"] = self._rss_start
        memory["rss_peak_bytes"] = peak_rss()
        memory["rss_peak_since_start_of_tracking"] = self._peak_rss_reset
        if (
            self._rss_start is not None
            and memory["rss_peak_bytes"] is not None
        ):
            memory["rss_peak_increase_bytes"] = (
                memory["rss_peak_bytes"] - self._rss_start
            )
        if self._cuda:
            memory["cuda_peak_allocated_bytes"] = (
                torch.cuda.max_memory_allocated()
            )

        return {
            "n_chunks": self.n_chunks,
            "n_events": self.n_events,
            "n_rows": self.n_rows,
            "stages": stages,
            "memory": memory,
        }

    def log_summary(self) -> None:
        """Log the profile in a human-readable form."""
        summary = self.summary()
        lines = [
            f"Profile: {summary['n_events']} events, {summary['n_rows']} rows, "
            + f"{summary['n_chunks']} chunk(s)",
            f"  {'stage':<10} {'total [s]':>10} {'per event [ms]':>15} "
            + f"{'per row [us]':>13}",
        ]
        for stage, t in summary["stages"].items():
            per_event = (
                f"{t['per_event_s'] * 1e3:15.4f}"
                if t["per_event_s"] is not None
                else f"{'-':>15}"
            )
            per_row = (
                f"{t['per_row_s'] * 1e6:13.3f}"
                if t["per_row_s"] is not None
                else f"{'-':>13}"
            )
            lines.append(
                f"  {stage:<10} {t['total_s']:10.3f} {per_event} {per_row}"
            )
        for key, value in summary["memory"].items():
            if isinstance(value, bool) or value is None:
                lines.append(f"  {key}: {value}")
            else:
                lines.append(f"  {key}: {format_bytes(value)}")
        logger.info("\n".join(lines))

    def dump(self, output_file: Path) -> None:
        """Write the profile to a JSON file."""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(json.dumps(self.summary(), indent=2))
        logger.info(f"Wrote profile to {output_file}")
