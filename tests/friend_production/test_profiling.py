"""Tests for the profiling of the friend production."""

import json
import logging

import numpy as np
import pandas as pd
import pytest

from xyh_ml.friend_production import profiling
from xyh_ml.friend_production.profiling import Profiler


@pytest.fixture
def clock(monkeypatch):
    """Replace time.perf_counter with a clock returning the given ticks."""

    def factory(*ticks):
        it = iter(ticks)
        monkeypatch.setattr(profiling.time, "perf_counter", lambda: next(it))

    return factory


@pytest.fixture
def proc_status(monkeypatch, tmp_path):
    """Fake /proc/self/status with given VmRSS and VmHWM values in kB."""

    def factory(rss_kb, hwm_kb):
        path = tmp_path / "status"
        path.write_text(
            f"Name:\tpython\nVmHWM:\t  {hwm_kb} kB\nVmRSS:\t  {rss_kb} kB\n"
        )
        monkeypatch.setattr(profiling, "_PROC_STATUS", path)
        return path

    return factory


def test_measure_accumulates_time(clock):
    clock(0.0, 1.5, 10.0, 10.25)
    profiler = Profiler()

    with profiler.measure("load"):
        pass
    with profiler.measure("load"):
        pass

    assert profiler.times["load"] == pytest.approx(1.75)


def test_measure_records_time_on_exception(clock):
    clock(0.0, 2.0)
    profiler = Profiler()

    with pytest.raises(RuntimeError), profiler.measure("evaluate"):
        raise RuntimeError

    assert profiler.times["evaluate"] == pytest.approx(2.0)


def test_summary_per_event_and_per_row(clock):
    clock(0.0, 2.0, 2.0, 3.0)
    profiler = Profiler()
    with profiler.measure("evaluate"):
        pass
    with profiler.measure("load"):
        pass
    profiler.add_chunk(n_events=10, n_rows=100)
    profiler.add_chunk(n_events=10, n_rows=100)

    summary = profiler.summary()

    assert (summary["n_chunks"], summary["n_events"], summary["n_rows"]) == (
        2,
        20,
        200,
    )
    assert summary["stages"]["evaluate"] == {
        "total_s": pytest.approx(2.0),
        "per_event_s": pytest.approx(0.1),
        "per_row_s": pytest.approx(0.01),
    }
    assert summary["stages"]["load"]["per_event_s"] == pytest.approx(0.05)


def test_summary_without_events(clock):
    clock(0.0, 1.0)
    profiler = Profiler()
    with profiler.measure("total"):
        pass

    stage = profiler.summary()["stages"]["total"]

    assert stage["per_event_s"] is None
    assert stage["per_row_s"] is None


def test_data_frame_sizes():
    profiler = Profiler()
    small = pd.DataFrame({"x": np.zeros(10)})
    large = pd.DataFrame({"x": np.zeros(30)})

    profiler.add_data_frame("loaded", small)
    profiler.add_data_frame("loaded", large)
    profiler.add_chunk(n_events=40, n_rows=40)

    memory = profiler.summary()["memory"]
    assert memory["loaded_data_frame_max_bytes"] == (
        profiling.data_frame_bytes(large)
    )
    assert memory["loaded_data_frame_per_event_bytes"] == pytest.approx(
        (profiling.data_frame_bytes(small) + profiling.data_frame_bytes(large))
        / 40
    )


def test_data_frame_bytes_includes_index_and_objects():
    df = pd.DataFrame(
        {"x": np.zeros(4), "s": ["a" * 100] * 4},
        index=pd.MultiIndex.from_arrays(
            [np.zeros(4, dtype=int), np.arange(4)], names=["chunk", "entry"]
        ),
    )

    assert profiling.data_frame_bytes(df) > 4 * 8 + 4 * 100


def test_read_proc_status(proc_status):
    proc_status(rss_kb=2048, hwm_kb=4096)

    assert profiling.current_rss() == 2048 * 1024
    assert profiling.peak_rss() == 4096 * 1024


def test_read_proc_status_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(profiling, "_PROC_STATUS", tmp_path / "missing")

    assert profiling.current_rss() is None
    assert profiling.peak_rss() is None


def test_reset_peak_rss_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(
        profiling, "_PROC_CLEAR_REFS", tmp_path / "missing" / "clear_refs"
    )

    assert profiling.reset_peak_rss() is False


def test_measure_memory_records_rss_increase(monkeypatch, clock):
    clock(0.0, 1.0)
    rss = iter([100, 350])
    monkeypatch.setattr(profiling, "current_rss", lambda: next(rss))
    profiler = Profiler()

    with profiler.measure_memory("load"):
        pass

    assert profiler.times["load"] == pytest.approx(1.0)
    assert profiler.summary()["memory"]["load_rss_increase_max_bytes"] == 250


def test_peak_memory_since_start_of_tracking(monkeypatch, proc_status):
    monkeypatch.setattr(profiling, "reset_peak_rss", lambda: True)
    monkeypatch.setattr(profiling.torch.cuda, "is_available", lambda: False)
    proc_status(rss_kb=1000, hwm_kb=1000)
    profiler = Profiler()
    profiler.start_memory_tracking()

    proc_status(rss_kb=1200, hwm_kb=3000)
    memory = profiler.summary()["memory"]

    assert memory["rss_start_bytes"] == 1000 * 1024
    assert memory["rss_peak_bytes"] == 3000 * 1024
    assert memory["rss_peak_increase_bytes"] == 2000 * 1024
    assert memory["rss_peak_since_start_of_tracking"] is True
    assert "cuda_peak_allocated_bytes" not in memory


def test_start_memory_tracking_on_this_system():
    profiler = Profiler()
    profiler.start_memory_tracking()
    _ = np.ones(2**20)  # 8 MiB

    memory = profiler.summary()["memory"]

    # Available on Linux, which the friend production runs on
    assert memory["rss_peak_bytes"] >= memory["rss_start_bytes"] > 0


def test_dump_and_log(tmp_path, clock, caplog):
    clock(0.0, 0.5)
    profiler = Profiler()
    with profiler.measure("predict"):
        pass
    profiler.add_chunk(n_events=5, n_rows=50)
    profiler.add_data_frame("loaded", pd.DataFrame({"x": np.zeros(5)}))

    output_file = tmp_path / "profile" / "profile.json"
    profiler.dump(output_file)
    with caplog.at_level(logging.INFO, logger=profiling.__name__):
        profiler.log_summary()

    assert json.loads(output_file.read_text()) == json.loads(
        json.dumps(profiler.summary())
    )
    assert "5 events, 50 rows" in caplog.text
    assert "predict" in caplog.text
    assert "loaded_data_frame_max_bytes" in caplog.text


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, "0 B"),
        (512, "512 B"),
        (1536, "1.5 KiB"),
        (15.3 * 2**20, "15.3 MiB"),
        (2.5 * 2**30, "2.5 GiB"),
        (3000 * 2**30, "3000.0 GiB"),
        (-2048, "-2.0 KiB"),
    ],
)
def test_format_bytes(n, expected):
    assert profiling.format_bytes(n) == expected
