"""Tests for the sample metadata files."""

import gzip
import json

import pytest

from xyh_ml.friend_production import metadata as md
from xyh_ml.friend_production.dataclasses import Sample, Shift, Variable


def _record(channel="et", sample="DY", era="2024"):
    base = f"root://server//store/ntuples/{era}/{sample}/{channel}"
    return Sample(
        era=era,
        sample=sample,
        channel=channel,
        main_files=[f"{base}/f_0.root", f"{base}/f_1.root"],
        friend_files={
            "fr": [
                f"root://server//store/fr/{era}/{sample}/{channel}/f_0.root",
                f"root://server//store/fr/{era}/{sample}/{channel}/f_1.root",
            ]
        },
        shifts=[
            Shift(
                name="nominal",
                variables=[Variable(name="pt", column="pt")],
            )
        ],
    )


def test_read_samples_file(tmp_path):
    path = tmp_path / "samples.txt"
    path.write_text("A\n\n# comment\n /B/C/NANOAODSIM \n")

    assert md.read_samples_file(path) == ["A", "B_C"]


def test_metadata_path(tmp_path):
    assert md.metadata_path(tmp_path, "2024", "DY") == (
        tmp_path / "2024" / "DY.json.gz"
    )


def test_write_and_load_round_trip(tmp_path):
    records = [_record("et"), _record("mt")]

    path = md.write_sample_metadata(tmp_path, "2024", "DY", records)

    assert path == tmp_path / "2024" / "DY.json.gz"
    assert md.load_sample_metadata(tmp_path, "2024", "DY") == records
    # Gzipped JSON, as the old metadata file
    with gzip.open(path, "rt") as f:
        assert [r["channel"] for r in json.load(f)] == ["et", "mt"]


def test_write_replaces_existing_file(tmp_path):
    md.write_sample_metadata(tmp_path, "2024", "DY", [_record("et")])
    md.write_sample_metadata(tmp_path, "2024", "DY", [_record("mt")])

    (record,) = md.load_sample_metadata(tmp_path, "2024", "DY")
    assert record.channel == "mt"


def test_failed_write_leaves_no_file(tmp_path, monkeypatch):
    def failing_dump(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(md.json, "dump", failing_dump)

    with pytest.raises(OSError, match="disk full"):
        md.write_sample_metadata(tmp_path, "2024", "DY", [_record()])

    assert list((tmp_path / "2024").iterdir()) == []


def test_failed_write_keeps_previous_file(tmp_path, monkeypatch):
    md.write_sample_metadata(tmp_path, "2024", "DY", [_record("et")])

    def failing_dump(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(md.json, "dump", failing_dump)

    with pytest.raises(OSError):
        md.write_sample_metadata(tmp_path, "2024", "DY", [_record("mt")])

    monkeypatch.undo()
    (record,) = md.load_sample_metadata(tmp_path, "2024", "DY")
    assert record.channel == "et"
    assert [p.name for p in (tmp_path / "2024").iterdir()] == ["DY.json.gz"]


def test_load_missing_file_points_to_scraper(tmp_path):
    with pytest.raises(ValueError, match="scrape_metadata"):
        md.load_sample_metadata(tmp_path, "2024", "DY")


def test_available_samples(tmp_path):
    md.write_sample_metadata(tmp_path, "2024", "TT", [_record(sample="TT")])
    md.write_sample_metadata(tmp_path, "2024", "DY", [_record()])
    md.write_sample_metadata(tmp_path, "2025", "WJets", [_record()])
    # A temporary file of a write in progress and other files do not count
    (tmp_path / "2024" / ".ZZ.json.gz.123.456.tmp").touch()
    (tmp_path / "2024" / "notes.txt").touch()

    assert md.available_samples(tmp_path, ["2024", "2023"]) == {
        "2024": ["DY", "TT"],
        "2023": [],
    }


@pytest.mark.parametrize(
    "input_file",
    [
        "root://server//store/ntuples/2024/DY/et/f_0.root",
        "/store/ntuples/2024/DY/et/f_0.root",
        "2024/DY/et/f_0.root",
    ],
)
def test_parse_input_path(input_file):
    assert md.parse_input_path(input_file) == ("2024", "DY", "et")


@pytest.mark.parametrize(
    "input_file", ["f_0.root", "et/f_0.root", "DY/et/f_0.root", ""]
)
def test_parse_input_path_too_short(input_file):
    with pytest.raises(ValueError, match="does not have a path"):
        md.parse_input_path(input_file)


def test_select_input_files_loads_each_sample_once(tmp_path, monkeypatch):
    md.write_sample_metadata(
        tmp_path, "2024", "DY", [_record("et"), _record("mt")]
    )
    md.write_sample_metadata(tmp_path, "2024", "TT", [_record(sample="TT")])
    load = md.load_sample_metadata
    loaded = []

    def recording_load(metadata_dir, era, sample):
        loaded.append((era, sample))
        return load(metadata_dir, era, sample)

    monkeypatch.setattr(md, "load_sample_metadata", recording_load)

    selected = md.select_input_files(
        tmp_path,
        [
            "2024/DY/mt/f_1.root",
            "2024/TT/et/f_0.root",
            "2024/DY/et/f_0.root",
        ],
    )

    assert loaded == [("2024", "DY"), ("2024", "TT")]
    assert [(s.sample, s.channel) for s in selected] == [
        ("DY", "mt"),
        ("TT", "et"),
        ("DY", "et"),
    ]
    assert selected[0].main_files == [
        "root://server//store/ntuples/2024/DY/mt/f_1.root"
    ]
    assert selected[0].friend_files == {
        "fr": ["root://server//store/fr/2024/DY/mt/f_1.root"]
    }


def test_select_input_files_unknown_file(tmp_path):
    md.write_sample_metadata(tmp_path, "2024", "DY", [_record()])

    with pytest.raises(ValueError, match="f_9.root not found"):
        md.select_input_files(tmp_path, ["2024/DY/et/f_9.root"])
