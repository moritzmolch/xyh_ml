import logging

import pytest
from hydra import compose, initialize_config_dir

from xyh_ml import PROJECT_DIR
from xyh_ml.cli.friends import scrape_metadata as cli
from xyh_ml.config.dataclasses import Campaign, Friend
from xyh_ml.friend_production import scraper as ms
from xyh_ml.friend_production.dataclasses import Sample, Shift, Variable
from xyh_ml.friend_production.metadata import (
    load_sample_metadata,
    metadata_path,
)

# ---------------------------------------------------------------- get_shifts


def _names(shifts):
    return sorted(s.name for s in shifts)


def _columns(shifts):
    return sorted(v.column for s in shifts for v in s.variables)


def test_get_shifts_all_files_invalid_returns_empty(make_root_file):
    files = [
        make_root_file("a.root", kind="no_tree"),
        make_root_file("b.root", kind="empty"),
    ]
    assert ms.get_shifts(files, {}) == []


def test_get_shifts_no_files_returns_empty():
    assert ms.get_shifts([], {}) == []


def test_get_shifts_skips_invalid_and_uses_first_valid(make_root_file):
    files = [
        make_root_file("a.root", kind="no_tree"),
        make_root_file("b.root", kind="empty"),
        make_root_file("c.root", branches=["x__aUp"]),
        make_root_file("d.root", branches=["x__bUp"]),
    ]
    shifts = ms.get_shifts(files, {})
    assert _names(shifts) == ["aUp"]
    assert _columns(shifts) == ["x__aUp"]


def test_get_shifts_merges_friend_keys_at_same_index(make_root_file):
    mains = [
        make_root_file("m0.root", kind="empty"),
        make_root_file("m1.root", branches=["x__aUp"]),
    ]
    friends = {
        "f": [
            make_root_file("f0.root", branches=["y__wrongUp"]),
            make_root_file("f1.root", branches=["y__bDown"]),
        ]
    }
    shifts = ms.get_shifts(mains, friends)
    assert _columns(shifts) == ["x__aUp", "y__bDown"]


def test_get_shifts_skipped_friend_is_ignored_without_error(make_root_file):
    mains = [make_root_file("m.root", branches=["x__aUp"])]
    friends = {
        "empty": [make_root_file("e.root", kind="empty")],
        "notree": [make_root_file("n.root", kind="no_tree")],
        "ok": [make_root_file("o.root", branches=["y__bUp"])],
    }
    shifts = ms.get_shifts(mains, friends)
    assert _columns(shifts) == ["x__aUp", "y__bUp"]


def test_get_shifts_warns_on_overlapping_keys(make_root_file, caplog):
    mains = [make_root_file("m.root", branches=["x__aUp"])]
    friends = {"f": [make_root_file("f.root", branches=["x__aUp"])]}
    with caplog.at_level(logging.WARNING, logger=ms.logger.name):
        ms.get_shifts(mains, friends)
    assert "overlap" in caplog.text
    assert "x__aUp" in caplog.text


def test_get_shifts_no_overlap_warning_for_distinct_keys(
    make_root_file, caplog
):
    mains = [make_root_file("m.root", branches=["x__aUp"])]
    friends = {"f": [make_root_file("f.root", branches=["y__aUp"])]}
    with caplog.at_level(logging.WARNING, logger=ms.logger.name):
        ms.get_shifts(mains, friends)
    assert "overlap" not in caplog.text


def test_get_shifts_warns_on_multiple_double_underscores(
    make_root_file, caplog
):
    mains = [make_root_file("m.root", branches=["a__b__cUp"])]
    with caplog.at_level(logging.WARNING, logger=ms.logger.name):
        ms.get_shifts(mains, {})
    assert "more than one '__'" in caplog.text


def test_get_shifts_builds_shift_and_variable_objects(make_root_file):
    mains = [make_root_file("m.root", branches=["pt__jesUp"])]
    assert ms.get_shifts(mains, {}) == [
        Shift(
            name="jesUp",
            variables=[Variable(name="pt", column="pt__jesUp")],
        )
    ]


def test_get_shifts_groups_columns_by_shift(make_root_file):
    mains = [
        make_root_file(
            "m.root",
            branches=["pt__jesUp", "mass__jesUp", "pt__jesDown"],
        )
    ]
    shifts = {s.name: s for s in ms.get_shifts(mains, {})}
    assert set(shifts) == {"jesUp", "jesDown"}
    assert sorted(v.column for v in shifts["jesUp"].variables) == [
        "mass__jesUp",
        "pt__jesUp",
    ]
    assert [v.column for v in shifts["jesDown"].variables] == ["pt__jesDown"]


@pytest.mark.parametrize(
    "column, variable, shift",
    [
        ("pt__jesUp", "pt", "jesUp"),
        ("m_vis__tauEsDown", "m_vis", "tauEsDown"),
        ("pt_1__jesUp", "pt_1", "jesUp"),
        ("pt_1", "pt_1", "nominal"),
        ("pt_jesUp", "pt_jesUp", "nominal"),
        ("foo__barSideways", "foo__barSideways", "nominal"),
    ],
)
def test_get_shifts_parses_column_names(
    make_root_file, column, variable, shift
):
    mains = [make_root_file("m.root", branches=[column])]
    (result,) = ms.get_shifts(mains, {})
    assert result.name == shift
    assert result.variables == [Variable(name=variable, column=column)]


def test_get_shifts_groups_unshifted_columns_as_nominal(make_root_file):
    mains = [
        make_root_file("m.root", branches=["pt_1", "pt_1__jesUp", "m_vis"])
    ]
    shifts = {s.name: s for s in ms.get_shifts(mains, {})}
    assert set(shifts) == {"nominal", "jesUp"}
    assert sorted(v.name for v in shifts["nominal"].variables) == [
        "m_vis",
        "pt_1",
    ]
    assert [v.name for v in shifts["jesUp"].variables] == ["pt_1"]


# ------------------------------------------------------------ friend_applies

C18 = Campaign(name="c18", era="2018")
C22 = Campaign(name="c22", era="2022")
ROOT = "root://server.example:1094/"


@pytest.mark.parametrize(
    "friend, expected",
    [
        (Friend(name="fr"), True),
        (Friend(name="fr", include_channels=["et"]), True),
        (Friend(name="fr", include_channels=["mt"]), False),
        (Friend(name="fr", exclude_channels=["et"]), False),
        (Friend(name="fr", exclude_channels=["mt"]), True),
        (Friend(name="fr", include_campaigns=["c18"]), True),
        (Friend(name="fr", include_campaigns=["c22"]), False),
        (Friend(name="fr", exclude_campaigns=["c18"]), False),
        (Friend(name="fr", exclude_campaigns=["c22"]), True),
        (Friend(name="fr", include_samples=["^TT"]), True),
        (Friend(name="fr", include_samples=["^DY"]), False),
        (Friend(name="fr", exclude_samples=["^TT"]), False),
        (Friend(name="fr", exclude_samples=["^DY"]), True),
    ],
)
def test_friend_applies(friend, expected):
    assert ms.friend_applies(friend, "c18", "et", "TT_2018") is expected


# ------------------------------------------------------------- scrape_sample


def _main_dir(sample="A", channel="et", era="2018"):
    return f"/base/tag/CROWNRun/{era}/{sample}/{channel}"


def _friend_dir(friend="fr", sample="A", channel="et", era="2018"):
    return f"/base/tag/CROWNFriend/{friend}/{era}/{sample}/{channel}"


@pytest.fixture
def no_shifts(monkeypatch):
    monkeypatch.setattr(ms, "get_shifts", lambda main, friends: [])


def _scrape(fs, friends=(), campaign=C18, sample="A", channels=("et",)):
    return ms.scrape_sample(
        fs, "/base", "tag", list(friends), campaign, sample, list(channels)
    )


def test_scrape_sample_builds_records(fake_fs_factory, monkeypatch):
    shift_calls = []

    def get_shifts(main, friends):
        shift_calls.append((main, friends))
        return [Shift(name="nominal", variables=[])]

    monkeypatch.setattr(ms, "get_shifts", get_shifts)
    fs = fake_fs_factory(
        {
            _main_dir(channel="et"): ["1.root", "log.txt"],
            _main_dir(channel="mt"): ["2.root"],
            _friend_dir(channel="et"): ["1.root"],
            _friend_dir(channel="mt"): ["2.root"],
        }
    )

    et, mt = _scrape(fs, [Friend(name="fr")], channels=["et", "mt"])

    assert isinstance(et, Sample)
    assert (et.era, et.sample, et.channel) == ("2018", "A", "et")
    assert et.main_files == [f"{ROOT}{_main_dir()}/1.root"]
    assert et.friend_files == {"fr": [f"{ROOT}{_friend_dir()}/1.root"]}
    assert et.shifts == [Shift(name="nominal", variables=[])]
    assert mt.channel == "mt"
    assert shift_calls[0] == (et.main_files, et.friend_files)


def test_scrape_sample_lists_only_selected_directories(
    fake_fs_factory, no_shifts
):
    """No era, CROWNRun or sample directory and no unselected channel or
    friend directory is listed."""
    fs = fake_fs_factory(
        {
            _main_dir(channel="et"): ["1.root"],
            _main_dir(channel="mt"): ["1.root"],
            _main_dir(channel="em"): ["1.root"],
            _friend_dir(channel="et"): ["1.root"],
            _friend_dir(channel="mt"): ["1.root"],
        }
    )

    _scrape(
        fs,
        [Friend(name="fr"), Friend(name="other", include_samples=["^TT"])],
        channels=["et", "mt"],
    )

    assert sorted(path for path, _ in fs.calls) == sorted(
        [
            _main_dir(channel="et"),
            _main_dir(channel="mt"),
            _friend_dir(channel="et"),
            _friend_dir(channel="mt"),
        ]
    )


def test_scrape_sample_not_in_era(fake_fs_factory, no_shifts):
    fs = fake_fs_factory()

    with pytest.raises(ms.SampleNotFoundError, match="A has no channel"):
        _scrape(fs, channels=["et", "mt"])


def test_scrape_sample_missing_channel_is_incomplete(
    fake_fs_factory, no_shifts
):
    fs = fake_fs_factory({_main_dir(channel="et"): ["1.root"]})

    with pytest.raises(
        ms.IncompleteSampleError, match="channel mt is missing"
    ):
        _scrape(fs, channels=["et", "mt"])


def test_scrape_sample_channel_without_files_is_incomplete(
    fake_fs_factory, no_shifts
):
    fs = fake_fs_factory({_main_dir(): ["log.txt"]})

    with pytest.raises(ms.IncompleteSampleError, match="has no files"):
        _scrape(fs)


def test_scrape_sample_reports_all_missing_friends(fake_fs_factory, no_shifts):
    fs = fake_fs_factory({_main_dir(): ["1.root"]})

    with pytest.raises(
        ms.IncompleteSampleError,
        match="et: fr1 is missing; et: fr2 is missing",
    ):
        _scrape(fs, [Friend(name="fr1"), Friend(name="fr2")])


def test_scrape_sample_excluded_friend_is_not_required(
    fake_fs_factory, no_shifts
):
    fs = fake_fs_factory({_main_dir(): ["1.root"]})

    (record,) = _scrape(fs, [Friend(name="fr", exclude_channels=["et"])])

    assert record.friend_files == {}


@pytest.mark.parametrize(
    "friend_names",
    [
        ["1.root"],  # missing file
        ["1.root", "2.root", "3.root"],  # additional file
        ["1.root", "3.root"],  # other file
    ],
)
def test_scrape_sample_incomplete_friend(
    fake_fs_factory, no_shifts, friend_names
):
    fs = fake_fs_factory(
        {
            _main_dir(): ["1.root", "2.root"],
            _friend_dir(): friend_names,
        }
    )

    with pytest.raises(
        ms.IncompleteSampleError,
        match=f"fr has {len(friend_names)} files with names different from "
        + "the 2 main files",
    ):
        _scrape(fs, [Friend(name="fr")])


def test_scrape_sample_other_listing_error_raises(fake_fs_factory, no_shifts):
    fs = fake_fs_factory(
        {_main_dir(): ["1.root"]}, errors={_friend_dir(): 3010}
    )

    with pytest.raises(RuntimeError, match="Failed to list") as error:
        _scrape(fs, [Friend(name="fr")])

    assert not isinstance(error.value, ms.IncompleteSampleError)


# ------------------------------------------------------------ scrape_samples


def _scrape_samples(fs, metadata_dir, names, campaigns=(C18,), **kwargs):
    return ms.scrape_samples(
        fs,
        "/base",
        "tag",
        kwargs.pop("friends", []),
        list(campaigns),
        kwargs.pop("channels", ["et"]),
        names,
        metadata_dir,
        **kwargs,
    )


def test_scrape_samples_writes_one_file_per_sample(
    fake_fs_factory, no_shifts, tmp_path
):
    fs = fake_fs_factory(
        {
            _main_dir("A", "et"): ["1.root"],
            _main_dir("A", "mt"): ["1.root"],
            _main_dir("B", "et"): ["1.root"],
            _main_dir("B", "mt"): ["1.root"],
        }
    )

    result = _scrape_samples(fs, tmp_path, ["A", "B"], channels=["et", "mt"])

    assert sorted(result.written) == [("2018", "A"), ("2018", "B")]
    assert sorted(
        p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")
    ) == ["2018", "2018/A.json.gz", "2018/B.json.gz"]
    records = load_sample_metadata(tmp_path, "2018", "A")
    assert [(r.sample, r.channel) for r in records] == [
        ("A", "et"),
        ("A", "mt"),
    ]


def test_scrape_samples_skips_existing_without_server_access(
    fake_fs_factory, no_shifts, tmp_path
):
    existing = metadata_path(tmp_path, "2018", "A")
    existing.parent.mkdir()
    existing.write_bytes(b"existing")
    fs = fake_fs_factory()

    result = _scrape_samples(fs, tmp_path, ["A"])

    assert result.skipped == [("2018", "A")]
    assert result.written == []
    assert result.not_found == []
    assert fs.calls == []
    assert existing.read_bytes() == b"existing"


def test_scrape_samples_overwrite(fake_fs_factory, no_shifts, tmp_path):
    existing = metadata_path(tmp_path, "2018", "A")
    existing.parent.mkdir()
    existing.write_bytes(b"existing")
    fs = fake_fs_factory({_main_dir(): ["1.root"]})

    result = _scrape_samples(fs, tmp_path, ["A"], overwrite=True)

    assert result.written == [("2018", "A")]
    assert load_sample_metadata(tmp_path, "2018", "A")[0].sample == "A"


def test_scrape_samples_one_bad_channel_writes_no_file(
    fake_fs_factory, no_shifts, tmp_path
):
    """A partial file would block the missing channel in later runs."""
    fs = fake_fs_factory(
        {
            _main_dir(channel="et"): ["1.root"],
            _friend_dir(channel="et"): ["1.root"],
            _main_dir(channel="mt"): ["1.root"],
        }
    )

    result = _scrape_samples(
        fs, tmp_path, ["A"], friends=[Friend(name="fr")], channels=["et", "mt"]
    )

    assert result.written == []
    assert result.incomplete == [("2018", "A", "mt: fr is missing")]
    assert not metadata_path(tmp_path, "2018", "A").exists()

    # A later run finds the friend
    fs.listings[_friend_dir(channel="mt")] = ["1.root"]
    result = _scrape_samples(
        fs, tmp_path, ["A"], friends=[Friend(name="fr")], channels=["et", "mt"]
    )
    assert result.written == [("2018", "A")]


def test_scrape_samples_failure_does_not_stop_others(
    fake_fs_factory, no_shifts, tmp_path, caplog
):
    fs = fake_fs_factory(
        {_main_dir("B"): ["1.root"]},
        errors={_main_dir("A"): 3010},
    )

    with caplog.at_level(logging.ERROR, logger=ms.logger.name):
        result = _scrape_samples(fs, tmp_path, ["A", "B"], n_parallel=1)

    assert result.failed == [("2018", "A")]
    assert result.written == [("2018", "B")]
    assert "Failed to scrape A (2018)" in caplog.text
    assert not metadata_path(tmp_path, "2018", "A").exists()


def test_scrape_samples_reports_samples_not_found(
    fake_fs_factory, no_shifts, tmp_path
):
    """A sample only has to exist in one of the selected eras."""
    fs = fake_fs_factory({_main_dir("A", era="2022"): ["1.root"]})

    result = _scrape_samples(fs, tmp_path, ["A", "Z"], campaigns=[C18, C22])

    assert result.written == [("2022", "A")]
    assert result.not_found == ["Z"]
    assert sorted(p.name for p in tmp_path.rglob("*.json.gz")) == ["A.json.gz"]


def test_scrape_samples_duplicate_names(fake_fs_factory, no_shifts, tmp_path):
    fs = fake_fs_factory({_main_dir(): ["1.root"]})

    result = _scrape_samples(fs, tmp_path, ["A", "A"])

    assert result.written == [("2018", "A")]


def test_scrape_samples_invalid_number_of_threads(tmp_path):
    with pytest.raises(ValueError, match="n_parallel"):
        _scrape_samples(None, tmp_path, ["A"], n_parallel=0)


# ----------------------------------------------------------------------- CLI


@pytest.fixture
def cli_args(tmp_path):
    samples_file = tmp_path / "samples.txt"
    samples_file.write_text("A\n/B/C/NANOAODSIM\n")
    return [
        "--samples-file",
        str(samples_file),
        "--campaign",
        "2024_nano_v15",
        f"metadata_dir={tmp_path / 'metadata'}",
        "ntuple.base_dir=/base",
        "ntuple.tag=tag",
        "ntuple.friends=[]",
    ]


def test_cli_requires_samples_file():
    with pytest.raises(SystemExit):
        cli.main(["--dry-run"])


def test_cli_dry_run(cli_args, tmp_path, capsys, monkeypatch):
    def no_server(server):
        raise AssertionError("The server must not be accessed")

    monkeypatch.setattr(cli, "FileSystem", no_server)
    existing = metadata_path(tmp_path / "metadata", "2024", "A")
    existing.parent.mkdir(parents=True)
    existing.touch()

    assert cli.main(["--dry-run", *cli_args]) == 0

    assert capsys.readouterr().out.splitlines() == ["2024 B_C"]


def test_cli_unknown_campaign(cli_args, caplog):
    assert cli.main(["--campaign", "2024", *cli_args]) == 2
    assert "Unknown campaigns" in caplog.text


def test_cli_scrapes_and_reports(
    cli_args, tmp_path, fake_fs_factory, no_shifts, monkeypatch, caplog
):
    channels = ["et", "mt", "tt"]
    listings = {
        _main_dir("A", c, era="2024"): ["1.root"] for c in channels
    } | {_main_dir("B_C", "et", era="2024"): ["1.root"]}
    fs = fake_fs_factory(listings)
    monkeypatch.setattr(cli, "FileSystem", lambda server: fs)

    with caplog.at_level(logging.INFO):
        assert cli.main(cli_args) == 0

    metadata_dir = tmp_path / "metadata"
    assert metadata_path(metadata_dir, "2024", "A").exists()
    assert not metadata_path(metadata_dir, "2024", "B_C").exists()
    assert "1 written, 0 skipped (exist), 1 incomplete" in caplog.text
    assert "2024 B_C: channel mt is missing" in caplog.text


def test_cli_failure_gives_exit_code(
    cli_args, fake_fs_factory, no_shifts, monkeypatch, caplog
):
    fs = fake_fs_factory(errors={_main_dir("A", "et", era="2024"): 3010})
    monkeypatch.setattr(cli, "FileSystem", lambda server: fs)

    assert cli.main(cli_args) == 1
    assert "1 samples failed" in caplog.text


def test_scrape_metadata_config_has_channels():
    with initialize_config_dir(
        str(PROJECT_DIR / "config"), version_base="1.3"
    ):
        cfg = compose("scrape_metadata")

    assert [c.name for c in cfg.channels] == ["et", "mt", "tt"]
    assert cfg.metadata_dir.endswith("/metadata")
    assert "samples" not in cfg.ntuple
