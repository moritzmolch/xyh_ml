import logging

import pytest
from hydra import compose, initialize_config_dir

from xyh_ml import PROJECT_DIR
from xyh_ml.config.dataclasses import Campaign, Channel, Friend
from xyh_ml.friend_production import scraper as ms
from xyh_ml.friend_production.dataclasses import Sample, Shift, Variable
from xyh_ml.friend_production.xrootd import DirectoryNotFoundError

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


# ----------------------------------------------------------- scrape_metadata


class Recorder:
    """Patch the get_* helpers used by scrape_metadata."""

    def __init__(
        self,
        monkeypatch,
        eras,
        samples,
        channels,
        files=None,
        friend_files=None,
    ):
        """``files`` maps a channel directory to its main files (default:
        ``m.root``). Friend files have the same file names as the main
        files, unless ``friend_files`` maps ``(friend, era, sample,
        channel)`` to other files, or to ``None`` for a missing directory.
        """
        self.friend_calls = []
        self.shift_calls = []
        files = files or {}
        friend_files = friend_files or {}

        def main_files(d):
            return files.get(d, [f"{d}/m.root"])

        monkeypatch.setattr(
            ms,
            "get_era_dirs",
            lambda fs, d: {e: f"{d}/CROWNRun/{e}" for e in eras},
        )
        monkeypatch.setattr(
            ms,
            "get_sample_dirs",
            lambda fs, d: {s: f"{d}/{s}" for s in samples},
        )
        monkeypatch.setattr(
            ms,
            "get_channel_dirs",
            lambda fs, d: {c: f"{d}/{c}" for c in channels},
        )
        monkeypatch.setattr(ms, "get_files", lambda fs, d: main_files(d))

        def get_friend_files(fs, base, tag, friend, era, sample, channel):
            self.friend_calls.append((base, tag, friend, era, sample, channel))
            key = (friend, era, sample, channel)
            if key in friend_files:
                if friend_files[key] is None:
                    raise DirectoryNotFoundError(f"no such dir for {key}")
                return friend_files[key]
            main_dir = f"{base}/{tag}/CROWNRun/{era}/{sample}/{channel}"
            friend_dir = f"{base}/{tag}/CROWNFriend/{friend}/{era}/{sample}"
            return [
                f.replace(main_dir, f"{friend_dir}/{channel}")
                for f in main_files(main_dir)
            ]

        monkeypatch.setattr(ms, "get_friend_files", get_friend_files)

        def get_shifts(main, friend):
            self.shift_calls.append((main, friend))
            return []

        monkeypatch.setattr(ms, "get_shifts", get_shifts)


@pytest.fixture
def recorder(monkeypatch):
    def factory(eras=("2018",), samples=("A",), channels=("et",), **kw):
        return Recorder(monkeypatch, eras, samples, channels, **kw)

    return factory


CAMPAIGNS = [
    Campaign(name="c18", era="2018"),
    Campaign(name="c22", era="2022"),
]


CHANNELS = [Channel(name="et"), Channel(name="mt"), Channel(name="tt")]


def run(friends=(), campaigns=CAMPAIGNS, channels=CHANNELS, samples=None):
    return ms.scrape_metadata(
        None, "/base", "tag", list(friends), campaigns, channels, samples
    )


def test_scrape_metadata_builds_sample_object(recorder):
    rec = recorder(eras=["2018"], samples=["A"], channels=["et"])
    result = run(friends=[Friend(name="fr")])
    assert len(result) == 1
    (sample,) = result
    assert isinstance(sample, Sample)
    assert sample.era == "2018"
    assert sample.sample == "A"
    assert sample.channel == "et"
    assert sample.main_files == ["/base/tag/CROWNRun/2018/A/et/m.root"]
    assert sample.friend_files == {
        "fr": ["/base/tag/CROWNFriend/fr/2018/A/et/m.root"]
    }
    assert rec.friend_calls == [("/base", "tag", "fr", "2018", "A", "et")]
    assert sample.shifts == []
    assert rec.shift_calls == [(sample.main_files, sample.friend_files)]


def test_scrape_metadata_covers_all_eras_and_samples(recorder):
    recorder(eras=["2018", "2022"], samples=["A", "B"], channels=["et"])
    result = run()
    assert [(s.era, s.sample) for s in result] == [
        ("2018", "A"),
        ("2018", "B"),
        ("2022", "A"),
        ("2022", "B"),
    ]


def test_scrape_metadata_without_friends_has_empty_friend_files(recorder):
    recorder()
    assert run()[0].friend_files == {}


def test_scrape_metadata_with_real_dict_returning_helpers(
    monkeypatch, fake_fs_factory
):
    fs = fake_fs_factory(
        {
            "/base/tag/CROWNRun": ["2018"],
            "/base/tag/CROWNRun/2018": ["A"],
            "/base/tag/CROWNRun/2018/A": ["et"],
            "/base/tag/CROWNRun/2018/A/et": ["x.root"],
        }
    )
    monkeypatch.setattr(ms, "get_shifts", lambda m, f: [])
    result = ms.scrape_metadata(fs, "/base", "tag", [], CAMPAIGNS, CHANNELS)
    assert [s.sample for s in result] == ["A"]
    assert result[0].main_files == [
        "root://server.example:1094//base/tag/CROWNRun/2018/A/et/x.root"
    ]


def test_scrape_metadata_era_not_in_campaigns_is_skipped(recorder, caplog):
    recorder(eras=["2018", "2022"])
    with caplog.at_level(logging.WARNING, logger=ms.logger.name):
        result = run(campaigns=[CAMPAIGNS[0]])
    assert [s.era for s in result] == ["2018"]
    assert "Folder '2022' ignored" in caplog.text


def test_scrape_metadata_channel_not_in_channels_is_skipped(recorder, caplog):
    rec = recorder(samples=["A"], channels=["em", "et", "mt"])
    with caplog.at_level(logging.INFO, logger=ms.logger.name):
        result = run(
            friends=[Friend(name="fr")],
            channels=[Channel(name="et"), Channel(name="mt")],
        )
    assert [s.channel for s in result] == ["et", "mt"]
    assert "Skipping channel em of sample A (2018)" in caplog.text
    # Neither files nor shifts are looked up for the skipped channel
    assert [c[-1] for c in rec.friend_calls] == ["et", "mt"]
    assert len(rec.shift_calls) == 2


def test_scrape_metadata_without_channels_is_empty(recorder):
    recorder(channels=["et", "mt"])
    assert run(channels=[]) == []


def test_scrape_metadata_missing_friend_skips_sample(recorder, caplog):
    rec = recorder(
        samples=["A", "B"],
        channels=["et", "mt"],
        friend_files={("fr2", "2018", "B", "mt"): None},
    )
    with caplog.at_level(logging.WARNING, logger=ms.logger.name):
        result = run(friends=[Friend(name="fr1"), Friend(name="fr2")])

    assert [(s.sample, s.channel) for s in result] == [
        ("A", "et"),
        ("A", "mt"),
        ("B", "et"),
    ]
    assert "Skipping sample B (2018, mt): fr2 is missing." in caplog.text
    assert "Skipped 1 samples" in caplog.text
    assert "2018 B mt: fr2 is missing" in caplog.text
    # The shifts are not read for the skipped sample
    assert len(rec.shift_calls) == 3


def test_scrape_metadata_reports_all_missing_friends_of_a_sample(
    recorder, caplog
):
    recorder(
        friend_files={
            ("fr1", "2018", "A", "et"): None,
            ("fr2", "2018", "A", "et"): None,
        }
    )
    with caplog.at_level(logging.WARNING, logger=ms.logger.name):
        result = run(friends=[Friend(name="fr1"), Friend(name="fr2")])

    assert result == []
    assert "fr1 is missing; fr2 is missing" in caplog.text


def test_scrape_metadata_excluded_friend_is_not_required(recorder, caplog):
    rec = recorder(friend_files={("fr", "2018", "A", "et"): None})
    with caplog.at_level(logging.WARNING, logger=ms.logger.name):
        (result,) = run(friends=[Friend(name="fr", exclude_channels=["et"])])

    assert result.friend_files == {}
    assert rec.friend_calls == []
    assert "Skipping sample" not in caplog.text


@pytest.mark.parametrize(
    "friend_names",
    [
        ["1.root"],  # missing file
        ["1.root", "2.root", "3.root"],  # additional file
        ["1.root", "3.root"],  # other file
    ],
)
def test_scrape_metadata_incomplete_friend_skips_sample(
    recorder, caplog, friend_names
):
    main_dir = "/base/tag/CROWNRun/2018/A/et"
    recorder(
        files={main_dir: [f"{main_dir}/1.root", f"{main_dir}/2.root"]},
        friend_files={
            ("fr", "2018", "A", "et"): [f"fr/{n}" for n in friend_names]
        },
    )
    with caplog.at_level(logging.WARNING, logger=ms.logger.name):
        result = run(friends=[Friend(name="fr")])

    assert result == []
    assert (
        f"fr has {len(friend_names)} files with names different from the 2 "
        + "main files"
    ) in caplog.text


def test_scrape_metadata_other_friend_listing_error_raises(
    recorder, monkeypatch
):
    recorder()

    def get_friend_files(*args):
        raise RuntimeError("not authorized")

    monkeypatch.setattr(ms, "get_friend_files", get_friend_files)
    with pytest.raises(RuntimeError, match="not authorized"):
        run(friends=[Friend(name="fr")])


def test_scrape_metadata_missing_friend_directory_on_file_system(
    monkeypatch, fake_fs_factory, caplog
):
    """With the real XRootD helpers, a missing friend directory skips the
    sample instead of stopping the scraping."""
    fs = fake_fs_factory(
        {
            "/base/tag/CROWNRun": ["2018"],
            "/base/tag/CROWNRun/2018": ["A", "B"],
            "/base/tag/CROWNRun/2018/A": ["et"],
            "/base/tag/CROWNRun/2018/A/et": ["1.root"],
            "/base/tag/CROWNRun/2018/B": ["et"],
            "/base/tag/CROWNRun/2018/B/et": ["1.root"],
            "/base/tag/CROWNFriend/fr/2018/A/et": ["1.root", "map.json"],
        }
    )
    monkeypatch.setattr(ms, "get_shifts", lambda m, f: [])
    with caplog.at_level(logging.WARNING, logger=ms.logger.name):
        result = ms.scrape_metadata(
            fs, "/base", "tag", [Friend(name="fr")], CAMPAIGNS, CHANNELS
        )

    assert [s.sample for s in result] == ["A"]
    assert result[0].friend_files == {
        "fr": [
            "root://server.example:1094//base/tag/CROWNFriend/fr/2018/A/et/1.root"
        ]
    }
    assert "Skipping sample B (2018, et): fr is missing." in caplog.text


def test_scrape_metadata_sample_filter(recorder):
    recorder(samples=["A", "B"])
    result = run(samples=["B"])
    assert [s.sample for s in result] == ["B"]


def test_scrape_metadata_multiple_channels_keep_sample_name(recorder):
    recorder(samples=["A"], channels=["et", "mt"])
    result = run()
    assert [(s.sample, s.channel) for s in result] == [
        ("A", "et"),
        ("A", "mt"),
    ]


@pytest.mark.parametrize(
    "friend, channel, expected",
    [
        (Friend(name="fr", include_channels=["et"]), "et", True),
        (Friend(name="fr", include_channels=["et"]), "mt", False),
        (Friend(name="fr", exclude_channels=["et"]), "et", False),
        (Friend(name="fr", exclude_channels=["et"]), "mt", True),
        (Friend(name="fr"), "tt", True),
    ],
    ids=["incl-hit", "incl-miss", "excl-hit", "excl-miss", "no-filter"],
)
def test_scrape_metadata_friend_channel_filter(
    recorder, friend, channel, expected
):
    recorder(channels=[channel])
    (result,) = run(friends=[friend])
    assert ("fr" in result.friend_files) is expected


def test_scrape_metadata_friend_without_campaign_filter_is_used(recorder):
    recorder()
    (result,) = run(friends=[Friend(name="fr")])
    assert "fr" in result.friend_files


@pytest.mark.parametrize(
    "friend, expected",
    [
        (Friend(name="fr", include_campaigns=["c18"]), True),
        (Friend(name="fr", include_campaigns=["c22"]), False),
        (Friend(name="fr", exclude_campaigns=["c18"]), False),
        (Friend(name="fr", exclude_campaigns=["c22"]), True),
    ],
    ids=["incl-hit", "incl-miss", "excl-hit", "excl-miss"],
)
def test_scrape_metadata_friend_campaign_filter(recorder, friend, expected):
    recorder(eras=["2018"])
    (result,) = run(friends=[friend])
    assert ("fr" in result.friend_files) is expected


@pytest.mark.parametrize(
    "sample, expected", [("TT_2018", False), ("DY_2018", True)]
)
def test_scrape_metadata_friend_exclude_samples_regex(
    recorder, sample, expected
):
    recorder(samples=[sample])
    (result,) = run(friends=[Friend(name="fr", exclude_samples=["^TT"])])
    assert ("fr" in result.friend_files) is expected


@pytest.mark.parametrize(
    "sample, expected", [("TT_2018", True), ("DY_2018", False)]
)
def test_scrape_metadata_friend_include_samples_regex(
    recorder, sample, expected
):
    recorder(samples=[sample])
    (result,) = run(friends=[Friend(name="fr", include_samples=["^TT"])])
    assert ("fr" in result.friend_files) is expected


def test_scrape_metadata_config_has_channels():
    with initialize_config_dir(
        str(PROJECT_DIR / "config"), version_base="1.3"
    ):
        cfg = compose("scrape_metadata")

    assert [c.name for c in cfg.channels] == ["et", "mt", "tt"]
