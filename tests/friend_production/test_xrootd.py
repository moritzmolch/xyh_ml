import logging

import pytest
from XRootD.client.flags import DirListFlags

from xyh_ml.friend_production.xrootd import (
    get_channel_dirs,
    get_era_dirs,
    get_files,
    get_friend_files,
    get_sample_dirs,
    xrd_ls,
)


def test_xrd_ls_returns_names(fake_fs_factory):
    fs = fake_fs_factory({"/d": ["a", "b"]})
    assert xrd_ls(fs, "/d") == ["a", "b"]


def test_xrd_ls_defaults_to_none_flag(fake_fs_factory):
    fs = fake_fs_factory({"/d": []})
    xrd_ls(fs, "/d")
    assert fs.calls == [("/d", DirListFlags.NONE)]


def test_xrd_ls_passes_custom_flags(fake_fs_factory):
    fs = fake_fs_factory({"/d": []})
    xrd_ls(fs, "/d", flags=DirListFlags.STAT)
    assert fs.calls == [("/d", DirListFlags.STAT)]


def test_xrd_ls_empty_directory(fake_fs_factory):
    assert xrd_ls(fake_fs_factory({"/d": []}), "/d") == []


def test_xrd_ls_failure_raises_with_message(fake_fs_factory, caplog):
    fs = fake_fs_factory({})
    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match="no such dir /missing"):
            xrd_ls(fs, "/missing")
    assert "Failed to list directory /missing" in caplog.text


def test_get_era_dirs(fake_fs_factory):
    fs = fake_fs_factory({"/b/CROWNRun": ["2018", "2022"]})
    assert get_era_dirs(fs, "/b") == {
        "2018": "/b/CROWNRun/2018",
        "2022": "/b/CROWNRun/2022",
    }


def test_get_sample_dirs(fake_fs_factory):
    fs = fake_fs_factory({"/e": ["A", "B"]})
    assert get_sample_dirs(fs, "/e") == {"A": "/e/A", "B": "/e/B"}


def test_get_channel_dirs(fake_fs_factory):
    fs = fake_fs_factory({"/s": ["et", "mt"]})
    assert get_channel_dirs(fs, "/s") == {"et": "/s/et", "mt": "/s/mt"}


def test_get_files_filters_root_and_sorts(fake_fs_factory):
    fs = fake_fs_factory({"/c": ["b.root", "log.txt", "a.root", "c.root.bak"]})
    assert get_files(fs, "/c") == ["/c/a.root", "/c/b.root"]


def test_get_files_empty(fake_fs_factory):
    assert get_files(fake_fs_factory({"/c": ["x.txt"]}), "/c") == []


def test_get_friend_files_absolute_base(fake_fs_factory):
    d = "/b/CROWNFriend/fr/2018/A/et"
    fs = fake_fs_factory({d: ["2.root", "1.root", "junk"]})
    assert get_friend_files(fs, "/b", "fr", "2018", "A", "et") == [
        f"{d}/1.root",
        f"{d}/2.root",
    ]
    assert [c[0] for c in fs.calls] == [d]


def test_get_friend_files_relative_base(fake_fs_factory):
    d = "b/CROWNFriend/fr/2018/A/et"
    fs = fake_fs_factory({d: ["1.root"]})
    assert get_friend_files(fs, "b", "fr", "2018", "A", "et") == [
        f"{d}/1.root"
    ]
