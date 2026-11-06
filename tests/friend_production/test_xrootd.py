import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from XRootD.client.flags import DirListFlags

from xyh_ml.friend_production import xrootd
from xyh_ml.friend_production.xrootd import (
    XRD_NOT_FOUND,
    DirectoryNotFoundError,
    get_files,
    get_friend_files,
    is_xrootd_url,
    split_xrootd_url,
    xrd_exists,
    xrd_ls,
    xrd_upload,
    xrootd_url,
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
    # 3010: not authorized
    fs = fake_fs_factory({}, errors={"/denied": 3010})
    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match="no such dir /denied") as e:
            xrd_ls(fs, "/denied")
    assert not isinstance(e.value, DirectoryNotFoundError)
    assert "Failed to list directory /denied" in caplog.text


def test_xrd_ls_missing_directory_raises_not_found(fake_fs_factory, caplog):
    fs = fake_fs_factory({})
    with caplog.at_level(logging.ERROR):
        with pytest.raises(DirectoryNotFoundError, match="/missing"):
            xrd_ls(fs, "/missing")
    # Left to the caller, which may expect the directory to be missing
    assert caplog.text == ""


def test_directory_not_found_error_is_runtime_error():
    assert issubclass(DirectoryNotFoundError, RuntimeError)


def test_get_files_filters_root_and_sorts(fake_fs_factory):
    fs = fake_fs_factory({"/c": ["b.root", "log.txt", "a.root", "c.root.bak"]})
    assert get_files(fs, "/c") == [
        "root://server.example:1094//c/a.root",
        "root://server.example:1094//c/b.root",
    ]


def test_get_files_empty(fake_fs_factory):
    assert get_files(fake_fs_factory({"/c": ["x.txt"]}), "/c") == []


def test_get_friend_files_absolute_base(fake_fs_factory):
    d = "/b/tag/CROWNFriend/fr/2018/A/et"
    fs = fake_fs_factory({d: ["2.root", "1.root", "junk"]})
    assert get_friend_files(fs, "/b", "tag", "fr", "2018", "A", "et") == [
        f"root://server.example:1094/{d}/1.root",
        f"root://server.example:1094/{d}/2.root",
    ]
    assert [c[0] for c in fs.calls] == [d]


def test_get_friend_files_relative_base(fake_fs_factory):
    d = "b/tag/CROWNFriend/fr/2018/A/et"
    fs = fake_fs_factory({d: ["1.root"]})
    assert get_friend_files(fs, "b", "tag", "fr", "2018", "A", "et") == [
        f"root://server.example:1094//{d}/1.root"
    ]


# ------------------------------------------------------------ URLs and upload

SERVER = "root://server.example:1094"


@pytest.mark.parametrize(
    "path, expected",
    [
        (f"{SERVER}//store/f.root", True),
        ("xroot://server//store/f.root", True),
        ("/store/f.root", False),
        ("data/f.root", False),
        (Path("root:/server/f.root"), False),
    ],
)
def test_is_xrootd_url(path, expected):
    assert is_xrootd_url(path) is expected


@pytest.mark.parametrize("server", [SERVER, SERVER + "/"])
@pytest.mark.parametrize("path", ["/store/user/f.root", "store/user/f.root"])
def test_xrootd_url(server, path):
    assert xrootd_url(server, path) == f"{SERVER}//store/user/f.root"


@pytest.mark.parametrize(
    "url",
    [
        f"{SERVER}//store/user/f.root",
        f"{SERVER}/store/user/f.root",
        f"{SERVER}///store/user/f.root",
    ],
)
def test_split_xrootd_url(url):
    assert split_xrootd_url(url) == (SERVER, "/store/user/f.root")


@pytest.mark.parametrize("url", ["/store/f.root", f"{SERVER}", f"{SERVER}/"])
def test_split_xrootd_url_invalid(url):
    with pytest.raises(ValueError, match="Not an XRootD URL"):
        split_xrootd_url(url)


class FakeStatFileSystem:
    """Fake ``FileSystem`` whose ``stat`` returns the given error number."""

    calls = []
    errnos = {}

    def __init__(self, server):
        self.server = server

    def stat(self, path):
        self.calls.append((self.server, path))
        errno = self.errnos.get(path, 0)
        status = SimpleNamespace(ok=errno == 0, errno=errno, message="msg")
        return status, None


@pytest.mark.parametrize(
    "errno, expected", [(0, True), (XRD_NOT_FOUND, False)]
)
def test_xrd_exists(monkeypatch, errno, expected):
    monkeypatch.setattr(FakeStatFileSystem, "errnos", {"/store/f": errno})
    monkeypatch.setattr(FakeStatFileSystem, "calls", [])
    monkeypatch.setattr(xrootd, "FileSystem", FakeStatFileSystem)

    assert xrd_exists(f"{SERVER}//store/f") is expected
    assert FakeStatFileSystem.calls == [(SERVER, "/store/f")]


def test_xrd_exists_other_error_raises(monkeypatch):
    monkeypatch.setattr(FakeStatFileSystem, "errnos", {"/store/f": 3010})
    monkeypatch.setattr(xrootd, "FileSystem", FakeStatFileSystem)

    with pytest.raises(RuntimeError, match="Failed to check"):
        xrd_exists(f"{SERVER}//store/f")


def _status(ok, message="msg"):
    return SimpleNamespace(ok=ok, message=message)


class FakeCopyProcess:
    """Fake ``CopyProcess`` recording its jobs; the outcome of prepare, run
    and the job is set on the class."""

    instances = []
    prepare_ok, run_ok, job_ok = True, True, True

    def __init__(self):
        self.jobs = []
        self.instances.append(self)

    def add_job(self, source, target, **kwargs):
        self.jobs.append((source, target, kwargs))

    def prepare(self):
        return _status(self.prepare_ok, "prepare failed")

    def run(self):
        return _status(self.run_ok, "run failed"), [
            {"status": _status(self.job_ok, "checksum mismatch")}
        ]


@pytest.fixture
def copy_process(monkeypatch):
    monkeypatch.setattr(FakeCopyProcess, "instances", [])
    monkeypatch.setattr(xrootd, "CopyProcess", FakeCopyProcess)
    return FakeCopyProcess


def test_xrd_upload(copy_process, tmp_path):
    xrd_upload(tmp_path / "f.root", f"{SERVER}//store/f.root")

    ((process,),) = [copy_process.instances]
    ((source, target, kwargs),) = process.jobs
    assert (source, target) == (
        str(tmp_path / "f.root"),
        f"{SERVER}//store/f.root",
    )
    # Overwrite, create directories, keep only complete files and compare
    # checksums
    assert kwargs["force"] and kwargs["mkdir"] and kwargs["posc"]
    assert kwargs["checksummode"] == "end2end"
    assert kwargs["checksumtype"] == "adler32"


@pytest.mark.parametrize(
    "failing, message",
    [
        ("prepare_ok", "prepare failed"),
        ("run_ok", "run failed"),
        ("job_ok", "checksum mismatch"),
    ],
)
def test_xrd_upload_failure_raises(
    copy_process, monkeypatch, tmp_path, failing, message, caplog
):
    monkeypatch.setattr(copy_process, failing, False)

    with pytest.raises(RuntimeError, match=f"Failed to copy .*{message}"):
        xrd_upload(tmp_path / "f.root", f"{SERVER}//store/f.root")
    assert "Failed to copy" in caplog.text
