"""Shared fixtures for the friend production tests."""

from types import SimpleNamespace

import awkward as ak  # noqa: F401  (ensures uproot's awkward backend loads)
import numpy as np
import pytest
import uproot

from xyh_ml.friend_production.xrootd import XRD_NOT_FOUND


SERVER_URL = "root://server.example:1094/"


class FakeFileSystem:
    """Minimal stand-in for ``XRootD.client.FileSystem``.

    ``listings`` maps a directory path to the list of entry names it holds.
    Unknown paths yield a failing status. All calls are recorded. ``url`` is
    the server URL, as ``str(FileSystem(...).url)`` returns it (with a
    trailing slash). Unknown paths fail like a missing directory on the
    server (error number 3011), unless ``errors`` maps them to another error
    number.
    """

    def __init__(self, listings, url=SERVER_URL, errors=None):
        self.listings = listings
        self.url = url
        self.errors = errors or {}
        self.calls = []

    def dirlist(self, path, flags=None):
        self.calls.append((path, flags))
        if path not in self.listings:
            status = SimpleNamespace(
                ok=False,
                errno=self.errors.get(path, XRD_NOT_FOUND),
                message=f"no such dir {path}",
            )
            return status, None
        listing = [SimpleNamespace(name=n) for n in self.listings[path]]
        return SimpleNamespace(ok=True, message=""), listing


@pytest.fixture
def fake_fs_factory():
    """Return a factory building a :class:`FakeFileSystem`."""

    def factory(listings=None, errors=None):
        return FakeFileSystem(dict(listings or {}), errors=errors)

    return factory


@pytest.fixture
def make_root_file(tmp_path):
    """Factory writing a tiny ROOT file with a tree called ``ntuple``.

    ``kind`` is ``"filled"`` (default), ``"empty"`` (tree without entries) or
    ``"no_tree"`` (file containing a differently named tree).
    """

    def factory(name, branches=("pt",), entries=3, kind="filled"):
        path = tmp_path / name
        with uproot.recreate(path) as f:
            if kind == "no_tree":
                f["other"] = {"x": np.arange(2, dtype="float64")}
            elif kind == "empty":
                f.mktree("ntuple", {b: "float64" for b in branches})
            else:
                data = {
                    b: np.arange(entries, dtype="float64") for b in branches
                }
                f["ntuple"] = data
        return str(path)

    return factory
