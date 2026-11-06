"""Tests for the files of jobs at sites without shared file systems."""

import subprocess
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from xyh_ml.friend_production import sandbox

URL = "root://server//store/sandboxes"


@pytest.fixture
def project(tmp_path):
    """A repository with code, config and a lock file."""
    project_dir = tmp_path / "repo"
    for name, content in {
        "src/xyh_ml/__init__.py": "code",
        "src/xyh_ml/__pycache__/__init__.pyc": "cache",
        "config/produce.yaml": "config",
        "notebooks_dev/x.ipynb": "not shipped",
        "pixi.lock": "lock",
    }.items():
        path = project_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return project_dir


@pytest.fixture
def models(tmp_path):
    directory = tmp_path / "training" / "fold0of5" / "et"
    directory.mkdir(parents=True)
    for name in [*sandbox.MODEL_FILES, "metrics.csv"]:
        (directory / name).write_text(name)
    return {"fold0of5/et": directory}


@pytest.fixture
def metadata(tmp_path):
    path = tmp_path / "meta" / "DY.json.gz"
    path.parent.mkdir()
    path.write_text("metadata")
    return {"2024/DY.json.gz": path}


@pytest.fixture
def storage(monkeypatch):
    """Grid storage with the uploaded content per URL."""
    files = {}
    monkeypatch.setattr(sandbox, "xrd_exists", lambda url: url in files)

    def upload(local_file, url):
        files[url] = local_file.read_bytes()

    monkeypatch.setattr(sandbox, "xrd_upload", upload)
    return files


@pytest.fixture
def fake_pixi_pack(monkeypatch, tmp_path):
    """Fake binary, recording the commands and writing the output file."""
    binary = tmp_path / "bin" / "pixi-pack"
    binary.parent.mkdir()
    binary.write_text("")
    monkeypatch.setattr(sandbox.sys, "executable", str(binary.parent / "py"))
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if "--output-file" in command:
            output = command[command.index("--output-file") + 1]
            with open(output, "w") as f:
                f.write("pack")
        return SimpleNamespace(
            stdout="pixi-pack 0.7.11", stderr="", returncode=0
        )

    monkeypatch.setattr(sandbox.subprocess, "run", run)
    return calls


# ------------------------------------------------------------ environment pack


def test_environment_pack_builds_and_uploads(project, storage, fake_pixi_pack):
    url = sandbox.environment_pack(URL, project)

    assert url.startswith(f"{URL}/env-cpu-")
    assert url.endswith(".sh")
    assert storage == {url: b"pack"}
    command = fake_pixi_pack[-1]
    assert command[1] == str(project / "pyproject.toml")
    assert command[command.index("--environment") + 1] == "cpu"
    assert "--create-executable" in command
    assert "--ignore-pypi-non-wheel" in command


def test_environment_pack_reused(project, storage, fake_pixi_pack):
    url = sandbox.environment_pack(URL, project)
    n_calls = len(fake_pixi_pack)
    storage[url] = b"from an earlier run"

    assert sandbox.environment_pack(URL, project) == url

    # Only the version is asked for, and the file is left as it is
    assert len(fake_pixi_pack) == n_calls + 1
    assert fake_pixi_pack[-1][1:] == ["--version"]
    assert storage[url] == b"from an earlier run"


def test_environment_pack_name_changes_with_lock_file(
    project, storage, fake_pixi_pack
):
    url = sandbox.environment_pack(URL, project)
    (project / "pixi.lock").write_text("other lock")

    assert sandbox.environment_pack(URL, project) != url
    assert len(storage) == 2


def test_environment_pack_failure(
    project, storage, fake_pixi_pack, monkeypatch
):
    def run(command, **kwargs):
        failed = "--output-file" in command
        return SimpleNamespace(
            stdout="pixi-pack 0.7.11",
            stderr="no network" if failed else "",
            returncode=int(failed),
        )

    monkeypatch.setattr(sandbox.subprocess, "run", run)

    with pytest.raises(sandbox.SandboxError, match="no network"):
        sandbox.environment_pack(URL, project)
    assert storage == {}


def test_environment_pack_timeout(
    project, storage, fake_pixi_pack, monkeypatch
):
    def run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 1)

    monkeypatch.setattr(sandbox.subprocess, "run", run)

    with pytest.raises(sandbox.SandboxError, match="failed"):
        sandbox.environment_pack(URL, project)


def test_pixi_pack_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(sandbox.sys, "executable", str(tmp_path / "python"))

    with pytest.raises(sandbox.SandboxError, match="pixi install"):
        sandbox.pixi_pack_binary()


def test_missing_lock_file(tmp_path, fake_pixi_pack):
    with pytest.raises(sandbox.SandboxError, match="lock file"):
        sandbox.environment_pack_name(tmp_path, tmp_path / "bin" / "pixi-pack")


def test_source_distribution_in_lock_file(project, storage, fake_pixi_pack):
    url = "https://files.pythonhosted.org/packages/antlr4-4.9.3.tar.gz"
    (project / "pixi.lock").write_text(
        f"""environments:
  cpu:
    packages:
      - pypi: https://files.pythonhosted.org/packages/a-1.0-py3-none-any.whl
      - pypi: {url}
      - pypi: ./
"""
    )

    with pytest.raises(sandbox.SandboxError, match="antlr4-4.9.3.tar.gz"):
        sandbox.environment_pack(URL, project)
    assert storage == {}


def test_wheels_and_editable_install_in_lock_file():
    sandbox.check_no_source_distributions(
        """indexes:
- https://pypi.org/simple
packages:
  - pypi: https://files.pythonhosted.org/packages/a-1.0-py3-none-any.whl
  - pypi: ./
"""
    )


def test_lock_file_of_the_repository_can_be_packed():
    sandbox.check_no_source_distributions(
        (Path(__file__).parents[2] / "pixi.lock").read_text()
    )


def test_storage_failure_is_reported(project, fake_pixi_pack, monkeypatch):
    def fail(url):
        raise RuntimeError("server down")

    monkeypatch.setattr(sandbox, "xrd_exists", fail)

    with pytest.raises(sandbox.SandboxError, match="server down"):
        sandbox.environment_pack(URL, project)


# --------------------------------------------------------------------- bundle


def test_bundle_content(project, models, metadata):
    content = sandbox.bundle_content(project, models, metadata)

    assert set(content) == {
        "src/xyh_ml/__init__.py",
        "config/produce.yaml",
        "models/fold0of5/et/transformations.pkl",
        "models/fold0of5/et/best_model.ckpt",
        "metadata/2024/DY.json.gz",
    }


def test_bundle_content_missing_model(project, models, metadata):
    directory = models["fold0of5/et"]
    (directory / "best_model.ckpt").unlink()

    with pytest.raises(sandbox.SandboxError, match="best_model.ckpt"):
        sandbox.bundle_content(project, models, metadata)


def test_bundle_content_missing_metadata(project, models, tmp_path):
    with pytest.raises(sandbox.SandboxError, match="gone.json.gz"):
        sandbox.bundle_content(
            project, models, {"2024/gone.json.gz": tmp_path / "gone.json.gz"}
        )


def test_write_bundle_can_be_unpacked(project, models, metadata, tmp_path):
    content = sandbox.bundle_content(project, models, metadata)
    path = tmp_path / "bundle.tar.gz"

    sandbox.write_bundle(content, path)

    with tarfile.open(path) as archive:
        assert sorted(archive.getnames()) == sorted(content)
        member = archive.extractfile("metadata/2024/DY.json.gz")
        assert member.read() == b"metadata"


def test_job_bundle_uploads_with_hash_name(project, models, metadata, storage):
    url = sandbox.job_bundle(URL, project, models, metadata)

    assert url.startswith(f"{URL}/bundle-")
    assert url.endswith(".tar.gz")
    assert list(storage) == [url]


def test_job_bundle_same_content_same_name(project, models, metadata, storage):
    url = sandbox.job_bundle(URL, project, models, metadata)
    # File times and order do not change the result
    (project / "config" / "produce.yaml").touch()
    storage[url] = b"from an earlier run"

    assert sandbox.job_bundle(URL, project, models, metadata) == url
    assert storage[url] == b"from an earlier run"


def test_job_bundle_new_content_new_name(project, models, metadata, storage):
    url = sandbox.job_bundle(URL, project, models, metadata)
    (project / "config" / "produce.yaml").write_text("changed")

    assert sandbox.job_bundle(URL, project, models, metadata) != url
    assert len(storage) == 2


def test_bundle_ignores_metrics(project, models, metadata):
    content = sandbox.bundle_content(project, models, metadata)

    assert not any(name.endswith("metrics.csv") for name in content)
