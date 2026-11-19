# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from pyro_temporal_api import main


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """Cache under tmp_path, downloads recorded instead of reaching Hugging Face."""
    monkeypatch.setattr(main, "MODEL_CACHE_ROOT", tmp_path / "temporal_models")
    monkeypatch.delenv("TEMPORAL_MODEL_PATH", raising=False)
    calls = []

    def fake_download(repo_id, filename, revision, local_dir):
        calls.append((repo_id, filename, revision))
        (Path(local_dir) / filename).write_bytes(b"zip")

    monkeypatch.setattr(main, "hf_hub_download", fake_download)
    return calls


# docker-compose passes TEMPORAL_MODEL_PATH as an empty string when .env does not set it
@pytest.mark.parametrize("env", [None, ""])
def test_downloads_pinned_model_once(hub, tmp_path, monkeypatch, env):
    if env is not None:
        monkeypatch.setenv("TEMPORAL_MODEL_PATH", env)
    path = main.resolve_model_path()
    assert path == tmp_path / "temporal_models" / main.MODEL_SLUG / main.MODEL_NAME
    assert path.is_file()
    assert hub == [(main.MODEL_REPO_ID, main.MODEL_NAME, main.MODEL_REVISION)]
    assert main.resolve_model_path() == path
    assert len(hub) == 1  # already on disk


@pytest.mark.usefixtures("hub")
def test_new_release_replaces_previous_one(tmp_path):
    old = tmp_path / "temporal_models" / "temporal-model_v0.0.1"
    old.mkdir(parents=True)
    (old / main.MODEL_NAME).write_bytes(b"old")
    assert main.resolve_model_path().is_file()
    assert not old.exists()


def test_custom_path_is_never_downloaded(hub, tmp_path, monkeypatch):
    old = tmp_path / "temporal_models" / "temporal-model_v0.0.1"
    old.mkdir(parents=True)
    custom = tmp_path / "missing.zip"
    monkeypatch.setenv("TEMPORAL_MODEL_PATH", str(custom))
    assert main.resolve_model_path() == custom
    assert hub == []
    assert old.exists()  # the cache is left alone
    with pytest.raises(FileNotFoundError), TestClient(main.app):  # a missing custom package fails the startup
        pass


@pytest.mark.usefixtures("hub")
def test_lifespan_loads_the_resolved_model(monkeypatch):
    loaded = []
    monkeypatch.setattr(main, "read_manifest", lambda _path: {"format_version": 2, "model_version": "test"})
    monkeypatch.setattr(
        main.OnnxTemporalModel, "from_package", lambda path, **_kwargs: loaded.append(path) or SimpleNamespace()
    )
    with TestClient(main.app) as client:
        assert client.get("/health").json() == {"status": "ok", "model_version": "test"}
    assert loaded == [main.MODEL_CACHE_ROOT / main.MODEL_SLUG / main.MODEL_NAME]
