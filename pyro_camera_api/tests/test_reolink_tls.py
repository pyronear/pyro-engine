# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import pytest
import requests

from pyro_camera_api.camera.adapters.reolink import _session


def test_https_session_offers_rsa_key_exchange_ciphers():
    # Old Reolink firmwares only support RSA-key-exchange ciphers, which
    # Python 3.10+ removed from its defaults. The mounted adapter must offer them.
    adapter = _session.get_adapter("https://192.168.1.10")
    ctx = adapter.poolmanager.connection_pool_kw["ssl_context"]
    names = {c["name"] for c in ctx.get_ciphers()}
    assert "AES128-GCM-SHA256" in names
    # Proxy env vars must not reroute camera calls around the TLS adapter
    assert not _session.trust_env


def test_session_applies_default_timeout(monkeypatch):
    # A call without timeout must not wait forever on a silent camera (pyro-engine#422)
    seen = []

    def fake_send(self, request, **kwargs):
        seen.append(kwargs["timeout"])
        raise requests.ConnectionError

    monkeypatch.setattr("requests.adapters.HTTPAdapter.send", fake_send)
    for url in ("https://192.168.1.10", "http://192.168.1.10"):
        for kwargs in ({}, {"timeout": 2}):
            with pytest.raises(requests.ConnectionError):
                _session.post(url, **kwargs)
    assert seen == [10, 2, 10, 2]
