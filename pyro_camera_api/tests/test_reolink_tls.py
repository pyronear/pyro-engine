# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

from pyro_camera_api.camera.adapters.reolink import _session


def test_https_session_offers_rsa_key_exchange_ciphers():
    # Old Reolink firmwares only support RSA-key-exchange ciphers, which
    # Python 3.10+ removed from its defaults. The mounted adapter must offer them.
    adapter = _session.get_adapter("https://192.168.1.10")
    ctx = adapter.poolmanager.connection_pool_kw["ssl_context"]
    names = {c["name"] for c in ctx.get_ciphers()}
    assert "AES128-GCM-SHA256" in names
