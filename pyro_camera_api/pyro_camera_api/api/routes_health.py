# Copyright (C) 2022-2026, Pyronear.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.


from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health():
    """
    Return a basic heartbeat response to confirm that the API is running.

    This endpoint does not perform any camera or service check.
    It is intended for uptime monitoring and infrastructure-level health probes.
    """
    return {"status": "ok"}
