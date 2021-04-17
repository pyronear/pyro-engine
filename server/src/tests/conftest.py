# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.


import pytest
from starlette.testclient import TestClient

from app.main import app
from app.api.deps import get_current_active_user


@pytest.fixture(scope="module")
def test_app():

    async def override_dependency():
        return {"username": "michel", "disabled": False}

    app.dependency_overrides[get_current_active_user] = override_dependency

    client = TestClient(app)
    yield client  # testing happens here
