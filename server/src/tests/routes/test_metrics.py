# Copyright (C) 2020-2022, Pyronear.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.


import json
import pytest


PARTIAL_PAYLOAD = {
    "id": 1,
    "created_at": "2021-04-01T01:12:34.567890",
    "mem_available_GB": 1.23,
    "cpu_usage_percent": 51.8
}


@pytest.mark.parametrize(
    "payload, cpu_temperature_C, status_code",
    [
        [PARTIAL_PAYLOAD, 17.3, 201],
        [PARTIAL_PAYLOAD, "cold", 422],
    ],
)
def test_log_metrics(test_app, payload, cpu_temperature_C, status_code):

    payload["cpu_temperature_C"] = cpu_temperature_C

    response = test_app.post("/metrics/", data=json.dumps(payload))

    assert response.status_code == status_code
