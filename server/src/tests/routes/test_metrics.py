# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.


import json

PAYLOAD = {
    "id": 1,
    "created_at": "2021-03-26T16:35:09.656Z",
    "cpu_temperature_C": 17.3,
    "mem_available_GB": 1.23,
    "cpu_usage_percent": 51.8
}


def test_send_metrics(test_app):

    response = test_app.post("/metrics/", data=json.dumps(PAYLOAD))

    assert response.status_code == 201


BAD_PAYLOAD = {
    "id": 1,
    "created_at": "2021-03-26T16:35:09.656Z",
    "cpu_temperature_C": "cold",
    "mem_available_GB": 1.23,
    "cpu_usage_percent": 51.8
}


def test_send_metrics_bad_type(test_app):

    response = test_app.post("/metrics/", data=json.dumps(BAD_PAYLOAD))

    assert response.status_code == 422
