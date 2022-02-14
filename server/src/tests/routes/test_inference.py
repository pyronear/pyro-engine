# Copyright (C) 2019-2022, Pyronear.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.


import requests


def test_inference(test_app):

    img_content = requests.get("https://pyronear.org/img/logo_letters.png").content

    response = test_app.post("/inference/", files=dict(file=img_content))

    assert response.status_code == 201
