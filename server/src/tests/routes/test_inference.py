# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.


import requests


def test_inference(test_app, mocker):

    mocker.patch('app.api.deps.get_current_active_user', return_value="kikou")

    img_content = requests.get("https://pyronear.org/img/logo_letters.png").content

    response = test_app.post("/inference/file/", files=dict(file=img_content))

    assert response.status_code == 201
