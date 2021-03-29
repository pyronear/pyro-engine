import json
import requests
import os


def test_inference(test_app):

    img_content = requests.get("https://pyronear.org/img/logo_letters.png").content

    response = test_app.post("/inference/", files=dict(file=img_content))

    assert response.status_code == 201
