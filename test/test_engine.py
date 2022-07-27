# Copyright (C) 2020-2022, Pyronear.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import unittest
from PIL import Image
import io
import requests
import tempfile
from pyroengine.engine import PyronearEngine


image_url = (
    "https://bloximages.newyork1.vip.townnews.com/union-bulletin.com/content/tncms/assets/v3/editorial/6/86/"
    "68647f68-f036-11eb-a656-93dbdb9cc4b8/61024b72a57c7.image.jpg?resize=666%2C500"
)

model_url = "https://github.com//pyronear//pyro-vision//releases//download//v0.1.2//yolov5s_v001.onnx"


class EngineTester(unittest.TestCase):
    def test_engine(self):
        with tempfile.TemporaryDirectory() as root:
            # Download model
            r = requests.get(model_url, allow_redirects=True)
            model_path = root + "/model.onnx"
            open(model_path, "wb").write(r.content)
            # Init
            engine = PyronearEngine(model_weights=model_path)
            # Get Image
            response = requests.get(image_url)
            image_bytes = io.BytesIO(response.content)
            im = Image.open(image_bytes)
            # Predict
            res = engine.predict(im)
            print(res)

            self.assertGreater(res, 0.25)

            # Check backup
            engine.save_cache_to_disk()


if __name__ == "__main__":
    unittest.main()
