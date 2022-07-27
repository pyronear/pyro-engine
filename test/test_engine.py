# Copyright (C) 2020-2022, Pyronear.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import unittest
from PIL import Image
import io
import requests
from pyroengine.engine import PyronearEngine


image_url = "https://beta.ctvnews.ca/content/dam/ctvnews/images/2020/9/15/1_5105012.jpg?cache_timestamp=1600164224519"

model_url = "https://github.com//pyronear//pyro-vision//releases//download//v0.1.2//yolov5s_v001.onnx"

class EngineTester(unittest.TestCase):
    def test_engine(self):
        # Download model
        r = requests.get(model_url, allow_redirects=True)
        open('model.onnx', 'wb').write(r.content)
        # Init
        engine = PyronearEngine(model_weights='model.onnx')
        # Get Image
        response = requests.get(image_url)
        image_bytes = io.BytesIO(response.content)
        im = Image.open(image_bytes)
        # Predict
        res = engine.predict(im)

        self.assertGreater(res, 0.5)

        # Check backup
        engine.save_cache_to_disk()


if __name__ == "__main__":
    unittest.main()
