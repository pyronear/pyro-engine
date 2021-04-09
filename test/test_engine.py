# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.

import unittest
from PIL import Image
import io
import requests
from pyroengine.engine import PyronearEngine


url = "https://beta.ctvnews.ca/content/dam/ctvnews/images/2020/9/15/1_5105012.jpg?cache_timestamp=1600164224519"


class EngineTester(unittest.TestCase):

    def test_engine(self):
        # Init
        engine = PyronearEngine()
        # Get Image
        response = requests.get(url)
        image_bytes = io.BytesIO(response.content)
        im = Image.open(image_bytes)
        # Predict
        res = engine.predict(im)

        self.assertGreater(res, 0.5)


if __name__ == '__main__':
    unittest.main()
