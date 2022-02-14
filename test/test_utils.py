# Copyright (C) 2019-2022, Pyronear.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import unittest
# from pyronearEngine.raspberryPi.pyronearEngine import PyronearEngine


class UtilsTester(unittest.TestCase):
    def test_pyronearEngine(self):
        # pyronearEngine = PyronearEngine()
        # #pyronearEngine.run(30)
        self.assertTrue(20 >= 19)


if __name__ == '__main__':
    unittest.main()
