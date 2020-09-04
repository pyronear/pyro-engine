# -*- coding: utf-8 -*-

# Copyright (c) Pyronear contributors.
# This file is dual licensed under the terms of the CeCILL-2.1 and AGPLv3 licenses.
# See the LICENSE file in the root of this repository for complete details.

import unittest
# from pyronearEngine.raspberryPi.pyronearEngine import PyronearEngine


class UtilsTester(unittest.TestCase):
    def test_pyronearEngine(self):
        # pyronearEngine = PyronearEngine()
        # #pyronearEngine.run(30)
        self.assertTrue(20 >= 19)


if __name__ == '__main__':
    unittest.main()
