# Copyright (C) 2020-2022, Pyronear.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import sys
import fake_gpiozero
import fake_picamera


sys.modules["gpiozero"] = fake_gpiozero
sys.modules["picamera"] = fake_picamera
