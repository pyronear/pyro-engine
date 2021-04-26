# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.

import sys
import fake_gpiozero
import fake_picamera


sys.modules['gpiozero'] = fake_gpiozero
sys.modules['picamera'] = fake_picamera
