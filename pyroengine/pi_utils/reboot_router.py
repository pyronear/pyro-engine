# Copyright (C) 2020-2022, Pyronear.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

from huawei_lte_api.Client import Client
from huawei_lte_api.AuthorizedConnection import AuthorizedConnection
from dotenv import load_dotenv
import os


"Reboot 4G rooter"
load_dotenv()

ROOTER_LOGIN = os.environ.get("ROOTER_LOGIN")
ROOTER_PASSWORD = os.environ.get("ROOTER_PASSWORD")
ROOTER_IP = os.environ.get("ROOTER_IP")

url = f"http://{ROOTER_LOGIN}:{ROOTER_PASSWORD}@{ROOTER_IP}/"
connection = AuthorizedConnection(url)

client = Client(connection)  # Creat client

client.device.reboot()  # Reboot
