from huawei_lte_api.Client import Client
from huawei_lte_api.AuthorizedConnection import AuthorizedConnection
from dotenv import load_dotenv
import os


def reboot_router():
    # Reboot 4G rooter
    load_dotenv()

    ROOTER_LOGIN = os.environ.get("ROOTER_LOGIN")
    ROOTER_PASSWORD = os.environ.get("ROOTER_PASSWORD")
    ROOTER_IP = os.environ.get("ROOTER_IP")

    url = f"http://{ROOTER_LOGIN}:{ROOTER_PASSWORD}@{ROOTER_IP}/"
    connection = AuthorizedConnection(url)

    client = Client(connection)  # Creat client

    client.device.reboot()  # Reboot
