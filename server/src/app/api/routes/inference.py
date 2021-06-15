# Copyright (C) 2021, Pyronear contributors.

# This program is licensed under the GNU Affero General Public License version 3.
# See LICENSE or go to <https://www.gnu.org/licenses/agpl-3.0.txt> for full license details.


from fastapi import APIRouter, BackgroundTasks, Request
from fastapi import File, UploadFile
from pyroengine.engine import PyronearEngine
from PIL import Image
import io
import json
import socket


def map_ip_hostname(hostnames):
    """This function map all hostnames with their coresponding ip adress"""
    ip_hostname_mapping = {}
    for hostname in hostnames:
        ip = socket.gethostbyname(hostname + '.local')
        ip_hostname_mapping[ip] = hostname

    return ip_hostname_mapping


def get_hostname_from_ip(ip, ip_hostname_mapping):
    """This function return the hostname for a given ip adress"""
    if ip in ip_hostname_mapping.keys():
        hostname = ip_hostname_mapping[ip]

    else:
        ip_hostname_mapping = map_ip_hostname(ip_hostname_mapping.values())
        hostname = ip_hostname_mapping[ip]

    return hostname, ip_hostname_mapping


def setup_engine():
    with open('config_data.json') as json_file:
        config_data = json.load(json_file)

    # Loading config datas
    detection_threshold = config_data['detection_threshold']
    api_url = config_data['api_url']
    save_evry_n_frame = config_data['save_evry_n_frame']
    latitude = config_data['latitude']
    longitude = config_data['longitude']

    # Loading pi zeros datas
    with open('pi_zeros_data.json') as json_file:
        pi_zeros_data = json.load(json_file)

    pi_zero_credentials = {}
    for hostname in pi_zeros_data.keys():
        d = pi_zeros_data[hostname]
        pi_zero_credentials[d['id']] = {'login': d['login'], 'password': d['password']}

    engine = PyronearEngine(detection_threshold, api_url, pi_zero_credentials, save_evry_n_frame, latitude, longitude)

    return engine, pi_zeros_data


router = APIRouter()
engine, pi_zeros_data = setup_engine()
ip_hostname_mapping = map_ip_hostname(pi_zeros_data.keys())


def predict_and_alert(file, ip):
    """ predict smoke from an image and if yes raise an alert """
    # Load Image
    image = Image.open(io.BytesIO(file))

    # Get hosname
    hostname, ip_hostname_mapping = get_hostname_from_ip(ip, ip_hostname_mapping)

    # Get pi zero id
    pi_zero_id = pi_zeros_data[hostname]['id']

    # Predict
    engine.predict(image, pi_zero_id)


@router.post("/file/", status_code=201, summary="Send img from a device to predict smoke")
async def inference(request: Request,
                    background_tasks: BackgroundTasks,
                    file: UploadFile = File(...)
                    ):
    """
    Get image from pizero and call engine for wildfire detection
    """

    # Call engine as background task
    background_tasks.add_task(predict_and_alert, await file.read(), request.client.host)
