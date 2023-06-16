import json
from requests.auth import HTTPDigestAuth
from io import BytesIO
import requests
from PIL import Image
import os
import time
import multiprocessing
from tqdm import tqdm
from itertools import repeat


def get_img(url):
    
    with open("data/last_img.json") as json_file:
        time_dt = json.load(json_file)
    
    cam_id = url.split('/cgi')[0].split(':')[-1]
    dt = time.time() - time_dt[cam_id]
    if time.time() - time_dt[cam_id] > 30:
        try:
            user, password = url.split('usr=')[1].split('&pwd=')
            response = requests.get(url, auth = HTTPDigestAuth(user, password))

            im = Image.open(BytesIO(response.content))
            assert isinstance(im, Image.Image)
           
            time_dt[cam_id] = time.time()
            with open("data/last_img.json", 'w') as fp:
                json.dump(time_dt, fp)
   
            name = os.path.join("data/last_img",cam_id+".jpg")
            os.makedirs(os.path.dirname(name), exist_ok=True)
            im.save(name, quality=100)
        except:
            pass


with open("data/credentials.json", "rb") as json_file:
    cameras_credentials = json.load(json_file)

 
if not os.path.isfile("data/last_img.json"):
    time_dt = {url.split('/cgi')[0].split(':')[-1]:time.time() for url in cameras_credentials.keys()}
    with open("data/last_img.json", 'w') as fp:
        json.dump(time_dt, fp)


urls = list(cameras_credentials.keys())

while True:
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()-1) as pool:
        results = pool.imap(get_img, urls)
        tuple(results) 
