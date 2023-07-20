import json
from requests.auth import HTTPDigestAuth
from io import BytesIO
import requests
from PIL import Image
import os
import time
import multiprocessing as mp
from tqdm import tqdm
from itertools import repeat
import signal

def handler():
    raise Exception("Analyze stream timeout")

def get_img(url, q):
    st = time.time()
    cam_id = url.split('/cgi')[0].split(':')[-1]

    with open("data/last_img.json") as json_file:
        time_dt = json.load(json_file)

    last, status = time_dt[cam_id]
    if status =="available":
        
    
        dt = time.time() - last

        if dt >200:
            name = os.path.join("data/last_img",cam_id+".jpg")
            if os.path.isfile(name):
                os.remove(name)

        if dt > 20:
            time_dt[cam_id] = (last, "busy")
            with open("data/last_img.json", 'w') as fp:
                json.dump(time_dt, fp)
            try:
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(60)

                user, password = url.split('usr=')[1].split('&pwd=')
                response = requests.get(url, auth = HTTPDigestAuth(user, password))

                im = Image.open(BytesIO(response.content))
                print(im.size)
                assert isinstance(im, Image.Image)
            
                time_dt[cam_id] = (time.time(), "available")
                with open("data/last_img.json", 'w') as fp:
                    json.dump(time_dt, fp)

                os.makedirs(os.path.dirname(name), exist_ok=True)
                im.save(name, quality=100)
                signal.alarm(0)
            except:
                time_dt[cam_id] = (last, "available")
                with open("data/last_img.json", 'w') as fp:
                    json.dump(time_dt, fp)
            print(url, dt, time.time()-st)



if __name__ == "__main__":

    with open("data/credentials.json", "rb") as json_file:
        cameras_credentials = json.load(json_file)

    
    if os.path.isfile("data/last_img.json"):
        os.remove("data/last_img.json")
    time_dt = {url.split('/cgi')[0].split(':')[-1]:(time.time()-30, "available") for url in cameras_credentials.keys()}
    with open("data/last_img.json", 'w') as fp:
        json.dump(time_dt, fp)


    urls = list(cameras_credentials.keys())

    manager = mp.Manager()
    pool = mp.Pool(3)
    q = manager.Queue() 

    idx = 0
    while True:
        
        pool.apply_async(get_img, (urls[idx],q))
        idx+=1
        idx = idx%len(urls)
        #print(urls[idx])
      
# while True:
#     with multiprocessing.Pool(processes=multiprocessing.cpu_count()-1) as pool:
#         results = pool.imap(get_img, urls)
#         tuple(results) 
