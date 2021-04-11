import io
import time
import picamera
from PIL import Image
import requests


url = "http://192.168.1.62:8000/file" # api url
stream = io.BytesIO()
with picamera.PiCamera() as camera:
    camera.resolution = (3280, 2464) #use maximal resolution
    while(True):
        print('capture')
        stream = io.BytesIO()
        camera.start_preview()
        time.sleep(3) # small sleep here improve image quality
        camera.capture(stream, format='jpeg')
        # "Rewind" the stream to the beginning so we can read its content
        stream.seek(0)
        files = {'file': stream}
        requests.post(url, files=files) #send image to pi_cental

        time.sleep(30) #Wait between two capture