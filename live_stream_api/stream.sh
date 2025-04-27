ffmpeg -fflags discardcorrupt+nobuffer -flags low_delay -rtsp_transport tcp \
-i rtsp://admin:@Pyronear@169.254.40.1:554/h264Preview_01_sub \
-c:v libx264 -bf 0 -b:v 300k -r 10 -preset veryfast -tune zerolatency -an -f mpegts \
'srt://91.134.47.14:8890?pkt_size=1316&mode=caller&latency=500&streamid=publish:mateostream'
