ffmpeg -hide_banner -loglevel warning -fflags discardcorrupt+nobuffer -flags low_delay -rtsp_transport tcp \
-i rtsp://admin:@Pyronear@192.168.1.11:554/h264Preview_01_sub \
-c:v libx264 -bf 0 -g 5 -b:v 500k -r 10 -preset veryfast -tune zerolatency -flush_packets 1 -an -f mpegts \
"srt://91.134.47.14:8890?pkt_size=1316&mode=caller&latency=30&streamid=publish:mateostream"
