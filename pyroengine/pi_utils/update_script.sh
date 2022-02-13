#!/bin/bash
# This script performs:
# pull origin master
#- if any change:
#    kill container
#	 rebuild docker compose

CID=$(docker ps | grep server_web | awk '{print $1}')

if [ `git -C /home/pi/pyro-engine pull origin master | grep -c "up to date."` -ne 1 ];
    then
        echo "pyro-engine updated from github";
        cd /home/pi/pyro-engine/server/;
        docker kill CID;
        echo "rebuild docker";
        PORT=8002 docker-compose up -d --build;
    else
        echo "pyro-engine up to date";
fi;
