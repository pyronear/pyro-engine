#!/bin/bash
# This script performs:
# pull origin main
#- if any change:
#    kill container
#    rebuild docker compose
# 
# This script must be run with a crontab, run every day at 3am
# 0 3 * * * bash /home/pi/pyro-engine/scripts/update_script.sh


if [ `git -C /home/pi/pyro-engine pull origin main | grep -c "up to date."` -ne 1 ];
    then
        echo "pyro-engine up to date";
    else
        echo "pyro-engine updated from github";
        make -C /home/pi/pyro-engine stop
        make -C /home/pi/pyro-engine run
fi;
