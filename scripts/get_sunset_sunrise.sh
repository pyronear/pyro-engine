#!/bin/bash
# This script must be run with a crontab, run every day at 4am
# 0 4 * * * bash /home/pi/pyro-engine/scripts/update_script.sh

# First obtain a location code from: https://weather.codes/search/
# Insert your location. For example FRXX0076 is a location code for Paris, FRANCE

location="FRXX0076"
tmpfile=/tmp/$location.out

# Obtain sunrise and sunset raw data from weather.com
wget -q "https://weather.com/weather/today/l/$location" -O "$tmpfile"

SUNR=$(grep SunriseSunset "$tmpfile" | grep -oE '((1[0-2]|0?[1-9]):([0-5][0-9]) ?([AaPp][Mm]))' | head -1)
SUNS=$(grep SunriseSunset "$tmpfile" | grep -oE '((1[0-2]|0?[1-9]):([0-5][0-9]) ?([AaPp][Mm]))' | tail -1)


sunrise=$(date --date="$SUNR" +%R)
sunset=$(date --date="$SUNS" +%R)

echo $sunrise > /home/pi/pyro-engine/data/sunset_sunrise.txt
echo $sunset >> /home/pi/pyro-engine/data/sunset_sunrise.txt
