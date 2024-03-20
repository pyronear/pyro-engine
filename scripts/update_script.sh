#!/bin/bash
# This script performs:
# fetch origin main
#- if any change:
#    pull changes
#    any others change needed
# 
# This script must be run with a crontab, run every hour
# 0 * * * * bash /home/pi/pyro-engine/scripts/update_and_execute.sh >> /home/pi/pyro-engine/logfile.log 2>&1


# Print current date and time
echo "$(date): Checking for updates"

# Navigate to the repository directory
cd /home/pi/pyro-engine

# Check for updates and pull
git fetch origin
HEADHASH=$(git rev-parse HEAD)
UPSTREAMHASH=$(git rev-parse origin/main)

if [ "$HEADHASH" != "$UPSTREAMHASH" ]
then
    echo "$(date): New changes detected ! Updating and executing script..."
    git pull origin main
    # Add any action here
    echo "$(date): Update done !"

else
    echo "$(date): No changes detected"
fi

