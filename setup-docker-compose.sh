#!/bin/bash

# Define the percentage of host memory you want to allocate
PERCENTAGE=50

# Get the total memory of the host system in kilobytes
TOTAL_MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')

# Calculate the memory limit in kilobytes
LIMIT_MEM_KB=$((TOTAL_MEM_KB * PERCENTAGE / 100))

# Convert the limit to a format Docker understands (e.g., "m" for megabytes)
LIMIT_MEM_MB=$((LIMIT_MEM_KB / 1024))m

# Define the Docker Compose file to modify
DOCKER_COMPOSE_FILE="docker-compose.yml"

# Backup the original Docker Compose file
cp $DOCKER_COMPOSE_FILE "${DOCKER_COMPOSE_FILE}.bak"

# Use awk to update the memory limits in the Docker Compose file, preserving indentation
awk -v mem_limit="$LIMIT_MEM_MB" '
/services:/ { in_services=1 }
in_services && /deploy:/ { in_deploy=1 }
in_deploy && /resources:/ { in_resources=1 }
in_resources && /limits:/ { in_limits=1 }
in_limits && /memory:/ {
    $0 = gensub(/memory:.*/, "memory: " mem_limit, 1)
}
{ print }
' "${DOCKER_COMPOSE_FILE}.bak" > $DOCKER_COMPOSE_FILE

echo "Memory limits set to $LIMIT_MEM_MB in $DOCKER_COMPOSE_FILE"
