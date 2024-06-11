#!/bin/bash

# Define the percentage of host memory you want to allocate
PERCENTAGE=90

# Get the total memory of the host system in kilobytes
TOTAL_MEM_KB=$(grep MemTotal /proc/meminfo | awk '{print $2}')

# Check if TOTAL_MEM_KB was successfully retrieved
if [ -z "$TOTAL_MEM_KB" ]; then
    echo "Failed to retrieve total memory."
    exit 1
fi

# Calculate the memory limit in kilobytes
LIMIT_MEM_KB=$((TOTAL_MEM_KB * PERCENTAGE / 100))

# Convert the limit to a format Docker understands (e.g., "m" for megabytes)
LIMIT_MEM_MB=$((LIMIT_MEM_KB / 1024))m

# Define the Docker Compose override file to create/update
DOCKER_COMPOSE_OVERRIDE_FILE="docker-compose.override.yml"

# Create/update the docker-compose.override.yml with the memory limit
cat <<EOF > "$DOCKER_COMPOSE_OVERRIDE_FILE"
services:
  run:
    deploy:
      resources:
        limits:
          memory: $LIMIT_MEM_MB
EOF

echo "Memory limits set to $LIMIT_MEM_MB in $DOCKER_COMPOSE_OVERRIDE_FILE"
