#!/bin/bash

# Default port
PORT=${1:-8501}

# Build the Docker image
docker build --build-arg DEFAULT_PORT=${PORT} -t oil-palm-detect .

# Run the Docker container, mapping the specified or default port
docker run -p ${PORT}:${PORT} -e PORT=${PORT} oil-palm-detect
