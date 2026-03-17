#!/bin/bash
set -euo pipefail

IMAGE_NAME="openbots"
NO_CACHE="false"

if [[ "${1:-}" == "--no-cache" ]]; then
    NO_CACHE="true"
fi

echo "Initializing Git submodules..."
git submodule update --init --recursive

BUILD_CMD=(docker build --network=host -t "$IMAGE_NAME" -f Dockerfile .)
if [[ "$NO_CACHE" == "true" ]]; then
    BUILD_CMD=(docker build --network=host --pull --no-cache -t "$IMAGE_NAME" -f Dockerfile .)
fi

echo "Building Docker image: $IMAGE_NAME"
echo "Command: ${BUILD_CMD[*]}"
"${BUILD_CMD[@]}"

echo "Docker image '$IMAGE_NAME' built successfully!"
